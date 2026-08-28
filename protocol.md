# Qwen token line protocol

状态：协议版本 0.1，C++ server/client 已实现，Python 尚未接入。

这是 Python 与 C++ 推理引擎之间的本地协议。它借用 HTTP 的 request line、
status line 和参数形式，但不是 HTTP/1.1：没有 `Host`、`Content-Type`、
`Content-Length`、chunked encoding 或 `Connection` header。

```text
Python: tokenizer / OpenAI API
              |
              | prompt token IDs
              v
Unix socket -> C++ Engine + Session pool
              |
              | completion token IDs
              v
Python: decode / JSON / SSE
```

## 1. 连接

- 传输使用 `AF_UNIX` + `SOCK_STREAM`。
- 一个连接只处理一个 request 和一个 response。
- 所有行使用 Unix 换行 `LF` (`\n`)，不使用 `CRLF`。
- request 和 response 均使用空行分隔参数与 token，再用空行结束 token。
- C++ 完成 response 后关闭连接。
- socket 是字节流；一次 `read()` 或 `write()` 的边界没有协议语义。

预期命令行形式：

```sh
# 加载一次模型并监听 Unix socket。
./qwen35 -l qwen35.sock -m qwen35.bin --parallel 4

# 本地 client 从 stdin 读取 token IDs。
printf '123 456 789\n' | ./qwen35 qwen35.sock -n 64
```

## 2. 通用行格式

参数行使用：

```text
name: value\n
```

- 参数名使用小写 ASCII 和 `-`。
- 参数名区分大小写。
- 重复参数非法。
- 未知参数非法，避免拼写错误被静默忽略。

token 行使用：

```text
token-id *(1*SP token-id) LF
```

每个非空行可以包含一个或多个十进制 token ID，token 之间使用一个或
多个 ASCII 空格分隔。空行只用于结束 section，不属于 token 内容。

```text
123 456
789
```

解析结果是：

```text
[123, 456, 789]
```

## 3. Request

唯一 endpoint 是：

```text
POST /infer
```

完整 request：

```text
POST /infer
max-tokens: 64
temperature: 0
top-k: 0
top-p: 1
presence-penalty: 0
seed: 42

123 456
789

```

结构是：

```text
request line
request parameters
empty line
prompt token lines
empty line
```

### 3.1 Request parameters

| 参数 | 必需 | 默认值 | 约束 |
| --- | --- | --- | --- |
| `max-tokens` | 是 | - | 最大生成 token 数，正整数 |
| `temperature` | 否 | `0` | `[0, 2]`；`0` 表示 greedy |
| `top-k` | 否 | `0` | 非负整数；`0` 表示不限制 |
| `top-p` | 否 | `1` | `(0, 1]` |
| `presence-penalty` | 否 | `0` | `[-2, 2]` |
| `seed` | 否 | `0` | unsigned 64-bit integer |

因此最小 request 是：

```text
POST /infer
max-tokens: 64

123 456 789

```

### 3.2 Prompt tokens

- prompt 至少包含一个 token ID。
- 每个 token ID 必须满足 `0 <= token-id < vocab-size`。
- `prompt-tokens + max-tokens` 不能超过 server context size。
- EOF 不能代替最后的空行；提前断开表示 request 不完整或取消。

## 4. Session 等待

C++ 验证 request 后从内部 Session pool 获取 Session。如果所有 Session 都在使用，
C++ 保持连接并等待，不返回 busy error。

```text
request -> validate -> wait for Session -> sync/prefill -> response
```

Python 可以自行管理并发限制、返回 429、设置超时或关闭 socket。

## 5. Successful response

C++ 获得 Session 并完成 `sync/prefill` 后开始 response：

```text
200 OK
cached-tokens: 128

9707 198
13

```

结构是：

```text
status line
response parameters
empty line
completion token lines
empty line
```

### 5.1 Response parameters

`cached-tokens` 是必需的非负整数，表示本次 prompt 中直接复用已有
Session state、没有重新执行 forward 的前缀 token 数：

```text
0 <= cached-tokens <= prompt-tokens
```

Python 可将它映射到 OpenAI-compatible usage 中的
`prompt_tokens_details.cached_tokens`。

### 5.2 Completion tokens

- completion 允许为空，表示模型立即生成 stop token。
- 模型生成的 stop token 本身不写入 response。
- C++ 可以每次写一个 token，也可以在一行中批量写入多个 token。
- client 读到 token section 的结束空行后，认为生成正常完成。

空 completion 的原始格式是三个连续换行：

```text
200 OK\ncached-tokens: 64\n\n\n
```

第一个空行结束 response parameters，第二个空行结束空的 token section。

## 6. Finish reason

协议不显式返回 finish reason。Python 根据已返回 token 数与 request 中的
`max-tokens` 推断：

```text
completion-tokens < max-tokens  -> stop
completion-tokens == max-tokens -> length
```

这个判断无歧义：如果某次采样得到 stop token，C++ 不会返回它，所以已返回
token 数一定小于 `max-tokens`。

## 7. Error response

开始返回 token 前发生错误，C++ 返回非 `200` status：

```text
400 Bad Request
message: invalid token ID


```

建议状态：

| Status | 含义 |
| --- | --- |
| `400 Bad Request` | request line、参数或 token ID 非法 |
| `404 Not Found` | endpoint 不存在 |
| `405 Method Not Allowed` | method 不是 `POST` |
| `413 Context Too Large` | prompt 或完整 context 超过限制 |
| `500 Internal Error` | 返回 token 前发生内部错误 |

error response 的 token section 为空，因此 response parameters 后面连续使用两个空行。

已经返回 `200 OK` 和部分 token 后发生错误，C++ 直接关闭 socket，不写入最后的
结束空行。client 将它视为不完整响应，而不是正常 stop。

## 8. Cancellation

协议没有 cancel endpoint。client 取消时直接关闭 socket。

C++ 检测到断开后：

1. 如果正在等待 Session，停止等待。
2. 如果正在生成，在当前 forward/token 边界停止。
3. 清空并释放已获得的 Session。
4. 继续服务其他连接。

## 9. Non-goals

协议 0.1 不包含：

- HTTP/1.1 兼容性或标准 HTTP client 支持。
- JSON、tokenizer 或 chat template。
- request ID 或 session ID。
- 一个连接上的多路复用。
- busy error、显式队列 ID 或队列管理 endpoint。
- 显式 finish reason。
- stop string、tools、vision 或 thinking 分离。
- 模型加载、卸载或切换 endpoint；一个 server 进程固定加载一个模型。
