# qwen3x

## 目的

这是一个用于 PoC 的极简、本地优先 Qwen C++ 推理引擎，并直接提供
OpenAI-compatible HTTP API。

## 原则

- 尽量用少量、直接的 C++ 文件完成完整数据流，只做不影响阅读的性能优化。
- 代码采用 C-oriented、exception-free C++17：模型计算使用数组、指针、循环和显式
  shape；C++ 主要用于 namespace、RAII、并发和动态存储，不使用异常、RTTI 或复杂模板。
- 项目优先级是 `correct -> simple -> readable -> usable -> fast`。

## 数据流

运行时完全位于一个 C++ 进程：

```text
HTTP JSON
  -> parser.cpp: OpenAI request -> plain structs
  -> render.cpp: chat template -> tokenizer -> token IDs
  -> runtime.cpp / engine.cpp: prefill -> decode -> token IDs
  -> render.cpp: decode
  -> parser.cpp: JSON / SSE
```

Python 不参与部署或推理，只用于离线权重转换、render 数据生成、eval 和官方 reference。

## 使用

日常只需要两个终端。

终端一，在 qwen3x 中启动 9B Q8_0 服务：

```sh
make serve-9b
```

终端二，进入希望 pi 操作的项目，然后运行 qwen3x 自带的启动脚本：

```sh
cd /path/to/project
/path/to/qwen3x/scripts/pi.sh
```

`pi.sh` 保持当前工作目录不变，自动创建隔离的 pi 配置，检查 qwen3x 是否 ready，并从
`/v1/models` 选择当前服务实际加载的 4B 或 9B 模型，以默认 thinking 启动 pi。配置模板是
[`scripts/pi-models.json`](scripts/pi-models.json)；40k context 使用的压缩余量在
[`scripts/pi-settings.json`](scripts/pi-settings.json)。

首次使用需要先准备模型：

```sh
make model-9b
```

它下载固定 revision 的官方 Qwen3.5-9B BF16 checkpoint，并在 `build/` 直接生成约
8.86 GiB 的 9B Q8_0 model bin，以及共享的 Qwen3.5 render 数据。量化只改变权重；activation、
recurrent state、KV cache、workspace 和 logits 仍是 FP32。`make serve-9b` 会检查产物并在
缺失时提示运行上述命令。环境需要 CUDA Toolkit、
[uv](https://docs.astral.sh/uv/)、Node >= 22.19 和 pi 0.84.4。

当前路线在 RTX 4080 SUPER 16 GiB、CUDA 12.8、Node 22.19.0 上验证，使用 40960
context。pi 可以流式 thinking 和 tool calls，完成分段读文件、review diff、修改文件和
运行测试；真实修改仍应限定范围并人工 review。

9B 在该机器上的 model、State 和 workspace 已知分配合计 11.538 GiB；进程实测增加
11.793 GiB 显存。4K prompt prefill 为 8.90 秒（460 tok/s），随后 decode 为 25.9 tok/s；
16K prompt prefill 为 56.54 秒（290 tok/s），随后 decode 为 21.5 tok/s。结果记录在
[`eval/q8-9b.md`](eval/q8-9b.md)。

显存更小或只想快速迭代时仍可使用现有 4B BF16 路径：

```sh
make model-4b
make serve-4b
```

不用 pi 时可以直接请求同一个服务：

```sh
scripts/chat.py -m qwen3.5-9b -u "你好"
```

或者使用 OpenAI-compatible HTTP：

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-9b","messages":[{"role":"user","content":"你好"}]}'
```

完整启动参数可以用 `make -n serve-9b` 查看。`serve-4b` 和 `serve-9b` 都固定监听
`127.0.0.1:8000`，使用一个 40960-token Session；需要其他参数时直接运行
`build/qwen35-cuda`。

## 开发和其他接口

0.8B 作为 CPU correctness 和协议 smoke 基线：

```sh
make -C scripts model render
make -j4
./build/qwen35 --prompt "hello" --max-tokens 128
```

日常请求可以使用零依赖的薄客户端，不需要手写 JSON：

```sh
scripts/chat.py -m qwen3.5-4b -u "你好"
scripts/chat.py -m qwen3.5-4b -s "回答简短" -u "你好" -a "你好！" -u "你是谁？"
```

`-c ID NAME ARGUMENTS` 表示 assistant tool call，`-t ID CONTENT` 用同一个
ID 返回工具结果。连续的 `-c` 属于同一条 assistant 消息：

```sh
scripts/chat.py \
  -m qwen3.5-4b \
  -u "杭州天气怎么样？" \
  -c call_weather weather '{"city":"杭州"}' \
  -t call_weather "晴，28°C"
```

`ARGUMENTS` 也可以写成 `@args.json`。增加 `--dry-run` 只生成可直接执行的
curl，不发送请求：

```sh
scripts/chat.py -m qwen3.5-4b -u "你好" --dry-run
```

用最小 Agent 闭环测试原生工具调用：

```sh
scripts/agent.py -y "Use bash to run pwd, then tell me the directory."
```

`agent.py` 只提供 `read_file`、`write_file` 和 `bash`，默认在当前目录工作；
`write_file` 和 `bash` 默认需要确认，`-y` 用于受控测试环境。服务支持普通文本与
tool-call SSE，包括同一轮多个 tool calls、usage 和 `finish_reason: tool_calls`。
带 tools 的流式请求会在模型完成后一次发送解析后的 tool-call chunks，避免把 Qwen
内部 XML 暴露为 content；因此工具调用没有逐参数增量输出。
`read_file` 默认读取 200 行，也接受 `start_line` 和 `line_count`，让模型按段阅读大文件。

测量不含 HTTP、JSON、chat template、tokenizer 和 sampling 的 Session 性能：

```sh
./build/qwen35-cuda \
  --model build/qwen35-4b-model.bin \
  --bench 4096 32 --session-context 40960
```

两个数字依次是 prefill token 数和 decode token 数。默认 Session context 是两者
之和，也可以用 `--session-context` 显式指定更大的容量。benchmark 默认创建一个
session slot；`--session-slots` 可以复现服务所用的内存配置，但计时仍只推进其中一个
Session，不代表并发吞吐量。

`qwen35` 默认从可执行文件所在目录加载 `qwen35-0.8b-model.bin` 和
`qwen35-0.8b-render.bin`。其他型号通过 `--model` 选择对应 model bin；
`--render` 只在需要覆盖默认 tokenizer 数据时使用。

普通 completion：

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.5-4b",
    "messages":[{"role":"user","content":"你好"}],
    "temperature":0,
    "max_completion_tokens":128
  }'
```

流式 completion 增加：

```json
{"stream":true,"stream_options":{"include_usage":true}}
```

服务还提供 `/healthz`、`/readyz` 和 `/v1/models`。`--listen` 默认使用 `info`
日志，prompt 和 benchmark 默认使用 `error`；显式 `--log-level` 会覆盖模式默认值。
指定 `--log-file build/qwen35.log` 后，滚动文件会替代 stderr 成为日志输出位置。

`make serve-4b` 和 `make serve-9b` 默认把完整 audit 分别写入独立的本地文件：

```sh
build/qwen35-audit.log
build/qwen35-9b-audit.log
```

audit 按事件记录原始请求、render prompt、模型输出、工具解析和实际 HTTP/SSE 输出。
`request_id` 关联完整请求，`session_id` 关联复用同一个缓存 Session 的请求；服务同时通过
`X-Request-Id` 和 `X-Session-Id` 响应头输出这两个 ID。文件权限为 `0600`，内容包含完整
对话和工具参数。手动启动 `qwen35` 时 audit 默认关闭，通过 `--audit-log PATH` 显式开启。

## 目录

```text
engine.cpp          CPU correctness engine 与完整单 token forward
arch/cuda/engine.cu CUDA Model/State、chunk prefill 和单 token forward
runtime.cpp         Session、sampling 和 cache 生命周期
main.cpp            main、HTTP routes 和 completion 数据流
parser.cpp          唯一 JSON-aware 的 C++ 边界
render.cpp          固定 Qwen3.5 chat template 和 tokenizer
scripts/            离线 packer 及其 Python 环境
tests/              parser、renderer、runtime 和端到端回归
reference/          官方 PyTorch/Transformers 数值 reference
eval/               EvalScope 评测工具和结果
third_party/        固定版本的 JSON、HTTP 和日志依赖
build/              下载的 checkpoint、生成的模型和编译产物
```

开发阶段 model 和 render 数据分开，方便调试；稳定后再考虑打包为一个模型文件。

## 历史

早期从基础数学逐步搭建 Qwen 的课程代码和学习笔记保存在 Git tag
`learning`，不再属于当前主线。

## Roadmap

1. 保持 Qwen3.5-0.8B correctness baseline。
2. 以 Qwen3.5-4B BF16 作为快速本机 coding agent 路线。
3. 以 Qwen3.5-9B Q8_0 作为 16 GiB 显卡上的默认高质量路线。
4. 27B 留给独立目标。
