# Qwen3.5-0.8B session runtime

这是面向个人 OpenAI server 的正式实现目录。它从课程和 Stage 2 CPU correctness 代码继续，
但运行时不再启动 `--worker` 子进程：Python 与 C++ 在同一个进程，通过窄 C ABI 直接调用。

```text
OpenAI client
  -> Python HTTP / tokenizer / streaming
  -> C++ SessionManager
  -> C++ Session
  -> C++ Engine + mmap weights
```

## 概念

```text
Engine
  = 一份加载好的只读 Model/weights

Session
  = 一条可变 token 时间线
  = State + Work + tokens + logits

SessionManager
  = 固定数量的预创建 Session
  = prefix lookup + FREE/IDLE/BUSY + LRU
```

`Engine`、`Session` 和 `SessionManager` 都定义在 C++。Python 只处理 HTTP、JSON、tokenizer 和
流式响应。Runtime 启动时创建固定数量的 Session，此后请求不会重新分配模型 State/Work。
一个 Session 同时只允许一个 writer；不同 Session 可以由调度器独立推进。

## 文件

```text
qwen35.h                 Engine、Session、SessionManager 的公共 C ABI
engine.cpp               CPU Model/State/Work/forward + Engine/Session
runtime.cpp              SessionManager、ID/prefix、BUSY 和 LRU
log.cpp                  Engine/Runtime 共用的进程级日志回调
internal.h               Engine 向 Runtime 提供只读 token timeline 的私有接口
qwen35.py                上述 C ABI 的薄 ctypes 包装
server.py                OpenAI chat completions、SSE、鉴权和 tokenizer
client                   默认连接本机的极简多轮命令行客户端
pack_weights.py          官方 safetensors -> mmap-friendly 固定 tensor stream
pyproject.toml / uv.lock Python 直接依赖和完整锁定环境
tests/                   HTTP、真实权重 ABI/SessionManager 和真实 HTTP e2e
```

模型计算仍然是固定 Qwen3.5-0.8B，不引入 Tensor abstraction、backend hierarchy 或 JSON parser。
错误在 C ABI 边界转换为错误码，C++ library 不会因为请求错误结束 Python 进程。

## 构建与验证

```sh
cd qwen35-0.8b
make
make weights
make test
make e2e
```

`unit-test` 不加载真实模型；`native-test` 验证 mmap 权重、独立 Session、SessionManager、
append-only sync、编辑/缩短后的 rebuild 和完整词表 logit；`e2e` 走真实 tokenizer、HTTP、
SessionManager 和 C++ forward。

## 启动

```sh
cd qwen35-0.8b
make run
```

另开一个终端发送最小流式请求：

```sh
make chat
```

更方便的多轮测试：

```sh
./client "你好" "我刚才说了什么？"
```

每个位置参数是一轮 user 消息。Client 会等待本轮响应完成，再携带完整可见对话发送下一轮；
服务端根据完整 token 前缀自动复用 Session，并以 `req / resp / usage` 格式显示结果。常用选项：

```sh
./client -n 64 -t 0.7 -p 0.9 "你好" "继续"
./client --help
```

`pyproject.toml` 声明依赖，`uv.lock` 锁定完整环境；两者都应提交。`uv run` 会自动创建和同步
项目自己的 `.venv`，不需要手动创建或激活虚拟环境。修改依赖使用 `uv add`/`uv remove`，不要
直接维护 requirements 文件。CI 或部署使用 `--locked`，确保声明和 lock 不一致时直接失败。
systemd 启动前先执行一次 `make sync-prod`；service 使用 `--no-sync`，启动时不会修改环境或访问
包索引。开发和测试使用 `make sync`，它会额外安装 `dev` dependency group。

Python 和 C++ 使用同一条日志链：C++ 通过全局 callback 上报消息和源码位置，Python
统一补上时间、native thread ID 和 `request_id`，同时写终端和
`logs/qwen35.log`。日志默认保留 5 个 20 MB
轮转文件；可以用 `--log-level`、`--log-file`、`--log-max-mb` 和 `--log-backups` 调整。每个 HTTP
响应都会返回服务端生成的 `X-Request-Id`，可直接用它串起一次请求的 Python/C++ 日志。

默认不启用鉴权。需要 Bearer 鉴权时，显式设置 `QWEN_API_KEY` 再启动：

```sh
QWEN_API_KEY='换成随机长字符串' make run
```

默认监听 `127.0.0.1:8000`。请求使用标准 OpenAI Chat Completions 字段：

```json
{
  "model": "qwen3.5-0.8b",
  "messages": [{"role": "user", "content": "你好"}],
  "temperature": 0,
  "stream": true
}
```

多轮请求仍需像标准 Chat Completions 一样携带完整 `messages`。Runtime tokenize 后，在所有空闲
Session 的 live/anchor 中选择最长 token prefix；没有命中时才使用 FREE/LRU Session 并 rebuild。
每个 Session 当前有两个可命中点：`live` 是当前 State，`anchor` 是额外保存在内存中的 State。
Session 不区分 prefill 和 decode；当前策略只是在每次 `sync()` 结束时更新 anchor。

保存 anchor 时只额外复制 DeltaNet 的 recurrent/conv state；Attention KV 本来就是按 token
append 的，所以恢复时只截断到 checkpoint position，不额外复制一整份 KV。

## 当前边界

- CPU BF16-weight / FP32-compute，支持 greedy、temperature 和 top-p sampling，text-only。
- 固定 Session 数量；全部 BUSY 时返回 429，IDLE Session 按最长 prefix 或 LRU 重新绑定。
- 中断只能在 token 边界发现；一次 CPU forward 尚不能抢占。
- 没有跨 Session 共享的 prefix block、disk cache、batching、vision、tools 或 MTP；当前 prefix
  命中是把一个完整 Session 重新交给请求，而不是让多个 Session 同时共享一份 State。

下一步可以在不改变 C ABI 基本关系的前提下加入 disk checkpoint 和 CUDA Engine。公共前缀以后可
作为 pinned prefix 优化：Attention block 可共享，DeltaNet 起点 State 仍需复制给各 Session。
