# Qwen3.5-0.8B session runtime

这是面向个人 OpenAI server 的正式实现目录。它从课程和 Stage 2 CPU correctness 代码继续，
但运行时不再启动 `--worker` 子进程：Python 与 C++ 在同一个进程，通过窄 C ABI 直接调用。

```text
OpenAI client
  -> Python HTTP / tokenizer / streaming
  -> SlotPool
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

Slot
  = Python Runtime 对一个预创建 Session 的调度记录
  = owner session_id + FREE/IDLE/BUSY + last_used
```

`Engine` 和 `Session` 定义在 C++。`Slot` 和 `SlotPool` 定义在 Python。Runtime 启动时创建固定
数量的 Session，此后请求不会重新分配模型 State/Work。一个 Session 同时只允许一个 writer；
不同 Slot 可以由调度器独立推进。

## 文件

```text
engine.h                 稳定的 opaque-handle C ABI
engine.cpp               CPU Model/State/Work/forward + Engine/Session
pack_weights.py          官方 safetensors -> mmap-friendly 固定 tensor stream
qwen_runtime/binding.py  ctypes Engine/Session 包装
qwen_runtime/slots.py    预分配 SlotPool、session_id 绑定和 LRU idle 淘汰
qwen_runtime/server.py   OpenAI chat completions、SSE、鉴权和 tokenizer
tests/                   Slot、HTTP、真实权重 ABI 和真实 HTTP e2e
```

模型计算仍然是固定 Qwen3.5-0.8B，不引入 Tensor abstraction、backend hierarchy 或 JSON parser。
错误在 C ABI 边界转换为错误码，C++ library 不会因为请求错误结束 Python 进程。

## 构建与验证

```sh
make -C qwen35-0.8b
make -C qwen35-0.8b weights MODEL=../models/Qwen3.5-0.8B
make -C qwen35-0.8b test
make -C qwen35-0.8b e2e
```

`unit-test` 不加载真实模型；`native-test` 验证 mmap 权重、两个独立 Session、append-only sync、
编辑/缩短后的 rebuild 和完整词表 logit；`e2e` 走真实 tokenizer、HTTP、Slot 和 C++ forward。

## 启动

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r qwen35-0.8b/requirements.txt

export QWEN_API_KEY='换成随机长字符串'
make -C qwen35-0.8b run
```

默认监听 `127.0.0.1:8000`。标准 OpenAI Chat Completions 请求可以不使用 Session，此时 Slot 在请求
结束后释放。需要持续保留模型 State 时，传扩展字段或请求头：

```json
{
  "model": "qwen3.5-0.8b",
  "session_id": "my-agent",
  "messages": [{"role": "user", "content": "你好"}],
  "temperature": 0,
  "stream": true
}
```

也可以使用 `X-Qwen-Session-Id: my-agent`。响应正文、SSE chunk 和响应头都会返回 session id。
删除常驻状态：

```sh
curl -X DELETE http://127.0.0.1:8000/v1/sessions/my-agent \
  -H "Authorization: Bearer $QWEN_API_KEY"
```

同一 `session_id` 的下一次请求会回到同一个 C++ Session。Runtime 仍发送完整 token 序列，
`q35_session_sync()` 判断旧时间线是否是新序列的完整前缀：是则只 forward 新后缀；否则 reset
并从头 rebuild。这与 GDN State 的约束一致，不假装支持任意位置回滚。

## 当前边界

- CPU BF16-weight / FP32-compute，greedy only，text-only。
- 固定 Session 数量；全部 BUSY 时返回 429，IDLE Slot 按 LRU 重新绑定。
- 中断只能在 token 边界发现；一次 CPU forward 尚不能抢占。
- 没有跨 Session common-prefix sharing、snapshot/disk cache、batching、vision、tools 或 MTP。
- `session_id` 是本项目扩展，不是 Chat Completions 标准字段。

下一步可以在不改变 C ABI 基本关系的前提下加入 sampler、snapshot 和 CUDA Engine。公共前缀以后可
作为 pinned prefix 优化：Attention block 可共享，DeltaNet 起点 State 仍需复制给各 Session。
