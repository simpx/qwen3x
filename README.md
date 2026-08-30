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

## 构建和使用

下载官方 checkpoint，并准备权重和开发阶段独立的 render 数据：

```sh
make -C scripts checkpoint model render
make -j4
```

默认构建 `build/qwen35` CPU correctness engine。NVIDIA CUDA 版本使用同一套
main/runtime/render，只替换 engine：

```sh
make cuda -j4
./build/qwen35-cuda --prompt "hello"
```

CUDA 构建需要 CUDA Toolkit，当前在 CUDA 12.8 / compute capability 8.9 上验证。
decode 使用显式单 token forward 的 CUDA Graph，prefill 按 128-token chunk 执行；完整
优化过程、正确性阈值和 CPU/CUDA benchmark 见 [eval/cuda.md](eval/cuda.md)。

所有下载和生成的文件都位于 `build/`。`make clean` 清理程序、render 和测试产物，
但保留下载的 checkpoint 与 pack 后的模型权重。首次构建可以并行编译；之后直接运行
`make` 只会重编发生变化的源码及其依赖。

直接完成一次请求：

```sh
./build/qwen35 --prompt "hello" \
  --max-tokens 128
```

`--prompt` 直接构造结构化的 chat request，不经过 JSON。启动服务则使用
`--listen`：

```sh
./build/qwen35 \
  --listen \
  --host 127.0.0.1 \
  --port 8000 \
  --session-slots 1 \
  --session-context 262144
```

设置 `QWEN_API_KEY` 会为 `/v1/models` 和 `/v1/chat/completions` 启用 Bearer
鉴权。

日常请求可以使用零依赖的薄客户端，不需要手写 JSON：

```sh
scripts/chat.py -u "你好"
scripts/chat.py -s "回答简短" -u "你好" -a "你好！" -u "你是谁？"
```

`-c ID NAME ARGUMENTS` 表示 assistant tool call，`-t ID CONTENT` 用同一个
ID 返回工具结果。连续的 `-c` 属于同一条 assistant 消息：

```sh
scripts/chat.py \
  -u "杭州天气怎么样？" \
  -c call_weather weather '{"city":"杭州"}' \
  -t call_weather "晴，28°C"
```

`ARGUMENTS` 也可以写成 `@args.json`。增加 `--dry-run` 只生成可直接执行的
curl，不发送请求：

```sh
scripts/chat.py -u "你好" --dry-run
```

用最小 Agent 闭环测试原生工具调用：

```sh
scripts/agent.py -y "Use bash to run pwd, then tell me the directory."
```

`agent.py` 只提供 `read_file`、`write_file` 和 `bash`，默认在当前目录工作；
`write_file` 和 `bash` 默认需要确认，`-y` 用于受控测试环境。当前 Agent 请求使用
非流式 Chat Completions；普通文本 completion 已支持流式，流式 tool calls 后续补充。
`read_file` 默认读取 200 行，也接受 `start_line` 和 `line_count`，让模型按段阅读大文件。

测量不含 HTTP、JSON、chat template、tokenizer 和 sampling 的 Session 性能：

```sh
./build/qwen35 --bench 512 128
```

两个数字依次是 prefill token 数和 decode token 数。默认 Session context 是两者
之和，也可以用 `--session-context` 显式指定更大的容量。benchmark 默认创建一个
session slot；`--session-slots` 可以复现服务所用的内存配置，但计时仍只推进其中一个
Session，不代表并发吞吐量。

`qwen35` 默认从可执行文件所在目录加载 `qwen35-0.8b-model.bin` 和
`qwen35-0.8b-render.bin`。`--model` 和 `--render` 只在需要覆盖默认路径时使用。

普通 completion：

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.5-0.8b",
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

1. 完整支持 Qwen3.5-0.8B。
2. 完整支持 Qwen3.8-27B。
