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
  --session-slots 2 \
  --session-context 4096
```

设置 `QWEN_API_KEY` 会为 `/v1/models` 和 `/v1/chat/completions` 启用 Bearer
鉴权。

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
指定 `--log-file logs/qwen35.log` 后，滚动文件会替代 stderr 成为日志输出位置。

## 目录

```text
engine.cpp          模型权重、State、Work 和完整 forward
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
