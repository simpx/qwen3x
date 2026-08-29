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
make
```

所有下载和生成的文件都位于 `build/`。`make clean` 清理程序、render、benchmark
和测试产物，但保留下载的 checkpoint 与 pack 后的模型权重。

启动服务：

```sh
./build/qwen35 \
  --model build/qwen35-0.8b.bin \
  --render build/qwen35-render.bin \
  --host 127.0.0.1 \
  --port 8000 \
  --slots 2 \
  --context 4096
```

也可以直接运行 `make run`。设置 `QWEN_API_KEY` 会为 `/v1/models` 和
`/v1/chat/completions` 启用 Bearer 鉴权。

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

服务还提供 `/healthz`、`/readyz` 和 `/v1/models`。默认只写结构化 stderr 日志；
`--log-file logs/qwen35.log` 可额外启用滚动文件。

## 目录

```text
engine.cpp          模型权重、State、Work 和完整 forward
runtime.cpp         Session、sampling 和 cache 生命周期
main.cpp            main、HTTP routes 和 completion 数据流
parser.cpp          唯一 JSON-aware 的 C++ 边界
render.cpp          固定 Qwen3.5 chat template 和 tokenizer
scripts/            离线 packer、benchmark 及其 Python 环境
tests/              parser、renderer、runtime 和端到端回归
reference/          官方 PyTorch/Transformers 数值 reference
eval/               EvalScope 评测工具和结果
third_party/        固定版本的 JSON、HTTP 和日志依赖
build/              下载的 checkpoint、生成的模型和编译产物
```

开发阶段 `weights.bin` 和 `render.bin` 分开，方便调试；稳定后再考虑打包为一个模型文件。

## 历史

早期从基础数学逐步搭建 Qwen 的课程代码和学习笔记保存在 Git tag
`learning`，不再属于当前主线。

## Roadmap

1. 完整支持 Qwen3.5-0.8B。
2. 完整支持 Qwen3.8-27B。
