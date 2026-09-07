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

项目自带独立的轻量 agent **q3x**（TypeScript + Bun），支持 bash、流式输出、JSONL
记录与恢复、手动/自动上下文压缩。构建后的单个可执行文件无需安装 Bun 或 Node：

```sh
cd agent && bun install --frozen-lockfile && cd ..
make q3x
build/q3x --help
```

在另一个终端运行 `make serve-9b` 或 `make serve-27b` 后，进入工作项目运行
`/path/to/qwen3x/build/q3x --thinking off`。服务默认地址为 `http://127.0.0.1:8000/v1`，
也可通过 `--base-url` 连接其他标准服务器。完整用法和独立构建说明见
[`agent/README.md`](agent/README.md)。`make q3x-test` 运行 agent 测试；测试设计和真实模型
验收结果见 [`agent/TESTING.md`](agent/TESTING.md)。

下面保留已验证的 pi 工作流：

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
`/v1/models` 选择当前服务实际加载的 4B、9B 或 27B 模型，以默认 thinking 启动 pi。
配置模板是 [`scripts/pi-models.json`](scripts/pi-models.json)；压缩余量在
[`scripts/pi-settings.json`](scripts/pi-settings.json)。

首次使用需要先准备模型：

```sh
make model-9b
```

在 Apple Silicon Mac 上，`make serve-9b` 和 `make serve-eval-9b` 自动选择 Metal；Linux
选择 CUDA。原生 CPU binary `build/qwen3x` 继续作为 correctness baseline，其中 Q8_0 dot
使用 Arm NEON，较大的矩阵按行通过系统线程池并行。9B 的 65,536-token CPU eval Session
需要约 13--15 GiB 统一内存，因此建议至少 24 GiB 内存。服务和 q3x 不设置整请求 deadline，
慢速本地生成由完成条件或用户取消结束。M5 Pro 48 GB 的 CPU baseline、容量和六题 smoke 记录在
[`eval/q8-9b-macos.md`](eval/q8-9b-macos.md)。

它下载固定 revision 的官方 Qwen3.5-9B BF16 checkpoint，并在 `build/` 直接生成约
8.86 GiB 的 9B Q8_0 model bin，以及共享的 `qwen3x-render.bin` tokenizer 数据；固定
Qwen3.8 template 编译在可执行文件中。量化只改变权重；activation、recurrent state、KV
cache、workspace 和 logits 仍是 FP32。`make serve-9b` 会检查产物并在
缺失时提示运行上述命令。准备模型需要 [uv](https://docs.astral.sh/uv/)；Linux CUDA 服务
另需 CUDA Toolkit。运行 pi 另需 Node >= 22.19；固定版本安装命令是
`npm install -g @earendil-works/pi-coding-agent@0.84.4`。

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
Apple Silicon Metal 的 `build/qwen3x-metal`、CPU baseline 的 `build/qwen3x`，或 Linux
CUDA 的 `build/qwen3x-cuda`。

M5 Pro 上的 27B 路径只保留 Q4_0：

```sh
make model-27b
make serve-27b
```

它生成 `build/qwen38-27b-q4_0-model.bin`，并固定使用 Metal、单 Session 和 32768 context。
该路径不宣称 CUDA 兼容；当前真实吞吐、峰值内存和长上下文容量仍需用下述 benchmark 验收。

### Apple M5 Pro / Metal 4（实验性）

Metal backend 固定使用 Apple M5 Pro、macOS 26.3+、Xcode 26.6 和 Metal 4.0，不维护旧
系统、旧 Xcode 或其他 GPU 的兼容路径。它直接读取 0.8B/4B BF16、9B Q8_0 和 27B Q4_0
model.bin，无需重新 pack。
Metal shader 编译后嵌入 `qwen3x-metal`，运行时不需要外部 `.metallib`、Python 或第三方
推理框架。平台接口集中在 `arch/metal/engine.mm`，数学计算在 `kernels.metal`。

先在 Mac 上运行不下载模型的数值 smoke：

```sh
make -j3 metal
MTL_DEBUG_LAYER=1 make metal-test
```

Xcode 26 的 Metal 编译器是独立可选组件，先在 Xcode 设置中安装。构建脚本只调用
`xcrun --toolchain Metal`，并固定 `metal4.0` 与 `air64-apple-macosx26.3`；缺少指定环境
就直接失败，不探测或回退到其他工具链。运行已编译的可执行文件不需要 Xcode。

`metal-test` 用非零随机小模型，对比 CPU/Metal 的 BF16、Q8_0 和 Q4_0 完整 forward、每步
logits、recurrent/KV state、prefill、checkpoint restore 和 reset；没有可用 GPU 时返回
非零，不能视为通过。CI 同时构建 macOS CPU/Metal；如果托管 runner 不提供 GPU，会明确
标记 GPU 测试未运行。合成测试不代替真实模型的数值验收。

当前已通过 macOS ARM64 的 CPU 测试及完整 Metal 编译；托管 CI 未提供 Apple GPU，
因此 GPU 数值测试和真实模型验收仍未完成。[Actions](https://github.com/simpx/qwen3x/actions)
中的 `qwen3x-metal-macos-arm64` 附件保留实验性可执行文件、smoke 程序和测试 dylib。
在对应 commit 的仓库根目录解压其中的 tar.gz 后，可直接运行
`MTL_DEBUG_LAYER=1 ./build/metal-test`，不需要在本机安装 Xcode；附件不包含模型。

准备好模型后，Mac 上使用 `make serve-4b` / `make serve-9b` / `make serve-27b`；前两者
自动选择 Metal，27B 明确只使用 Metal。Linux/WSL 的 4B/9B 继续选择 CUDA。连接 pi 的方式
不变。也可以先用 0.8B 做短请求：

```sh
./build/qwen3x-metal -c "你好" --session-context 128 --max-tokens 16
```

首版 prefill 逐 token 执行完整 forward，没有批量矩阵优化；长 prompt 的吞吐和 40K
context 的真实内存占用尚未测量。9B 的权重本身约 8.86 GiB，统一内存还要供系统、KV
和 recurrent state 使用；不要按 CUDA 显存数字直接推断 Mac 的可用容量。

性能改动统一用同一 Engine、单 Session 测量；每种 prompt 长度先预热一次，再取三次
中位数：

```sh
make metal-library
caffeinate -i python3 scripts/bench_session.py \
  --library build/metal/libqwen3x-metal.dylib \
  --model build/qwen35-9b-q8_0-model.bin --output build/bench-9b.json
caffeinate -i python3 scripts/bench_session.py \
  --library build/metal/libqwen3x-metal.dylib \
  --model build/qwen38-27b-q4_0-model.bin --context 32768 \
  --output build/bench-27b.json
```

这个入口不绑定具体模型，可通过 `--model`、`--context`、`--prompts` 和 `--decode` 验收
27B。测试 dylib 只调用与可执行文件相同的 C ABI，不是运行或分发依赖。
测量期间应停止其他推理服务；输出路径必须不存在，以保留每轮原始证据。报告记录 load、
prefill、decode、runtime TTFT、Darwin peak footprint、swap 变化和 memory pressure。

完整 0.8B 官方 FP32 oracle 验证入口为 `make metal-reference`，复用
[`reference/`](reference/README.md) 的 `build/cpu` 向量、相同误差契约和逐 token/cache
测试；需先准备或从 WSL 复制 model.bin、官方 checkpoint 和 reference 向量。

不想把官方 checkpoint 复制到 Mac 时，可以先用现有 CPU engine 制作跨机器 smoke：

```sh
# WSL：各生成 35 个位置的完整词表 logits，不做 sampling。
make metal-smoke-vectors
make metal-smoke-9b-vectors
```

把 `build/metal-smoke-0.8b/`、`build/metal-smoke-9b/` 和对应的 model.bin 复制到 Mac 的
相同目录，再运行 `make metal-smoke` / `make metal-smoke-9b`。仅需系统 Python；不需要
PyTorch、Transformers、tokenizer 或额外 checkpoint。每套向量约 34 MiB。

测试先核对 model.bin 的 SHA-256，再逐 token 比较全部 logits（最大绝对误差 ≤ `5e-4`、
argmax 相同），并检查 prefill/decode、checkpoint 恢复、cache hit、reset 和 Session
隔离（同 backend 路径误差 ≤ `5e-5`）。结果写在向量目录的 `check.json`，记录实际候选
library 和平台；CPU 自检通过不表示 Metal 通过。这是相同 pack 的实现对齐，不是官方
模型跑分，也不替代 `metal-reference`。

## 开发和其他接口

0.8B 作为 CPU correctness 和协议 smoke 基线：

```sh
make -C scripts model render
make -j4
./build/qwen3x --chat "hello" --max-tokens 128
```

`-c/--chat` 把文本作为一条 user message 套用 Qwen chat template；`-p/--prompt` 直接
tokenize 原始文本，主要用于续写和 logits 对齐。保存最后一个 prompt 位置的完整 logits：

```sh
./build/qwen3x-cuda -m build/qwen35-9b-q8_0-model.bin \
  -r build/qwen3x-render.bin -p "Hello" --session-context 128 \
  --save-logits --logits-output-dir build/logits
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
./build/qwen3x-cuda \
  --model build/qwen35-4b-model.bin \
  --bench 4096 32 --session-context 40960
```

两个数字依次是 prefill token 数和 decode token 数。默认 Session context 是两者
之和，也可以用 `--session-context` 显式指定更大的容量。benchmark 默认创建一个
session slot；`--session-slots` 可以复现服务所用的内存配置，但计时仍只推进其中一个
Session，不代表并发吞吐量。

`qwen3x` 默认从可执行文件所在目录加载 `qwen35-0.8b-model.bin` 和
`qwen3x-render.bin`。其他型号通过 `--model` 选择对应 model bin；
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
指定 `--log-file build/qwen3x.log` 后，滚动文件会替代 stderr 成为日志输出位置。

`make serve-4b` 和 `make serve-9b` 默认把完整 audit 分别写入独立的本地文件：

```sh
build/qwen3x-audit.log
build/qwen35-9b-audit.log
```

audit 按事件记录原始请求、render prompt、模型输出、工具解析和实际 HTTP/SSE 输出。
`request_id` 关联完整请求，`session_id` 关联复用同一个缓存 Session 的请求；服务同时通过
`X-Request-Id` 和 `X-Session-Id` 响应头输出这两个 ID。文件权限为 `0600`，内容包含完整
对话和工具参数。手动启动 `qwen3x` 时 audit 默认关闭，通过 `--audit-log PATH` 显式开启。

## 目录

```text
engine.cpp          CPU correctness engine 与完整单 token forward
arch/cuda/engine.cu CUDA Model/State、chunk prefill 和单 token forward
arch/metal/         Metal Model/State、完整 forward 与 MSL kernel
runtime.cpp         Session、sampling 和 cache 生命周期
main.cpp            main、HTTP routes 和 completion 数据流
parser.cpp          唯一 JSON-aware 的 C++ 边界
render.cpp          固定 Qwen3.8 chat template 和 Qwen tokenizer
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
4. 以 Qwen3.8-27B Q4_0 验收 M5 Pro Metal 本地 agent 路线。
