# 官方 Qwen3.5 reference

这一目录的任务只有一个：调用官方 Hugging Face checkpoint，产出带来源、可重复的
**test vectors**。它不是另一套 C++ engine，也不是 production runtime。

这里直接在真实 checkpoint 上把“官方模型此时应输出什么”固定下来。0.8B 保留严格
FP32 correctness baseline；4B 增加面向 BF16 Tensor Core 生产路径的行为契约。参考服务、vectors dump、
Engine 对比工具和完整 Python 环境都留在本目录，不污染实际 runtime。

## 为什么同时用 PyTorch 和 Transformers

Transformers 负责加载官方 Qwen3.5 checkpoint 和 tokenizer；PyTorch 执行官方 forward。两者
只属于开发/测试环境：C++ runtime 不链接 Python、PyTorch 或 Transformers。

这里不会用 `model.generate()` 作为黑盒。`dump_vectors.py` 直接逐 token 调用官方 model，保留
`past_key_values`；没有额外的 HF wrapper，因此它的执行语义一眼就对应 C++ 的：

```text
prefill(A, B, C, D) → decode(E) → decode(F)
```

每次 token forward 后都保存 full-vocabulary logits；greedy continuation 也单独保存。这样出错时
可以定位到第几个 token，而不是只看到最后一句文本不对。

## 独立环境

当前 Qwen3.5 checkpoint 需要支持 `qwen3_5` 的 Transformers。本目录用自己的
`pyproject.toml` 和 `uv.lock` 锁定 Torch、Transformers、NumPy、FastAPI 和 Uvicorn：

```sh
uv sync --locked
```

这些不是 runtime dependency；`.venv/` 和 `build/` 都不提交到 Git。下面所有 Make target
都会自动通过 `uv run --locked` 使用这个环境。

## 运行

仓库已经有本地 checkpoint 时：

```sh
make cpu MODEL=../build/models/Qwen3.5-0.8B    # official CPU FP32 vectors
make cuda MODEL=../build/models/Qwen3.5-0.8B   # official CUDA FP32 vectors
```

默认 `make` 等同于 `make cpu`。CPU 和 CUDA 都以 FP32 加载约 0.8B 参数；两者的 vectors
分别保存，避免 GPU reduction order 被误当作 CPU reference。没有 CUDA 时只运行 `make cpu`。

生成的文件：

```text
build/cpu/vectors.json  # CPU model/config/tokenizer fingerprint、cases、dtype、误差契约
build/cpu/vectors.npz   # CPU: 每个 case 的输入 token ids、每步 full logits、greedy ids
build/cuda/...          # CUDA oracle 的同样两份文件
```

`.npz` 是测试工具的交换格式，不是模型格式。Engine 使用的模型格式仍由
根目录的 `scripts/pack_weights.py` 决定。

Engine 的比较门槛也记录在 `vectors.json`：官方 FP32 PyTorch GEMV 与“BF16 weights 展开为
FP32、C++ scalar accumulation”的运算顺序不同，因此用实测的 `max_abs_error <= 5e-4`；每一步
argmax 仍必须完全相同。这个门槛来自全量 vector regression，不是生成文本后临时放宽的阈值。

## llama.cpp Q8_0 smoke test

`make llama-smoke` 用独立的 llama.cpp 实现检查 9B Q8_0 的端到端 logits。它调用外部
`llama-debug --save-logits` 和 qwen3x 对称的 `--save-logits` CLI。两边从相同的原始 prompt
各自 tokenize，token IDs 必须完全相同，然后才比较最后一个位置的完整 logits。三个固定短输入
分别覆盖单 token、短序列和包含中英文的 20-token 序列；每个输入比较
完整词表的最大/平均绝对误差、cosine、argmax 和 top-10。固定通过门槛为最大误差不超过
`0.50`、平均误差不超过 `0.075`、cosine 不低于 `0.9995`、argmax 相同且 top-10 至少重合
9 项。

对齐路径不做 sampling 或生成，所以 seed、temperature、top-k、top-p 和 max tokens 都不参与。
两边从 position 0 开始，使用 128-token context 和单次 prefill；llama.cpp 另外固定
`-b 128 -ub 128 -fa on -ctk f32 -ctv f32 --no-warmup`，qwen3x 使用固定的 CUDA FP32 state。

llama.cpp 不是项目依赖：仓库不包含、下载、编译或链接它，默认 `make test` 也不运行这个
检查。已经在仓库外准备好与 qwen3x model.bin 来自同一官方 checkpoint 的 Qwen3.5-9B
Q8_0 GGUF，以及带 CUDA 的 `llama-debug` 后执行：

```sh
make llama-smoke
```

默认路径对应 `eval/q8-9b.md` 固定的 llama.cpp b10516 和 Unsloth GGUF。其他位置显式传入：

```sh
make llama-smoke \
  LLAMA_DEBUG=/path/to/llama.cpp/build/bin/llama-debug \
  LLAMA_GGUF=/path/to/Qwen3.5-9B-Q8_0.gguf \
  QWEN_BIN=/path/to/qwen35-9b-q8_0-model.bin
```

外部工具缺失时 target 直接报错，不自动安装。整个命令有 15 分钟硬超时；当前 16 GiB CUDA
测试机的预期耗时是几十秒。这个 smoke test 判断两套完整模型是否数值等价；Q8 block 和
CPU/CUDA kernel 的局部错误仍由 `test_pack_weights.py`、`q8-cpu-test` 和 CUDA reference 测试定位。

CUDA 实现使用 BF16 checkpoint weights、FP32 activation 和 FP32 cuBLAS accumulation；它保存独立
CUDA oracle，但同样以紧的 `cuda_max_abs_error = 5e-4` 逐步比较。以后 Tensor-Core BF16-activation
优化必须新建自己的 reference/contract，不能修改这个 correctness baseline。

4B prefill 的 cuBLAS 路径使用 BF16 权重和 BF16 activation、FP32 accumulation。官方
checkpoint 无法在 16 GiB GPU 上同时保留完整 FP32 权重，因此 `mixed-fp32` oracle 保留官方
BF16 权重，并让 Embedding、Linear 和 conv 在每个算子边界以 FP32 计算。它的
`--behavior-only` 契约要求：greedy token 完全一致（或处于 0.02 的近并列 argmax）、top-10
至少重叠 9 项，并继续报告完整 logits 最大误差。旧的 0.8B strict contract 没有放宽。

## Case 语义

每个 case 有 `prefill` 与 `decode` 两段 token ids。保存的 `step_logits[i]` 是输入
`prefill + decode` 中第 `i` 个 token 之后、预测**下一个 token**的所有 logits。

因此 CPU/CUDA 都能比较：prompt 中的每一步、跨 state 的 decode 步，以及第一个 greedy token。
它们不得只比较最终 argmax。

## 对比 C++ Engine

生成 vectors 后，可以在本目录直接对比 C++ Engine 的逐 token 完整 logits：

```sh
make compare \
  MODEL=../build/models/Qwen3.5-0.8B \
  CHAT_TEMPLATE=chat_template.jinja
```

shared library、vectors 和报告都只生成在本目录的 `build/` 下；报告默认写入
`reference/build/reference-report.json`（从仓库根目录看）。

4B CUDA 的完整 8-case 比较使用：

```sh
make compare-cuda-4b
```

case 覆盖长度 1/2/16/64、短序列、prefix、thinking/non-thinking chat，并检查 fresh
decode、chunk、checkpoint restore、Session cache 和 4-token greedy continuation。

## EvalScope reference server

需要判断分数差异来自 Engine 还是评测 recipe 时，可在 GPU 上启动最小的官方 Transformers
OpenAI-compatible oracle：

```sh
make serve MODEL=../build/models/Qwen3.5-0.8B \
  CHAT_TEMPLATE=chat_template.jinja DEVICE=cuda DTYPE=float32 \
  CACHE=static MAX_CONTEXT=40960 PORT=8002
```

`server.py` 只实现 EvalScope 使用的 non-streaming chat completions；它不是生产 Server。
它使用同一份 runtime 模板，并支持 temperature、top-p、top-k、generated-only presence penalty
和 seed，因而可以用相同的 `eval/run.py` 生成 reference 分数。
默认 `DTYPE=float32`，用于和 BF16-weight/FP32-compute 的 C++ Engine 做正确性比较。
`DTYPE=bfloat16` 只适合追求速度的近似分数；它可能让 greedy 路径在接近的 logits 处分叉，
不能据此判断 Engine 数值错误。

默认使用 Transformers 原生 `StaticCache`。Qwen3.5 的 dynamic KV cache 在逐 token 生成时会
反复 `torch.cat`；32K 输出不仅不断复制旧 KV，还可能因 CUDA allocator 碎片耗尽显存。
StaticCache 一次分配 `prompt_tokens + max_tokens` 的容量，后续原地写入，数学和 dtype 不变。
`/readyz` 会记录 cache 类型和 context 上限；需要做实现消融时可显式传 `CACHE=dynamic`。
