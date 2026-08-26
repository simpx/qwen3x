# 官方 Qwen3.5-0.8B reference

这一目录的任务只有一个：调用官方 Hugging Face checkpoint，产出带来源、可重复的
**test vectors**。它不是另一套 C++ engine，也不是 production runtime。

`00-lessons/` 已经用小尺寸解释了 RMSNorm、DeltaNet、attention、state 和完整 CPU forward；
这里直接在真实 0.8B 上把“官方模型此时应输出什么”固定下来。参考服务、vectors dump、
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
make cpu MODEL=../models/Qwen3.5-0.8B    # official CPU FP32 vectors
make cuda MODEL=../models/Qwen3.5-0.8B   # official CUDA FP32 vectors
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
`qwen35-0.8b/pack_weights.py` 决定。

Engine 的比较门槛也记录在 `vectors.json`：官方 FP32 PyTorch GEMV 与“BF16 weights 展开为
FP32、C++ scalar accumulation”的运算顺序不同，因此用实测的 `max_abs_error <= 5e-4`；每一步
argmax 仍必须完全相同。这个门槛来自全量 vector regression，不是生成文本后临时放宽的阈值。

CUDA 实现使用 BF16 checkpoint weights、FP32 activation 和 FP32 cuBLAS accumulation；它保存独立
CUDA oracle，但同样以紧的 `cuda_max_abs_error = 5e-4` 逐步比较。以后 Tensor-Core BF16-activation
优化必须新建自己的 reference/contract，不能修改这个 correctness baseline。

## Case 语义

每个 case 有 `prefill` 与 `decode` 两段 token ids。保存的 `step_logits[i]` 是输入
`prefill + decode` 中第 `i` 个 token 之后、预测**下一个 token**的所有 logits。

因此 CPU/CUDA 都能比较：prompt 中的每一步、跨 state 的 decode 步，以及第一个 greedy token。
它们不得只比较最终 argmax。

## 对比 C++ Engine

生成 vectors 后，可以在本目录直接对比 C++ Engine 的逐 token 完整 logits：

```sh
make compare \
  MODEL=../models/Qwen3.5-0.8B \
  CHAT_TEMPLATE=../qwen35-0.8b/chat_template.jinja
```

也可以在 `qwen35-0.8b/` 中运行 `make reference`，一次完成生成和对比。报告默认写入
`qwen35-0.8b/build/reference-report.json`。

## EvalScope reference server

需要判断分数差异来自 Engine 还是评测 recipe 时，可在 GPU 上启动最小的官方 Transformers
OpenAI-compatible oracle：

```sh
make serve MODEL=../models/Qwen3.5-0.8B \
  CHAT_TEMPLATE=../qwen35-0.8b/chat_template.jinja DEVICE=cuda DTYPE=float32 \
  CACHE=static MAX_CONTEXT=40960 PORT=8002
```

`server.py` 只实现 EvalScope 使用的 non-streaming chat completions；它不是生产 Server。
它使用同一份 runtime 模板，并支持 temperature、top-p、top-k、generated-only presence penalty
和 seed，因而可以用相同的 `qwen35-0.8b/evaluation/run.py` 生成 reference 分数。
默认 `DTYPE=float32`，用于和 BF16-weight/FP32-compute 的 C++ Engine 做正确性比较。
`DTYPE=bfloat16` 只适合追求速度的近似分数；它可能让 greedy 路径在接近的 logits 处分叉，
不能据此判断 Engine 数值错误。

默认使用 Transformers 原生 `StaticCache`。Qwen3.5 的 dynamic KV cache 在逐 token 生成时会
反复 `torch.cat`；32K 输出不仅不断复制旧 KV，还可能因 CUDA allocator 碎片耗尽显存。
StaticCache 一次分配 `prompt_tokens + max_tokens` 的容量，后续原地写入，数学和 dtype 不变。
`/readyz` 会记录 cache 类型和 context 上限；需要做实现消融时可显式传 `CACHE=dynamic`。
