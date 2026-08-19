# Stage 1：官方 Qwen3.5-0.8B reference

这一目录的任务只有一个：调用官方 Hugging Face checkpoint，产出带来源、可重复的
**test vectors**。它不是另一套 C++ engine，也不是 production runtime。

`lessons/` 已经用小尺寸解释了 RMSNorm、DeltaNet、attention、state 和完整 CPU forward；
这里直接在真实 0.8B 上把“官方模型此时应输出什么”固定下来。随后 `02-cpu-0.8b/` 和
`03-cuda-0.8b/` 都只需读取同一份 vectors 比较。

## 为什么同时用 PyTorch 和 Transformers

Transformers 负责加载官方 Qwen3.5 checkpoint 和 tokenizer；PyTorch 执行官方 forward。两者
只属于开发/测试环境：C++ runtime 不链接 Python、PyTorch 或 Transformers。

这里不会用 `model.generate()` 作为黑盒。`reference.py` 显式逐 token 调用官方 model，保留
`past_key_values`，因此它的执行语义正好对应 C++ 的：

```text
prefill(A, B, C, D) → decode(E) → decode(F)
```

每次 token forward 后都保存 full-vocabulary logits；greedy continuation 也单独保存。这样出错时
可以定位到第几个 token，而不是只看到最后一句文本不对。

## 一次性开发依赖

当前 Qwen3.5 checkpoint 需要支持 `qwen3_5` 的 Transformers。全局 Python 若版本较旧，请在
此目录建立虚拟环境：

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

这不是 runtime dependency；`.venv/` 和 `build/` 都不提交到 Git。

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

`.npz` 是测试工具的交换格式，不是模型格式。模型格式仍由 Stage 2 的 `pack_weights.py` 决定。

Stage 2 的比较门槛也记录在 `vectors.json`：官方 FP32 PyTorch GEMV 与“BF16 weights 展开为
FP32、C++ scalar accumulation”的运算顺序不同，因此用实测的 `max_abs_error <= 5e-4`；每一步
argmax 仍必须完全相同。这个门槛来自全量 vector regression，不是生成文本后临时放宽的阈值。

Stage 3 使用 BF16 checkpoint weights、FP32 activation 和 FP32 cuBLAS accumulation；它保存独立
CUDA oracle，但同样以紧的 `cuda_max_abs_error = 5e-4` 逐步比较。以后 Tensor-Core BF16-activation
优化必须新建自己的 reference/contract，不能修改这个 correctness baseline。

## Case 语义

每个 case 有 `prefill` 与 `decode` 两段 token ids。保存的 `step_logits[i]` 是输入
`prefill + decode` 中第 `i` 个 token 之后、预测**下一个 token**的所有 logits。

因此 CPU/CUDA 都能比较：prompt 中的每一步、跨 state 的 decode 步，以及第一个 greedy token。
它们不得只比较最终 argmax。
