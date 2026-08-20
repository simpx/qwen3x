# 当前状态

快照日期：2026-08-20。这是一份项目事实检查点；它不代表 Qwen3.8-27B 已经能够运行。

## 项目边界

`qwen3x` 是一个可阅读、模型专用的 Qwen hybrid inference 项目。它不是通用 Tensor
框架、多模型 runtime 或 serving system。

教学模型是 **Qwen3.5-0.8B**；最终实际目标是 **Qwen3.8-27B**，text-only、batch=1、
single GPU。顺序始终是：

    correct -> simple -> readable -> usable -> fast

## 已完成的内容

| 区域 | 当前证据 |
| --- | --- |
| `lessons/` | Lesson 00--08 解释各个小数学部件；lesson 09 能运行固定 Qwen3.5-0.8B 的完整 CPU forward。 |
| `01-hf-reference/` | 本地官方 Hugging Face checkpoint 以 eager FP32、逐 token 方式运行；CPU 与 CUDA oracle vectors 分开生成。 |
| `02-cpu-0.8b/` | plain C++ `Model + State + Work`；BF16 checkpoint、FP32 activations/state；具备 prefill、decode、DeltaNet state、attention KV cache 与 greedy decode。 |
| CPU regression | 每一步都将完整 248,320 词表 logits 与官方 CPU oracle 对比。版本化最大绝对误差为 `5e-4`，另检查 argmax、greedy ids 与 API state。 |
| `03-cuda-0.8b/` | non-linear/state 操作由直接 CUDA kernel 实现；所有 linear projection 由 cuBLAS GEMV 实现；模型 state 常驻 GPU。 |
| CUDA regression | 每一步都与独立生成的官方 CUDA FP32 oracle 对比完整 logits，使用相同 `5e-4` 阈值；另检查 greedy ids 与 state API。 |
| `04-0.8b-e2e/` | 官方 Python tokenizer/chat template 仅是一层很薄的外壳；C++ 仍然只接收 token ids。已测一个真实中文 chat prompt 的端到端路径。 |
| `05-qwen38-27b/` | 不下载 weight shard 的情况下，已检查官方 Qwen3.8-27B config 和 safetensors index；错误 config/schema/byte-count fixture 均会被拒绝。 |

当前 0.8B vector suite 有三组：3-token prompt + 2 decode token、4-token prompt +
2 decode token、18-token official chat prompt + 2 decode token；每组再跑 8-token
greedy continuation。

## 已确认的 Qwen3.8-27B contract

metadata preflight 已固定以下 text-only 架构事实：

- 64 层：48 个 Gated DeltaNet layer、16 个 full-attention layer，3:1 循环。
- hidden size 5120；FFN intermediate size 17408。
- attention：24 个 query head、4 个 KV head、head dim 256。
- DeltaNet：16 个 key head、48 个 value head，key/value dim 都是 128，conv width 为 4。
- checkpoint weight 为 BF16；`lm_head` **不与** embedding table tied。
- text-only pack 需要 851 个 tensor；vision 与 MTP tensor 不属于它。
- 原始 checkpoint payload 为 55,562,855,904 bytes（51.747 GiB），分 18 个 shard。

在 4096-token context 下，直接 FP32 runtime state 的估算为：DeltaNet recurrent
state 约 0.141 GiB，DeltaNet conv history 约 0.005 GiB，full-attention KV cache
约 0.500 GiB。这些数字不包括 weights、scratch buffer 或 cuBLAS workspace。

## 当前限制：都是有意的，但必须正视

- 当前开发 GPU 为 16 GiB；raw 27B BF16 weights 无法放入。
- Stage 3 CUDA 路径为了得到清楚的 0.8B correctness contract，会将每个 linear BF16
  matrix 展开成 FP32。若用于 27B，仅 weights 就约为 103.5 GiB，尚未加 runtime state；
  因而绝不能直接复用。
- 0.8B CPU/CUDA snapshot 有意将 context 上限设为 4096；checkpoint 本身宣称 262,144
  positions。
- C++ 还没有 native tokenizer、greedy 以外的 sampling、HTTP server、prefix cache、
  batching、quantization、vision 或 MTP。
- text-only wrapper 仅有一个 user message 的 regression；system/history 以及明确拒绝
  image/video placeholder token 仍需测试。

## 写 27B CUDA forward 前必须通过的 gate

1. **真实的 27B reference。** 在目标大显存机器上生成 pinned、官方 Qwen3.8-27B BF16
   reference bundle：token ids、greedy ids、logits 和选定的 intermediate probes。
2. **中间诊断。** 记录 embedding、每层 mixer output/residual/MLP output、final norm、
   选定 DeltaNet state、选定 attention KV entry。只看最终 logits 无法定位漂移发生在哪一层。
3. **解决 attention-gate 语义。** 27B config 写的是 `output_gate_type = swish`，但当前
   Transformers Qwen3.5 attention implementation 实际乘的是 `sigmoid(gate)`。不能猜；
   必须以 27B attention probe 确立 reference 行为，再写 C++。
4. **定义 27B packed format。** 必须有 format version、精确 model revision/config fingerprint、
   complete-file validation 和独立的 `lm_head`。
5. **BF16 runtime design。** 27B linear weight 必须在 device 上保持 BF16，并建立单独测试的
   BF16 numerical tolerance；不能静默沿用 0.8B 的 FP32-expanded CUDA 路径。
6. **长 state 覆盖。** 测 conv boundary、attention cache growth、RoPE position，直到选定的
   初始 context limit（若先定为 4096，完全可以，但必须写明并强制）。

## 小的维护项

- 0.8B CPU/CUDA stage 的 `make weights` 只依赖 packer script，不依赖 source checkpoint
  file；替换 `MODEL` 而不删除已有 bin 时，可能复用 stale packed weights。
- CUDA self-test 暴露为 `make cuda-test`；当前 `make test` 并不会执行它。
- lesson、CPU、CUDA snapshot 有意分开，换取可读性；但未来 27B snapshot 必须明确唯一的
  source of truth，不能出现第四份悄悄漂移的实现。

## 精确的下一步

不要优化，也不要加 framework。先到能够容纳 27B 的目标机器，构建官方 27B
reference/probe bundle；只有它存在之后，项目才写固定 27B packer 与 BF16 CUDA decode path。
