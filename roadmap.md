# qwen3x roadmap

> 做一个极简、模型专用、可读的 Qwen hybrid inference engine。

这不是 llama.cpp、vLLM 或通用推理框架。开发和 debug 使用 **Qwen3.5-0.8B**；最终
目标是 **Qwen3.8-27B** 在单张 NVIDIA GPU 上可用。CPU reference 可以在 Mac 或普通
Linux 上跑，CUDA 的主要目标机器是 DGX Spark。

项目的优先级始终是：

~~~text
correct → simple → readable → usable → fast
~~~

“能生成看起来正常的文字”不是正确性的定义。每一阶段都应能与上一层 reference 对比
中间 tensor 或 logits；性能优化只能发生在正确版本之后。

## 1. 范围与非目标

### V1 必须支持

| 类别 | 范围 |
| --- | --- |
| 模型 | Qwen3.5-0.8B（开发）、Qwen3.8-27B（目标） |
| 平台 | Mac CPU reference；NVIDIA CUDA / DGX Spark 实用运行 |
| 输入 | text-only、batch=1、single process、single GPU |
| 模型执行 | prefill、decode、attention KV cache、GDN recurrent state、conv state |
| 用户功能 | prefix cache、sampling、CLI、流式 OpenAI-compatible HTTP API |

### 第一版明确不做

- 通用 Tensor framework、computation graph、arbitrary model loader；
- PyTorch 或 Transformers 的 runtime dependency；
- batching、continuous batching、scheduler、paged attention；
- multi-GPU、distributed inference、TP/PP；
- vision、MTP、多模态、training、LoRA；
- 自己发明或维护大量 quantization format。

这些并不是遗留功能；它们是刻意的范围控制。模型本身就是 execution graph：读
`forward()` 应该能直接看见 Qwen 的 layer 顺序，而不是跳过一串 registry/interface。

## 2. 目录就是课程阶段

仓库按“可阅读的阶段”组织。一个目录只承担一个清晰目的；后阶段可以复用已经验证的
模型格式与测试数据，但不把前一阶段的教学实现抽成框架。

~~~text
lessons/                 Phase 0–1：从 toy math 到 0.8B CPU reference 的课程
qwen35-0.8b/             Phase 2–3：真实 0.8B engine、state 与 CUDA correctness
qwen38-27b/              Phase 4–6：27B、prefix cache、日常 CLI/server
bench/                   Phase 7+：可重复 benchmark、Nsight 记录、优化实验
tools/                   HF 转换、reference dump、chat/template 等外围工具
tests/                   tiny / CPU / CUDA / 端到端 regression
~~~

目前 `lessons/` 是 reference 路线：00--08 用可手算的 toy dimensions 讲一个零件，09
直接读取真实 Qwen3.5-0.8B 权重并提供 CPU forward。CUDA、服务和性能代码不应混进这里，
否则读者无法只通过课程理解模型数学。

未来实现文件可以很直接，例如 `model.cpp`、`weights.cpp`、`tokenizer.cpp`、`sampler.cpp`、
`cpu.cpp`、`cuda.cu`、`cuda_gdn.cu`、`cuda_attention.cu`、`prefix_cache.cpp`、`server.cpp`。
不能为“可扩展性”引入 `Tensor`、`Graph`、`Node`、`Operator`、`KernelRegistry`、
`BackendRegistry` 或 `Executor`。

## 3. 依赖原则

**零 inference framework 依赖，但允许使用无聊而成熟的基础设施。**

| 部分 | 允许依赖 | 不允许承担的角色 |
| --- | --- | --- |
| CPU runtime | C++20、libc/STL、pthread | Tensor runtime 或模型框架 |
| CUDA runtime | CUDA Runtime、cuBLAS | PyTorch、TensorRT-LLM、FlashInfer、CUTLASS 首发版 |
| Python tools | HF → 自定义格式转换、官方 reference、测试脚本 | production runtime |
| tokenizer / HTTP | 小型外围库可以评估 | 不得污染 `forward(token_ids)` 的模型核心 |

核心 forward 始终只接收 token ids。chat template 与文字编解码属于外围：在早期可以由
Python 工具调用官方 tokenizer 以保证 reference 一致；日常可用的 V1 再提供不依赖 Python
的 tokenizer/CLI/server 包装。无论采用何种库，它都不能改变模型执行代码的直接性。

权重格式只需要服务两个固定模型。HF safetensors 经 Python 转成一个小的、mmap-friendly
自定义格式；不需要实现 GGUF、GPTQ、AWQ 或 arbitrary Hugging Face config。

## 4. 正确性阶梯

每一层实现只向下游交付已经证明正确的行为：

~~~text
official Python / HF reference
             ↓  intermediate tensors + logits
tiny C++ CPU reference
             ↓  same input / same state
Qwen3.5-0.8B CPU reference
             ↓  same input / same state
Qwen3.5-0.8B CUDA
             ↓  end-to-end regression
Qwen3.8-27B CUDA
~~~

最小测试应覆盖：

- toy RMSNorm、RoPE、attention、SwiGLU、GDN recurrence 的手算断言；
- Python 与 C++ 的 embedding、每个 layer 输入/输出、KV/GDN/conv state、final logits；
- `prefill(ABCD)` 后 `decode(E)`、`decode(F)` 与一次性参考执行等价；
- 固定 seed / greedy decode 的 token 序列；
- 真实 checkpoint 上的 prompt 到文本的端到端 smoke test。

只有在 CPU 对 reference 正确后，CUDA 才以 CPU 为 oracle；只有在 CUDA 正确后，才讨论
profiling、fusion 或 quantization。

## 5. Phase 0 — Tiny reference

**目标时间：1–2 天。**

定义一个可手算的 tiny Qwen：

~~~text
hidden = 64
layers = 4
layer pattern = GDN, GDN, GDN, Attention
FFN intermediate = 128
head dim = 16
vocab = 256
~~~

先写 Python reference，再写等价的 C++ CPU。实现并逐 tensor 对比：RMSNorm、linear、RoPE、
causal attention、GDN、SwiGLU 与 residual。

**验收：** 秒级运行；每项算子与 Python 对齐；失败时可定位到单个算子/单个 token。
这一阶段不要求输出有意义文字。它是后续 CPU kernel、CUDA kernel 和优化 kernel 的
永久 correctness test，不能跳过。

## 6. Phase 1 — Qwen3.5-0.8B CPU reference

**目标时间：3–5 天。**

将 tiny shape 换成真实 Qwen3.5-0.8B text backbone，保持最朴素、最可读的 CPU loops：

~~~text
tokens → embedding → 24 layers
                    (GDN, GDN, GDN, Attention) × 6
       → final RMSNorm → tied lm_head → logits
~~~

完成：固定模型 loader、权重打包、prefill、decode、KV cache、GDN state、conv state 和
greedy sampling。linear 可以很慢；本阶段唯一优先级是正确。

**验收：** 官方 HF logits 与 CPU logits 在约定误差内一致；`Hello` 能正常续写；固定 prompt
的 greedy token 序列稳定。当前 `lessons/09_qwen35_0_8b.cpp` 就是这一阶段的课程 capstone。

## 7. Phase 2 — 明确 prefill、decode 与 state

**目标时间：2–3 天。**

在真实 0.8B engine 中把两条执行路径写清楚：

~~~cpp
prefill(tokens);  // [T, H]；大量 token，一次初始化 state
decode(token);    // [1, H]；每次生成一个 token，复用 state
~~~

状态应显式属于一条序列，而不是藏在全局变量中：

~~~cpp
struct SequenceState {
    AttentionKV kv[...];
    GDNState    gdn[...];
    ConvState   conv[...];
};
~~~

**验收：** `prefill(ABCD) → decode(E) → decode(F)` 与 reference 的同一序列一致；不会为
E 或 F 重新计算 ABCD。到这里，已经是一个真正的 batch=1 LLM inference engine。

## 8. Phase 3 — CUDA first-correct version

**目标时间：4–7 天。**

目标不是快，而是**权重、activations 和 state 都留在 GPU 上，并正确生成**。

- linear / lm_head：首先使用 cuBLAS；
- 自己写最直接的 CUDA kernels：RMSNorm、RoPE、SiLU/multiply、residual、GDN、attention；
- 先学并只使用必要的 CUDA 基础：`cudaMalloc`、`cudaMemcpy`、kernel launch、grid/block、
  `threadIdx`、`blockIdx`、shared memory 与 warp 基础；
- 每个 kernel 都由 0.8B CPU reference 比较，不以“文本看起来对”代替数值测试。

**大 milestone：**

~~~sh
./qwen --model qwen35-0.8b.bin --cuda
~~~

能持续聊天，且 CUDA 与 CPU 的逐层/端到端 regression 均通过。

## 9. Phase 4 — 切换到 Qwen3.8-27B

**目标时间：2–4 天。**

如果 0.8B 版本没有把 shape 无意写死在算法中，这一步应该主要是替换固定模型描述与权重：

- config、layer count、hidden/intermediate/head shapes；
- 真实 GDN shape 与 attention shape；
- 27B BF16 权重格式与内存检查。

仍然不追 TPS、不引入量化、不做多卡。先在 DGX Spark 上证明：

~~~sh
./qwen --model qwen38-27b.bin --cuda
~~~

可以正确 prefill、decode、正常生成。**这是“27B CUDA 正确跑起来”的 milestone。**

## 10. Phase 5 — Prefix cache

**目标时间：2–4 天。**

对单用户场景，prefix cache 的实际价值高于 batching。第一版只做 longest-prefix match：

~~~text
system + user1 + assistant1
-------------------------- checkpoint
                  + user2 + assistant2
~~~

一个 checkpoint 必须完整保存/恢复：

- attention 的 KV blocks；
- 每个 DeltaNet layer 的 recurrent state；
- 每个 DeltaNet layer 的 conv state；
- prefix 对应 token ids 与 length。

缓存命中后恢复最近的最长 state，只 prefill 新增加的 tokens。第一版每轮对话结束创建
checkpoint 即可；block hashing、LRU、共享 cache 与 eviction 都留到以后。

**验收：** 命中 prefix 的输出必须与从头 prefill 完全等价，同时能测到省去的 prefill tokens。

## 11. Phase 6 — 日常可用 V1

**目标时间：3–5 天。**

补齐外围，而不改造模型核心：

- tokenizer 和 chat template 包装；
- temperature、top-k、top-p、seed；
- EOS、stop strings、context length；
- token streaming、Ctrl-C / cancellation；
- 极薄的 OpenAI-compatible HTTP server：`POST /v1/chat/completions` 与 `stream=true`。

目标使用方式：

~~~sh
./qwen-server --model qwen38-27b.bin --port 8000
~~~

**V1 验收：** 自己的 IDE、agent 或聊天客户端可连接；同一用户连续对话命中 prefix cache；
错误、取消和 EOS 都不会破坏下一次请求的 state。

## 12. Phase 7 — 用 profiling 学性能

优化之前必须先测量。用 Nsight Systems / Nsight Compute 建立固定 benchmark，并记录：

~~~text
token latency
  ├─ FFN
  ├─ GDN projections / GDN state update
  ├─ attention
  ├─ lm_head
  └─ kernel launch gaps

HBM bandwidth / SM utilization / occupancy
~~~

先得到瓶颈，再改一个因素、benchmark、用 Nsight 解释结果。不能因为 kernel “看起来应该更快”
就合并或改 layout。

### 7a. CUDA 学习重点：GDN

从已知数学开始：

~~~text
old   = k @ S
delta = (v - old) * beta
S    *= decay
S    += outer(k, delta)
out   = q @ S
~~~

按以下顺序实验：naive kernel → one block/head → coalesced access → warp reduction → shared
memory → 减少 global IO → fuse `decay + old + delta + update + query`。每一步都保留 correctness
test、benchmark 和 Nsight 记录。

### 7b. 第二重点：decode attention

固定 GQA、batch=1、连续 KV 的 decode attention 可以从最清晰的实现开始：

~~~text
Q → scan K → QK → online softmax → scan V → accumulate
~~~

先写 naive attention，再写 fused decode attention；只有此后才阅读 FlashAttention / FlashInfer，
用它们解释自己的实现为什么需要分块、online softmax 和更少的内存读写。

### 7c. 第三重点：decode GEMV

prefill GEMM 可以长期交给 cuBLAS。模型稳定后再把 decode linear 当作 CUDA 学习练习：

~~~text
naive GEMV → warp reduction → vectorized load → half2 → layout → quantized GEMV
~~~

每版与 cuBLAS 和上一版本同时比较正确性与性能。不要为了“全自研”重写成熟 GEMM。

## 13. Quantization、batching 与以后

### Quantization

先保证 BF16 正确。之后按 Q8 / Q4 / FP8 学习并优先复用成熟格式或 kernel，重点理解：block
quantization、scale、zero point、weight-only、activation quantization，以及 dequant + GEMV
fusion 为什么能减少 decode 的带宽压力。不要把项目变成“维护 20 种 GGUF quant”的工程。

### Batching

单用户、单 GPU 时，feature priority 是：

~~~text
正确性 → CUDA → 27B → prefix cache → CLI/server → 性能 → quant → 长上下文
                                                              ↓
                           之后才是 static batching → continuous batching
                                      → paged KV → scheduler
~~~

做到 batching 才从 inference engine 进入 serving engine；这可以是半年后的独立学习路线。

## 14. 里程碑与时间预期

~~~text
Tiny Python reference
        ↓
Tiny C++ reference
        ↓
Qwen3.5-0.8B CPU
        ↓
prefill / decode / state
        ↓
CUDA + cuBLAS
        ↓
0.8B CUDA correct
        ↓
Qwen3.8-27B CUDA correct
        ↓
prefix cache
        ↓
CLI + streaming OpenAI API
        ↓
★ 日常可用 V1 ★
        ↓
Nsight profiling → GDN / attention / GEMV optimization
        ↓
Q8 / Q4 / FP8
        ↓
★ 高性能 V2 ★
~~~

| 集中开发的累计时间 | 目标 |
| --- | --- |
| 2–3 天 | Tiny Python/C++ reference 跑通 |
| 约 1 周 | 0.8B CPU 正确生成 |
| 约 2 周 | 0.8B CUDA 正确 |
| 2–3 周 | 27B CUDA 正确运行 |
| 3–4 周 | 27B + prefix cache + server，开始自己用 |
| 4–6 周 | CUDA profiling 与第一轮性能优化 |
| 2–3 个月 | 一个自己能从 token 到 kernel 全部解释的 inference engine |

AI coding 能显著压缩 converter、loader、HTTP、CUDA API 调用、测试 boilerplate 和 build
脚本；它不能替代数值 debug 与性能理解。最有价值的问题仍然是：为什么 layer 17 开始 logits
不对、为什么 GDN state 在第 200 token 漂移、为什么 bandwidth 只有理论值的 35%、为什么换
layout 反而变慢。能亲自沿着这些问题解释 token → hidden → GDN/attention → FFN → state →
decode → CUDA kernel → Nsight，这个项目就达成学习目标了。
