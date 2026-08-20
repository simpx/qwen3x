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

## 2. 目录就是阶段快照

`lessons/` 保留为**前置课程**，不再承担新 engine 的演进。它解释模型数学；之后的代码
按编号目录前进，像 buildyourownllm 的每个版本快照一样。打开一个目录应能知道：此时模型
能做什么、仍然不能做什么、怎样一条命令验证它。

~~~text
lessons/                 前置课程：toy math + 已有的 0.8B CPU capstone；尽量冻结

01-hf-reference/         Stage 1：官方 Qwen3.5-0.8B 的 PyTorch / Transformers oracle
02-cpu-0.8b/             Stage 2：真实 0.8B 的可读 C++ CPU forward + state
03-cuda-0.8b/            Stage 3：同一 0.8B 的 CUDA first-correct version
04-0.8b-e2e/             Stage 4：0.8B 的 CLI wrapper、test vectors、端到端集成测试
05-qwen38-27b/           Stage 5：固定 Qwen3.8-27B CUDA correctness
06-prefix-cache/         Stage 6：longest-prefix state snapshot / restore
07-server/               Stage 7：tokenizer wrapper、CLI、streaming OpenAI-compatible API
08-performance-lab/      Stage 8：benchmark、Nsight、GDN / attention / GEMV 实验
~~~

每个 stage 是一个**可独立阅读、可独立编译测试的快照**。少量复制是有意的；不要用
`common/`、Tensor 类或 backend interface 把读者带出当前目录。后阶段可以读取前阶段生成的
测试向量、权重格式和已验证公式，但不会在源码层 `#include` 前一阶段。

目前的 `qwen35-0.8b/` 是已有 CUDA 性能实验，保留不动作为证据和对照。等 Stage 2 的 CPU
`prefill/decode/state` 语义固定、Stage 3 的测试从零跑绿后，再将其有价值的实现迁入 `03-cuda-0.8b/`；不要在
今晚为了目录漂亮而移动正在工作的 CUDA 代码。

未来每个 stage 内的实现文件可以很直接，例如 `model.cpp`、`weights.cpp`、`sampler.cpp`、
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
lessons/ 的手算 toy tests
             ↓  已理解的公式
official Python / HF reference（Qwen3.5-0.8B）
             ↓  test vectors + logits
Qwen3.5-0.8B C++ CPU engine
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

## 今晚的自动执行计划：完成 0.8B 基础，停在 27B 之前

`lessons/00--08` 已经承担 tiny/手算练习，`lessons/09` 已证明真实 0.8B CPU forward。因此
今晚不再复制一套 toy Python/C++ 模型；直接把这两个已有资产变成 **真实 0.8B 的 oracle、CPU
engine、CUDA engine 和集成测试链**。目标是完成 Stage 1--4，明天切 27B 时不再修改基础语义。

### 从 ds4 借用的做法，而不是接口

ds4 值得借鉴的是模型专用、mmap 权重、CPU 只做 reference、测试先于速度这些做法。不要照搬它的
公开 `engine/session` API：我们当前只服务 batch=1 的 Qwen，直接的三块数据已经足够。

~~~cpp
struct Model;  // mmap 的固定 0.8B weights。
struct State;  // position、KV cache、GDN state、conv state；跨 token 保留。
struct Work;   // 一次 forward 的 FP32 scratch 和 logits；下一 token 可覆盖。

void forward(const Model&, State&, int token, Work&);
void prefill(const Model&, State&, const int* tokens, int count, Work&);
void decode(const Model&, State&, int token, Work&);
~~~

这不是 abstract backend。每个 stage 仍有明确的 `mv_cpu()` 或 `mv_cuda()`；`prefill()` 和
`decode()` 也只是有名字的循环，内部仍直接调用 `forward()`。不同请求一律 `State{}` 重建；跨请求
的最长前缀复用只在 Stage 6 用明确的 snapshot cache 实现。

### Stage 1 — `01-hf-reference/`：官方 0.8B oracle

这里使用 **PyTorch + Transformers**，因为目的不是再写一份模型，而是调用官方 checkpoint 作为
裁判。当前仓库已经有本地 `models/Qwen3.5-0.8B`，所以默认命令不需要下载模型。CPU FP32 和 CUDA
FP32 的 vectors 分别保存到 `build/cpu/` 与 `build/cuda/`；测试会拒绝混用它们。

~~~text
01-hf-reference/
  README.md
  Makefile
  dump_vectors.py    # 直接调用官方 model；固定 FP32 / token ids；输出 logits 与 greedy ids
  dump_vectors.py    # 生成 versioned binary test vectors；可选 module hooks dump layer tensors
  test_reference.py  # 验证 tokenizer、config fingerprint、vector 可重现性
~~~

默认 vector 覆盖短 prefill、`prefill(ABCD)+decode(E)+decode(F)`、greedy continuation 与一个
真实官方 chat-template prompt。metadata 记录 checkpoint/tokenizer fingerprint、device、dtype、input
ids、chat messages 与允许误差；不能把“目前本机跑出来的数”变成无来源 golden file。

**验收：** `make -C 01-hf-reference cpu MODEL=../models/Qwen3.5-0.8B` 和同样的 `cuda` target
各生成一份可复现 vectors；第二次运行得到同一 config/input 和同一 greedy ids。默认只比较官方
输出，不做 performance benchmark。

### Stage 2 — `02-cpu-0.8b/`：真实 C++ CPU baseline

以 `lessons/09_qwen35_0_8b.cpp` 为已验证数学来源，复制为一个可演进的、仍然模型专用的 CPU
snapshot。保留朴素的 `Model/State/Work` 三个 struct；只将隐含在 `generate()` 的 prefill/decode
循环写成两个小函数。不要在这一步“重构成框架”。

~~~text
02-cpu-0.8b/
  README.md
  Makefile
 qwen35.cpp          # 直接的 CPU forward、Model / State / Work
 pack_weights.py     # HF safetensors → 固定、mmap-friendly qwen35-0.8b.bin
 test_cpu.py         # 读取 Stage 1 vectors，逐项比较 CPU 与官方结果
~~~

**验收：** `make test MODEL=...` 同时跑 state tests、完整 vocabulary logits、greedy ids 与
`prefill(ABCD) → decode(E) → decode(F)` 的 state 等价测试。trace failure 会报告 case、最坏 token
step 与 max-abs error；CPU 就是以后所有 CUDA kernel 的数学 oracle，不追性能。

### Stage 3 — `03-cuda-0.8b/`：CUDA first-correct

以现有 `qwen35-0.8b/` 的 cuBLAS、GPU-resident weights/state、GPU argmax 和 regression 作为
材料迁入；不从头发明更快的实现。

~~~text
03-cuda-0.8b/
 README.md
 Makefile
  qwen35_cpu.cpp      # 同一 stage 内的 mmap loader、固定 structs、CPU helpers
  qwen35_cuda.cu      # fixed 0.8B CUDA forward
  test_cuda.py        # CPU vector / CUDA vector 的完整比较
~~~

linear 与 lm_head 首先用 cuBLAS `sgemv`；checkpoint BF16 linear matrix 在加载时展开为 FP32，使
CUDA correctness path 与 CPU 的 FP32 activation 语义一致。RMSNorm、RoPE、SiLU/multiply、residual、
GDN、attention 是直白 kernels。所有 weights、work buffers、KV/GDN/conv state 必须留在 GPU；正常
decode 不得下载全 vocabulary logits。

**验收：** `make cuda-test` 通过无权重 self-test；`make test MODEL=...` 生成独立 CUDA official
vectors，再比较全 logits、greedy ids、prefill/decode/state。GPU 有自己的 reduction order 与 vector
bundle，但门槛必须写在 metadata/README 中，不能临时放宽。

### Stage 4 — `04-0.8b-e2e/`：将 0.8B 测试封口

这个目录不重新实现 layer math。它以 Stage 2/3 的 binary 作为黑盒，完成以后迁 27B 必须具备的
测试契约：

~~~text
04-0.8b-e2e/
 README.md
 Makefile
  qwen35_chat.py      # 官方 tokenizer / chat template 的薄包装，不进入 C++ forward
  test_e2e.py         # prompt → tokens → generate → decode 的固定 contract test
~~~

此时仍不做 HTTP server 或跨 request prefix cache；但必须将 token-id CLI 与 state 测试固定下来，保证
接下来只替换 model dimensions/weights 而不是重做基础循环。可保留 MMLU-Pro short coverage 作为慢速
质量回归，但它不能代替数值 vectors。

### 今晚的严格顺序与停机条件

| 顺序 | 自动任务 | 绿灯命令 | 禁止提前做的事 |
| --- | --- | --- | --- |
| 1 | Stage 1：生成官方 CPU/CUDA test vectors | `make -C 01-hf-reference cpu` / `make -C 01-hf-reference cuda` | 不改 lessons 数学 |
| 2 | Stage 2：从 lesson 09 建真实 CPU `Model/State/Work` API | `make -C 02-cpu-0.8b test MODEL=...` | 不优化 linear、不加 tokenizer |
| 3 | Stage 3：CUDA + cuBLAS correctness path | `make -C 03-cuda-0.8b cuda-test` + `make test` | 不写新 fused kernel |
| 4 | Stage 4：固定 CLI/e2e test contract | `make -C 04-0.8b-e2e test` | 不做 HTTP/prefix snapshot |
| 5 | 总回归与提交 | 四个目录的 Makefile 都绿 | 不开始 27B |

**今晚结束定义：** 真实官方 0.8B 权重能经过 HF → CPU → CUDA → tokenizer wrapper 的完整验证，
并且 `Model + State + Work + prefill/decode` 的行为有固定测试。到这一步才能开始 `05-qwen38-27b/`。
任何 layer/logit/state 不一致都要先停在 0.8B 修复；绝不能携带“不知道为什么”的误差去 27B。

## 5. Stage 1 — 官方 Qwen3.5-0.8B Python oracle

**目标：** 以官方 checkpoint 和 `Qwen3_5ForCausalLM` 建立带来源的 test vectors，而不是用
“C++ 看起来生成了文字”做 golden。PyTorch / Transformers 只在本目录运行；它们永远不链接进
C++ binary。

HF reference 要支持固定 token ids 的 prefill、decode 和 greedy continuation；默认 dump final
logits 与 token ids，并在需要定位误差时以 hooks/官方输出暴露 layer boundary tensors。它还验证
config、tokenizer revision 与权重文件，避免不同 checkpoint 生成的 vector 混用。

**验收：** 任意 Stage 2/3 regression 都能回溯到一个 versioned 的 `01-hf-reference` vector
及其官方模型来源。

## 6. Stage 2 — Qwen3.5-0.8B CPU reference

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

## 7. Stage 2 的完成条件 — 明确 prefill、decode 与 state

**目标时间：2–3 天。**

在真实 0.8B engine 中把两条执行路径写清楚：

~~~cpp
prefill(tokens);  // [T, H]；大量 token，一次初始化 state
decode(token);    // [1, H]；每次生成一个 token，复用 state
~~~

状态应显式属于一条序列，而不是藏在全局变量中：

~~~cpp
struct State {
    AttentionKV kv[...];
    GDNState    gdn[...];
    ConvState   conv[...];
};
~~~

**验收：** `prefill(ABCD) → decode(E) → decode(F)` 与 reference 的同一序列一致；不会为
E 或 F 重新计算 ABCD。到这里，已经是一个真正的 batch=1 LLM inference engine。

## 8. Stage 3 — CUDA first-correct version

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

## 9. Stage 4 — 0.8B runtime integration

在切换 27B 前，将 Stage 2/3 的 `Model + State + Work` 数据流固定为一个真实可调用的 0.8B runtime：
token id CLI、官方 tokenizer/chat template 薄包装、prefill/decode state regression、CPU/CUDA vector
runner 与固定 prompt smoke test。这个 stage 只编排已经正确的 engines；不新增 layer math，也不
开始 HTTP server。

**验收：** `prefill` 与逐 token `decode` 必须等价；reset 后状态不得泄漏；CPU/CUDA 能被同一个
token-id case 调用；全部 0.8B 数值 vector、greedy vector 与文本 smoke test 一条
命令执行。它是 27B 迁移前最后一道 gate。

## 10. Stage 5 — Qwen3.8-27B

**目标时间：2–4 天，在具备足够显存和本地权重之后。**

27B 不是把 0.8B 的 `constexpr` 替换成大数字。0.8B snapshot 刻意为教学固定模型；27B 要另写同样
直接的 `Model/State/Work/forward`，并先通过 [05-qwen38-27b/](05-qwen38-27b/README.md) 的官方 metadata
gate。它已经固定以下事实：

- 64 layers = 48 Gated DeltaNet + 16 full attention；
- `H=5120`、`I=17408`；attention 是 24 query / 4 KV heads、head dim 256；
- DeltaNet 是 16 key / 48 value heads、各 dim 128，conv kernel 4；
- lm_head 不再 tied；checkpoint 总计 1199 tensors，其中 text-only runtime 需要 851 个；
- raw BF16 payload 约 51.75 GiB；这台 16 GiB 4080 无法加载，且绝不能复用 Stage 3 的 FP32
  linear-weight expansion（约 103.5 GiB）。

因此真正的 Stage 5 implementation 要在 DGX Spark 或同等容量机器上以 BF16 weights / BF16 GEMM
路径开始，先建立 27B HF vectors，再实现 text-only loader、prefill、decode 和 state。仍然不追 TPS、
不引入量化、不做多卡。先证明：

~~~sh
./qwen --model qwen38-27b.bin --cuda
~~~

可以正确 prefill、decode、正常生成。**这是“27B CUDA 正确跑起来”的 milestone。**

## 11. Stage 6 — Prefix cache

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

## 12. Stage 7 — 日常可用 V1

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

## 13. Stage 8 — 用 profiling 学性能

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

### Stage 8a. CUDA 学习重点：GDN

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

### Stage 8b. 第二重点：decode attention

固定 GQA、batch=1、连续 KV 的 decode attention 可以从最清晰的实现开始：

~~~text
Q → scan K → QK → online softmax → scan V → accumulate
~~~

先写 naive attention，再写 fused decode attention；只有此后才阅读 FlashAttention / FlashInfer，
用它们解释自己的实现为什么需要分块、online softmax 和更少的内存读写。

### Stage 8c. 第三重点：decode GEMV

prefill GEMM 可以长期交给 cuBLAS。模型稳定后再把 decode linear 当作 CUDA 学习练习：

~~~text
naive GEMV → warp reduction → vectorized load → half2 → layout → quantized GEMV
~~~

每版与 cuBLAS 和上一版本同时比较正确性与性能。不要为了“全自研”重写成熟 GEMM。

## 14. Quantization、batching 与以后

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

## 15. 里程碑与时间预期

~~~text
lessons/：公式与 toy state 已读懂
        ↓
official Qwen3.5-0.8B HF vectors
        ↓
0.8B CPU Model + State + Work
        ↓
0.8B CUDA + cuBLAS
        ↓
0.8B tokenizer wrapper / end-to-end regression
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
| 今晚 | 建好并尽力跑通 0.8B Stage 1–4：official vectors、CPU/CUDA forward、tokenizer/e2e regression |
| 1–3 天 | 修完 0.8B 的所有数值/state 差异，固定 weights、vectors 与 CLI contract |
| 之后 2–4 天 | 27B CUDA 正确运行 |
| 3–4 周 | 27B + prefix cache + server，开始自己用 |
| 4–6 周 | CUDA profiling 与第一轮性能优化 |
| 2–3 个月 | 一个自己能从 token 到 kernel 全部解释的 inference engine |

AI coding 能显著压缩 converter、loader、HTTP、CUDA API 调用、测试 boilerplate 和 build
脚本；它不能替代数值 debug 与性能理解。最有价值的问题仍然是：为什么 layer 17 开始 logits
不对、为什么 GDN state 在第 200 token 漂移、为什么 bandwidth 只有理论值的 35%、为什么换
layout 反而变慢。能亲自沿着这些问题解释 token → hidden → GDN/attention → FFN → state →
decode → CUDA kernel → Nsight，这个项目就达成学习目标了。
