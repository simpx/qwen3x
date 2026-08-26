# qwen3x

一个循序渐进的 C++ 课程：从零写出 Qwen3.8-style hybrid LLM inference。

这不是 llama.cpp、vLLM 或通用推理框架。目标是让读者按文件顺序理解：

    token -> embedding -> RMSNorm -> DeltaNet / attention -> residual
          -> SwiGLU FFN -> logits -> next token

最终答案是少于 1000 行的 CPU 程序，运行官方 Qwen3.5-0.8B 的 text backbone。
它与 Qwen3.8-27B 共享 Qwen3.5 hybrid 结构：每四层是三个 Gated DeltaNet
layer 加一个 Gated Attention layer。

## 为什么是 0.8B，而不是 Qwen2.5-0.5B

Qwen2.5-0.5B 是纯 attention Transformer，能教 embedding、RoPE、GQA 与 KV
cache，但它没有 DeltaNet recurrent state、DeltaNet convolution 和 3:1 hybrid
stack。Qwen3.5-0.8B 是目前足够小、又真正保留目标架构的官方 checkpoint。

0.8B 与 27B 的尺寸、层数和 embedding/lm_head tied 策略不同；课程不会掩盖
这些差异。它的价值是：同一套核心 forward 概念可以端到端运行真实权重。

## 从这里开始

不要先打开两千行的 prototype。先阅读 [00-lessons/README.md](00-lessons/README.md)，
进入课程目录编译全部手写 lesson：

~~~
cd 00-lessons
make
~~~

再运行每一课：

~~~
make test
~~~

课程每一步都是独立、可编译的 C++ 文件：

| 课 | 新概念 |
| --- | --- |
| 00 | token id、embedding、tied lm_head、logits |
| 01 | RMSNorm、linear |
| 02 | SwiGLU、residual |
| 03 | RoPE |
| 04 | causal attention |
| 05 | GQA、KV cache、prefill/decode |
| 06 | Gated DeltaNet recurrent state |
| 07 | 完整 toy DeltaNet layer |
| 08 | 3 DeltaNet : 1 attention hybrid stack |
| 09 | 真实 Qwen3.5-0.8B：权重加载、完整 forward、prefill 和 decode |

`00-lessons/` 是第 0 章：完整的 CPU 教学路径。第 00--08 课使用可手算 toy dimensions，第 09 课把同一
顺序直接扩大为真实 0.8B 权重。CUDA 不放进 lesson，避免读到模型数学时被 GPU runtime、库句柄
和性能细节打断。

## 课程之后：真实 0.8B 的验证阶梯

`00-lessons/` 解释数学并保持冻结；从真实权重、CPU、CUDA 到更大模型则按下面几个独立 stage 前进。
它们刻意少量复制代码，避免为了共享而引入 `Tensor`、`Backend` 或 engine/session 框架。完整方向见
[roadmap.md](roadmap.md)。

| 目录 | 做什么 | 主要验收命令 |
| --- | --- | --- |
| [reference/](reference/README.md) | 官方 PyTorch/Transformers 的 CPU/CUDA FP32 logits oracle | `make -C reference cpu` / `cuda` |
| [02-cpu-0.8b/](02-cpu-0.8b/README.md) | 独立 plain C++ CPU forward + 文字/chat e2e | `make -C 02-cpu-0.8b test` |
| [03-cuda-0.8b/](03-cuda-0.8b/README.md) | 直接 CUDA kernels + cuBLAS，state 留在 GPU | `make -C 03-cuda-0.8b test` |
| [04-runtime/](04-runtime/README.md) | Python OpenAI-compatible server + 常驻 CPU/CUDA engine | `make -C 04-runtime test` / `e2e` |
| [05-qwen38-27b/](05-qwen38-27b/README.md) | 官方 27B config/tensor schema/显存 preflight（不下载权重） | `make -C 05-qwen38-27b test` |
| [qwen35-0.8b/](qwen35-0.8b/README.md) | 正式的同进程 Engine/Session C ABI + Python Session Slots/OpenAI runtime | `make -C qwen35-0.8b test` / `e2e` |

每一层都用前一层无法伪造的证据验证：HF full logits → C++ CPU full logits → CUDA full logits →
真实 prompt 的 tokenizer ids 与 greedy text。CPU 目录本身已经包含 tokenizer 薄壳和文字 e2e；
C++ inference core 不依赖 Python。

## 真实 0.8B

[00-lessons/09_qwen35_0_8b.cpp](00-lessons/09_qwen35_0_8b.cpp) 是约 520 行的整合版。它严格固定为
Qwen3.5-0.8B text backbone，采用 BF16 权重、FP32 计算，支持真实 prefill、
decode、DeltaNet state、KV cache 和 greedy generation。

~~~
cd 00-lessons
make
python3 09_pack_weights.py ../models/Qwen3.5-0.8B ../models/qwen35-0.8b.bin
./09_qwen35_0_8b --generate ../models/qwen35-0.8b.bin 248044,198,198 16
~~~

详细模型格式和 regression 请看
[00-lessons/README.md](00-lessons/README.md)。

有本地官方 checkpoint 时：

~~~
cd 00-lessons
make model-test MODEL=../models/Qwen3.5-0.8B
~~~

该测试会临时转换权重，对照已固定的官方权重结果：三 token prompt 的
next-token 必须为 198，logit 误差必须低于 1e-3，并比较八个 greedy decode
token。

## 教学与 production runtime 的边界

- 通用 Hugging Face 模型、GGUF 或量化格式
- Metal、通用 Tensor/Operator/Backend 抽象框架
- vision、MTP、多模态、训练或 LoRA
- 教学 stage 内的 continuous batching、并行与分布式

这些范围控制仍适用于 `00-lessons` 和编号 correctness stage。完成课程以后，
[qwen35-0.8b](qwen35-0.8b/README.md) 才是可演进的正式 runtime：C++ 定义共享 Engine 和
可变 Session，Python 预创建 Session Slots 并负责 HTTP、tokenizer、streaming 和调度。
它直接复用已验证的固定模型数学，但不再受课程代码行数限制。

旧 prototype 已保留在远端 git tag prototype-v0；它不在当前教学仓库的
工作树中。
