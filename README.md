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

不要先打开两千行的 prototype。先阅读 [lessons/README.md](lessons/README.md)，
进入课程目录编译全部手写 lesson：

~~~
cd lessons
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

`lessons/` 只放这些可手算的教学文件、它们的 bin 和 Makefile。真实 0.8B 的
权重程序另放在同级 [qwen35-0.8b](qwen35-0.8b/README.md)。

## 真实 0.8B

[qwen35-0.8b/qwen35.cpp](qwen35-0.8b/qwen35.cpp) 是约 520 行的整合版。它严格固定为
Qwen3.5-0.8B text backbone，采用 BF16 权重、FP32 计算，支持真实 prefill、
decode、DeltaNet state、KV cache 和 greedy generation。

~~~
cd qwen35-0.8b
make
python3 pack_weights.py ../models/Qwen3.5-0.8B out/qwen35-0.8b.bin
./qwen35 --generate out/qwen35-0.8b.bin 248044,198,198 16
~~~

详细模型格式和 regression 请看
[qwen35-0.8b/README.md](qwen35-0.8b/README.md)。

有本地官方 checkpoint 时：

~~~
cd qwen35-0.8b
make oracle-test MODEL=../models/Qwen3.5-0.8B
~~~

该测试会临时转换权重，对照已固定的官方权重结果：三 token prompt 的
next-token 必须为 198，logit 误差必须低于 1e-3，并比较八个 greedy decode
token。

## 明确不做

- 通用 Hugging Face 模型、GGUF 或量化格式
- CUDA、Metal、BLAS 性能优化
- vision、MTP、多模态、训练或 LoRA
- continuous batching、服务、并行与分布式
- Tensor/Operator/Backend 抽象框架

这些不是缺失功能，而是为了让核心 inference 可读而做的范围控制。未来真正的
C++ 性能引擎会另开仓库；它可以从这里已验证的数学与测试开始，但不受一千行
限制。

旧 prototype 已保留在远端 git tag prototype-v0；它不在当前教学仓库的
工作树中。
