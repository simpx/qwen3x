# 真实 Qwen3.5-0.8B text inference

qwen35.cpp 是课程后的真实权重整合答案，目前少于 1000 行。它不是通用 runtime：

- 固定 Qwen3.5-0.8B 的 24 层 text backbone；
- CPU、batch 1、贪婪生成、最多 2048 token；
- BF16 权重即时转 FP32，DeltaNet recurrent state 与 KV cache 都保留 FP32；
- 同时识别 Qwen text EOS 与 chat template 的 assistant 回合结束 token；
- 忽略视觉 encoder、MTP、量化、Metal、通用 safetensors 兼容和 CUDA 性能优化。

这个 0.8B 模型与 Qwen3.8-27B 共享 Qwen3.5 hybrid text 架构：每四层中前三层
是 Gated DeltaNet，第四层是 Gated Attention。尺寸、层数和 embedding/lm_head
tied 策略不同，但 forward 的概念和顺序相同。

## 运行 token-id 版本

先下载官方 Qwen3.5-0.8B checkpoint 到仓库根目录的
`models/Qwen3.5-0.8B`；随后进入本目录并转换：

~~~
cd qwen35-0.8b
make
python3 pack_weights.py ../models/Qwen3.5-0.8B out/qwen35-0.8b.bin
./qwen35 --forward out/qwen35-0.8b.bin 248044,198,198
./qwen35 --generate out/qwen35-0.8b.bin 248044,198,198 16
~~~

转换器逐 tensor 拷贝原始 safetensors 字节，所以不需要 PyTorch、NumPy 或
safetensors Python 包。它的输出约 1.4 GiB：只包含 language model 所需的
320 个 text tensors，不包含视觉和 MTP tensors。

有本地官方 checkpoint 时，可运行以下 regression；它会临时转换权重，并检查
一个已固定的官方权重 forward logit 和八个 greedy token：

~~~
make oracle-test MODEL=../models/Qwen3.5-0.8B
~~~

本课程的 C++ 只接受和输出 token id。文本 tokenizer 是独立外围工具：可用
官方 Python tokenizer、Transformers 或任意兼容工具把 text 编码为这些 id，
再将生成 id 解码回来；它不进入本仓库的 C++ build。

## 可选 CUDA backend

CPU `qwen35.cpp` 永远是直接、可读的 correctness reference。CUDA 不修改它，
而是作为平行的 `qwen35_cuda.cu` 编译为另一个 binary：权重、hidden、KV cache 与
DeltaNet recurrent state 都常驻 GPU；矩阵向量乘和模型专属小算子直接在 CUDA 上执行。

~~~
make cuda                 # 当前 RTX 4080 SUPER 默认使用 sm_89
make cuda-test            # 不读取 checkpoint，只确认 GPU runtime 可用
make cuda-oracle-test MODEL=../models/Qwen3.5-0.8B
~~~

最后一个命令会在相同官方权重上比较 CPU 与 CUDA 的 prefill next-token/logit，并比较
八步 greedy decode。不同 reduction 顺序会带来极小 FP32 logit 差异，当前容差为 `1e-3`。
其他 GPU 可以覆盖 arch，例如 `make cuda CUDA_ARCH=120`。
