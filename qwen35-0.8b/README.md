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

### 最强回归：官方 FP32 end-to-end oracle

`oracle-test` 使用固定值防止日常重构回归；`official-oracle` 则是更强的开发期测试。
它将一条真实 text chat 用 checkpoint 自带的官方 tokenizer/template 编码，并以相同的
token ids 分别运行官方 Transformers FP32 model、`qwen35` 与 `qwen35_cuda`。它检查：

- 全部 248,320 个 prefill logits 的 max/mean absolute error 与 argmax；
- 八步 greedy decode token，因而覆盖 attention KV cache 和 DeltaNet recurrent state；
- CPU 与 CUDA 都直接对官方 reference，而不只是彼此相同。

Qwen3.5 支持需要较新的 Transformers；这个只用于开发期 oracle，不是 C++ runtime 依赖：

~~~
pip install 'transformers>=5.0' torch
make official-oracle MODEL=../models/Qwen3.5-0.8B

# 已有权重包时可跳过临时 pack：
make official-oracle MODEL=../models/Qwen3.5-0.8B WEIGHTS=out/qwen35-0.8b.bin
~~~

测试有意让官方模型以 FP32 运行。本项目的语义是“BF16 权重 + FP32 activation/state”；
默认要求全词表 max absolute error 不超过 `1e-4`。若把官方 model 以 BF16 运行，每一层
matmul 的舍入会放大为明显的最终 logit 差异，不能
当作这个 CPU/CUDA reference implementation 的数值黄金值。

## Chat：官方 tokenizer 留在 Python

`chat.py` 是不到 130 行的外围 wrapper，不属于 C++ inference core。它读取官方
checkpoint 的 tokenizer 和 chat template，调用 C++ binary，最后 decode 输出 ids：

~~~
# 第一次只需安装文字外围依赖；它不是 C++ runtime dependency。
pip install transformers

# 先创建一次 C++ 可 mmap 的权重包。
python3 pack_weights.py ../models/Qwen3.5-0.8B out/qwen35-0.8b.bin

# 单轮 chat（默认 CPU）。
python3 chat.py --model ../models/Qwen3.5-0.8B \
  --weights out/qwen35-0.8b.bin --prompt "用一句话介绍 DeltaNet。"

# 同一套 tokenizer/chat template，换成 CUDA executable。
python3 chat.py --cuda --model ../models/Qwen3.5-0.8B \
  --weights out/qwen35-0.8b.bin --prompt "用一句话介绍 DeltaNet。"

# 多轮演示；为保持 C++ 课程可读性，每回合会重新 prefill 完整 history。
python3 chat.py --cuda --interactive --model ../models/Qwen3.5-0.8B \
  --weights out/qwen35-0.8b.bin
~~~

传 `--show-ids` 可以看到 Python/C++ 边界上的 prompt 和 output token ids。这样 chat
template、Unicode、history 都在 Python，而 `qwen35.cpp` 和 `qwen35_cuda.cu` 继续只做
模型 forward。

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
