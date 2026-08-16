# 最后一课：真实 Qwen3.5-0.8B text inference

capstone/qwen38.cpp 是课程的整合答案，目前少于 1000 行。它不是通用 runtime：

- 固定 Qwen3.5-0.8B 的 24 层 text backbone；
- CPU、batch 1、贪婪生成、最多 2048 token；
- BF16 权重即时转 FP32，DeltaNet recurrent state 与 KV cache 都保留 FP32；
- 同时识别 Qwen text EOS 与 chat template 的 assistant 回合结束 token；
- 忽略视觉 encoder、MTP、量化、CUDA、Metal 和通用 safetensors 兼容。

这个 0.8B 模型与 Qwen3.8-27B 共享 Qwen3.5 hybrid text 架构：每四层中前三层
是 Gated DeltaNet，第四层是 Gated Attention。尺寸、层数和 embedding/lm_head
tied 策略不同，但 forward 的概念和顺序相同。

## 运行 token-id 版本

先下载官方 Qwen3.5-0.8B checkpoint 到 models/Qwen3.5-0.8B，然后转换：

~~~
python3 convert.py models/Qwen3.5-0.8B out/qwen38-0.8b.bin
make course-test
./qwen38_course --forward out/qwen38-0.8b.bin 248044,198,198
./qwen38_course --generate out/qwen38-0.8b.bin 248044,198,198 16
~~~

转换器逐 tensor 拷贝原始 safetensors 字节，所以不需要 PyTorch、NumPy 或
safetensors Python 包。它的输出约 1.4 GiB：只包含 language model 所需的
320 个 text tensors，不包含视觉和 MTP tensors。

有本地官方 checkpoint 时，可运行以下 regression；它会临时转换权重，并将
capstone 的 logits 与八个 greedy token 同已保留的完整 CPU prototype 比较：

~~~
make course-oracle-test MODEL=models/Qwen3.5-0.8B
~~~

## 文本版本

CMake 默认会编译官方 tokenizer.json 的轻量 C ABI adapter：

~~~
cmake -S . -B build
cmake --build build --target qwen38_course -j
./build/qwen38_course --generate-text out/qwen38-0.8b.bin models/Qwen3.5-0.8B '你好' 16
~~~

Tokenizer 是外围基础设施，不计入 capstone 的模型数学代码；它只负责 text 和
token id 的互相转换。
