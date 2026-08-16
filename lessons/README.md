# qwen3x：从零实现 Qwen 风格 inference 的课程

这里的目标不是一次读懂根目录中完整的 prototype，而是每次只增加一个
概念，并得到一个可以独立编译、独立验证的 C++ 程序。

每一课都遵守四条规则：

1. 不依赖 CUDA、PyTorch 或 Tensor framework；
2. 不继承上一课的隐藏库；少量重复是故意的，读者可以只打开当前文件；
3. 运行时先用玩具维度和手写权重，确保结果可以手算；
4. 完成课程后再进入同级 `../qwen35-0.8b/`，验证同一套计算能跑真实
   Qwen3.8-style hybrid 架构。

## 课程地图

| 课 | 文件 | 新增概念 | 当前状态 |
| --- | --- | --- | --- |
| 00 | 00_toy_logits.cpp | token id、embedding、tied lm_head、argmax | 已完成 |
| 01 | 01_rmsnorm_linear.cpp | Qwen ordinary RMSNorm、线性层 | 已完成 |
| 02 | 02_swiglu_residual.cpp | SwiGLU FFN、residual | 已完成 |
| 03 | 03_rope.cpp | RoPE | 已完成 |
| 04 | 04_attention.cpp | causal attention | 已完成 |
| 05 | 05_gqa_kv_cache.cpp | GQA、prefill、decode、KV cache | 已完成 |
| 06 | 06_deltanet_recurrence.cpp | Gated DeltaNet 的固定 recurrent state | 已完成 |
| 07 | 07_deltanet_layer.cpp | Q/K/V 投影、causal conv、门控、DeltaNet layer | 已完成 |
| 08 | 08_hybrid_qwen.cpp | 3 DeltaNet : 1 attention 的 Qwen hybrid stack | 已完成 |

课程文件的总行数不会限制在 1000 行，因为每课为了自包含会重复少量代码；
限制的是同级 [qwen35-0.8b/qwen35.cpp](../qwen35-0.8b/qwen35.cpp)。否则会为了
“少几行”而把关键细节藏进帮助函数，反而失去教学价值。

## 运行

本目录有自己的 Makefile；进入本目录后，`make` 只编译，`make test` 才会运行：

~~~
cd lessons
make
make test
~~~

也可以单独编译一课：

~~~
c++ -O2 -std=c++17 00_toy_logits.cpp -o 00_toy_logits
./00_toy_logits
~~~

## 完成课程后

真实官方 0.8B 的固定权重程序、转换器与回归脚本都在同级
[qwen35-0.8b](../qwen35-0.8b/README.md)。它刻意不混在本目录：这里始终只保留
可手算、可逐步阅读的 lesson。

## 与旧 prototype 的关系

旧的完整 reference prototype 已由 git tag `prototype-v0` 保留。它不在当前
工作树中：这个仓库只保留从玩具模型到最终 capstone 的教学路径。
