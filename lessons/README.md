# qwen38.cpp：从零实现 Qwen 风格 inference 的课程

这里的目标不是一次读懂根目录中完整的 prototype，而是每次只增加一个
概念，并得到一个可以独立编译、独立验证的 C++ 程序。

每一课都遵守四条规则：

1. 不依赖 CUDA、PyTorch 或 Tensor framework；
2. 不继承上一课的隐藏库；少量重复是故意的，读者可以只打开当前文件；
3. 运行时先用玩具维度和手写权重，确保结果可以手算；
4. 最终才切换到 Qwen3.5-0.8B 的 text backbone，验证同一套计算能跑真实
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
| 07 | 07_deltanet_layer.cpp | Q/K/V 投影、causal conv、门控、DeltaNet layer | 待实现 |
| 08 | 08_hybrid_qwen.cpp | 3 DeltaNet : 1 attention 的 Qwen hybrid stack | 待实现 |
| 09 | 09_load_08b.cpp | 固定 0.8B 的简单二进制权重格式 | 待实现 |
| 10 | 10_tokenizer_generate.cpp | 官方 tokenizer 库、采样、文本生成 | 待实现 |
| 11 | qwen38.cpp | 小于 1000 行的整合 CPU 版本 | 待实现 |

课程文件的总行数不会限制在 1000 行，因为每课为了自包含会重复少量代码；
限制的是最后的整合版 qwen38.cpp。否则会为了“少几行”而把关键细节藏进
帮助函数，反而失去教学价值。

## 运行

在仓库根目录执行：

~~~
make lesson-test
~~~

也可以单独编译一课：

~~~
c++ -O2 -std=c++17 lessons/00_toy_logits.cpp -o lesson00
./lesson00
~~~

## 与现有 prototype 的关系

当前根目录的 qwen38.cpp 是已经验证过的完整 reference prototype，已被
本地 git tag prototype-v0 固定保存。课程会从玩具模型重新写起，而不是把
那个大文件拆碎；后者仍可作为最终课的数值与结构参考。
