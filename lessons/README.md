# qwen3x：从零实现 Qwen 风格 inference 的课程

这里的目标不是一次读懂根目录中完整的 prototype，而是每次只增加一个
概念，并得到一个可以独立编译、独立验证的 C++ 程序。

每一课都遵守四条规则：

1. 不依赖 CUDA、PyTorch 或 Tensor framework；
2. 不继承上一课的隐藏库；少量重复是故意的，读者可以只打开当前文件；
3. 运行时先用玩具维度和手写权重，确保结果可以手算；
4. 第 09 课直接运行真实 Qwen3.5-0.8B 权重；它是全课程唯一不使用 toy dimensions 的
   CPU reference implementation。

## 先记住一条完整主线

不管模型有 0.8B 还是 27B 参数，生成一个 token 的主线始终是：

~~~text
文字 -- tokenizer（Python，课程外）--> token ids
     --> embedding --> 重复 N 次 [mixer + FFN]
     --> final RMSNorm --> tied lm_head --> logits
     --> 选一个 token --> 回到下一次 forward
~~~

其中 mixer 是两种“让当前 token 读取前文”的方式之一：attention 读取会增长的 KV
cache，DeltaNet 读取固定大小的 recurrent state。00--08 课把这条线拆开练习；09 课把
所有真实部分接回一条可运行的 Qwen3.5-0.8B forward。

## 像读 buildyourownllm 一样读

`buildyourownllm` 的可读性来自“每个文件只前进一小步，而且运行后能看到结果”。这里也
采用这个约定：每个 `.cpp` 开头都会固定写出四件事：

1. **已经会**：把本课放回上面的整条主线；
2. **本课只加**：这一次真正需要理解的新概念；
3. **运行后看**：运行程序时哪一行输出或断言最值得看；
4. **下一课**：新概念会在完整 forward 的哪个位置继续出现。

这些是**概念上连续的快照**，不是相互 `#include` 的代码版本。刻意重复十几行循环，
比把读者带去 `common.h`、Tensor 类或抽象 backend 更适合教学：打开任意一课就能读完、
编译并手算。

## 课程地图

| 课 | 已经会 | 本课只加 | 运行后看 |
| --- | --- | --- | --- |
| 00 | token id 已经由 tokenizer 给出 | embedding、tied lm_head、argmax | id 2 如何得到四个分数并选回 id 2 |
| 01 | 一个 token 的 hidden 向量 | ordinary RMSNorm、线性层 | 先稳定数值，再混合各维度 |
| 02 | 线性层可以产生新特征 | SwiGLU FFN、residual | gate=0 怎样关闭一个通道 |
| 03 | 之后 attention 会比较 Q/K | RoPE | 同一向量在不同位置怎样旋转 |
| 04 | Q/K/V 都是向量 | causal attention | Q 怎样找 K、从 V 取回内容 |
| 05 | attention 能读取已有 token | GQA、prefill、decode、KV cache | 两个 Q head 共用一组 K/V |
| 06 | 前文可以储存起来 | DeltaNet 固定 recurrent state | state 不随 context 变大 |
| 07 | DeltaNet 的写入/读取公式 | 投影、causal conv、门控、完整 DeltaNet layer | 一个 token 怎样更新一个 DeltaNet layer |
| 08 | 两种 mixer 都有各自 state | 3 DeltaNet : 1 attention 的 hybrid stack | 第二个 token 为何和第一个不同 |
| 09 | 前面所有数学零件 | 固定 Qwen3.5-0.8B 的真实 forward / prefill / decode | token id 如何走完整模型并产生下一个 logit |

课程文件的总行数不会限制在 1000 行，因为每课为了自包含会重复少量代码；限制的是
[09_qwen35_0_8b.cpp](09_qwen35_0_8b.cpp)。否则会为了“少几行”而把关键细节藏进帮助函数，
反而失去教学价值。

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

## 第 09 课：真实 0.8B

真实权重的格式转换、CPU regression 和最小 chat wrapper 也属于第 09 课：

~~~
cd lessons
make
python3 09_pack_weights.py ../models/Qwen3.5-0.8B ../models/qwen35-0.8b.bin
./09_qwen35_0_8b --generate ../models/qwen35-0.8b.bin 248044,198,198 16
make model-test MODEL=../models/Qwen3.5-0.8B
~~~

`09_chat.py` 只在 Python 侧调用官方 tokenizer/chat template；`09_qwen35_0_8b.cpp`
依旧只接收 token ids。CUDA 不在课程中：需要实际 GPU 性能时，才进入同级
[qwen35-0.8b](../qwen35-0.8b/README.md)。

## 与旧 prototype 的关系

旧的完整 reference prototype 已由 git tag `prototype-v0` 保留。它不在当前
工作树中：这个仓库只保留从玩具模型到最终 capstone 的教学路径。
