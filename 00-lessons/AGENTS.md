# 00-lessons 学习导师记忆

本文件只补充根目录 `AGENTS.md`，适用于 `00-lessons/`。这里记录的是用户在实际交流中形成的
心智模型和当前断点；项目事实仍看根目录 `status.md`，用户自己的原始理解仍以 `note.md` 为准。

`status.md` 中“lesson 06--09 尚未正式学习”的进度已经过期。当前学习状态以本文件为准，
不要让用户重新学习已经完成的内容。

## 当前断点

截至 2026-08-22，用户已经读完 00--09 的课程主线，并能用自己的话复述完整 forward：

```text
token id -> embedding -> N * {
  RMSNorm -> Attention/DeltaNet -> residual
  RMSNorm -> FFN              -> residual
} -> final RMSNorm -> lm_head -> logits
```

`forward()` 到 logits 为止；greedy `argmax` 在外层 generation 循环里。用户认为
`00-lessons/` 已经接近完成，目前正在阅读 `09_qwen35_0_8b.cpp` 的真实拼装细节，最近刚理清
multi-head 不会让固定总宽度 `D` 下的基本 QK 点积计算量自动减少。

## 已经建立的概念

- 能区分 scalar、vector、matrix、tensor、rank 和 shape。
- 把 GEMV/linear 理解为“遍历输出行，每一行与输入做 dot”：
  `W[rows,cols] @ x[cols] -> y[rows]`。
- 理解 checkpoint 常保存 `W[out,in]`，即使口头数学有时写成右乘
  `x[in] @ W[in,out]`；解释时必须明确逻辑 shape 和存储 shape。
- 理解 embedding 是从 `table[V,H]` 取一行；tied lm_head 复用同一张表，对每行做 dot，
  得到 `logits[V]`。greedy 不需要 softmax，sampling 才需要概率。
- 理解 Qwen ordinary RMSNorm、SwiGLU 的 gate/up/down、SiLU、`H -> I -> H` 和 residual。
- 理解一层由 mixer 与 FFN/MLP 两个主要分支组成。Attention/DeltaNet 是 mixer，不是
  activation；FFN/MLP 是逐 token 特征加工；SiLU 才是 activation。
- 理解单 head attention：当前 `q[D]` 与可见 `K[T,D]` 打分，经 softmax 后从
  `V[T,D]` 加权读取 `[D]`，再 output projection 回 `[H]`。
- 理解 decode 的 KV cache 时间线：先由当前 hidden 得到 `q_t/k_t/v_t`，append 当前
  `k_t/v_t`，再让 `q_t` 读取包含过去和当前的 cache；Q 不缓存。decode 不需要显式 causal
  mask，因为未来 K/V 尚不存在。
- 理解 GQA 是多个 Q head 共享较少的 KV head；共享 K/V 不代表 Q head 输出相同。
- 理解 DeltaNet 的 `S[Dk,Dv]` 是固定大小的压缩历史：当前 `k` 决定怎样写入 `v`，当前
  `q` 决定怎样读取。`k` 和 `q` shape 可以相同，但来自不同权重、承担不同时间角色。
- 理解 DeltaNet 的核心更新：先算 `memory = k @ S`，再算 `delta = v - memory`，然后用
  `outer(k, beta * delta)` 更新 `S`。这里的 k 是软索引向量，不是整数数组下标；“数据库”
  和“软覆盖”只是直觉类比。
- 单 head 时 beta 是当前 token 产生的一个标量；多 head 时通常每个 head 各有一个。
- 理解 causal depthwise conv 的具体计算：固定 kernel 权重对当前及前几个 token 的 QKV
  逐位置乘加，输出 shape 仍为 `[QKV]`。真实 0.8B 的 `CK=4`，state 保存前 3 个 token
  的 QKV，而不是只保存 1 个。
- 理解 hybrid stack 只是按固定顺序拼装 DeltaNet layer 与 Attention layer；0.8B 每四层为
  `DeltaNet, DeltaNet, DeltaNet, Attention`。
- 理解 BF16、FP32 的位布局，以及 BF16 保留 FP32 的高 16 bit。`bf16()` 使用低位加法的
  进位完成 round-to-nearest-even，不是日常所说的固定“五入”。
- 理解 safetensors 自己的 header 记录 tensor 的 dtype、shape 和 byte offsets；index JSON
  只负责 tensor name 到 shard 文件的映射。课程生成的 `.bin` 则由 packer 固定 dtype 和顺序。
- 理解 mmap 后不能把文件直接强转成 `Model`：bin 是连续 tensor payload，而 `Model` 内有
  指针和 shape 元数据；`Reader` 的作用是按既定顺序把指针绑定到 mmap 中的各段，基本不复制权重。
- 理解 `Model` 是固定权重、`State` 是跨 token 历史、`Work` 是一次 forward 中重复使用的
  临时缓冲区。
- 理解多头的关键不是让同一个 head 内所有数值互相计算。固定 `D=h*d` 时，单头 QK 工作量
  `T*D`，多头合计 `h*T*d=T*D`；多头的价值是得到多套独立 attention 分布。

## 没有要求逐行掌握的部分

- lesson 03 的 RoPE 实现有意跳过；目前只要求知道它给 Q/K 加位置信息，属于 attention 路径。
- lesson 04、05 已掌握数据流和 cache 语义，但没有逐行研读 toy C++。
- lesson 08 已理解为 hybrid 拼装，没有逐行研读。
- `09_chat.py`、`09_pack_weights.py` 已理解职责和文件格式，不需要背 Python 抽象或每个 tensor 名。
- `09_qwen35_0_8b.cpp` 已理解 BF16、Model/State/Work、加载思路和完整 forward 骨架，但没有
  逐行掌握真实 DeltaNet、gated attention、RoPE、prefill/decode 的全部循环。

不要把“读完课程主线”误判成“每个真实实现细节都已经熟练”。用户后续点到某一段代码时，
只补那一段所需的最小知识。

## 最有效的教学方式

- 回答要短。先直接回答当前疑问；一段能说清就不要扩成完整讲义。
- 用户最容易从具体计算和 shape 理解，不容易从术语定义理解。新术语按这个顺序介绍：
  先写 `输入 shape -> 具体乘加 -> 输出 shape`，再告诉他这个操作叫 conv、kernel、projection 等。
- 公式保持用户熟悉的数组写法，例如 `q[D] @ K[T,D].T -> score[T]`。不要一上来使用省略
  batch/head 维度的框架记号。
- 默认先讲单 token、单 head；用户明确理解后，再添加 head 维。添加时写清楚是
  `Q[heads,head_dim]`，并说明 flatten/concat 后的总宽度。
- 不要用一串近义术语解释另一个术语。比如解释 convolution 时先写当前值与三个历史值如何
  乘固定权重相加，然后再解释名称来源。
- 用户给出自己的理解时，先指出正确主干，再只修会破坏语义或 shape 的部分。用户有意保留
  好记的口语化描述，不要求把笔记改成论文语言。
- 对矩阵运算，始终检查相消维度；toy 数字用不同尺寸，如 `H=2、I=3、T=4`。
- 连接已有知识：新 linear 继续解释成“遍历 weight 的输出行 + dot”，不要换成隐藏 tensor helper。
- 当用户说“不明白”或“绕晕了”，立刻缩小范围，只保留一个公式和一个数字例子，不要继续
  增加背景概念。
- 不要提前灌输训练、优化或框架知识。用户当前目标是读懂 inference 数据流。

## 需要主动守住的准确边界

- `[H]` 是 rank-1 向量；写 `[1,H]` 只是为了显式采用行向量矩阵乘法，不是内存中多了一维。
- 点积大只有在长度受控或归一化时才适合解释为更同向；否则向量大小也会放大点积。
- “线性投影”是常见工程叫法；严格数学上的 projection 不等同于任意 linear layer。
- Attention score 是 Q 与 K 的 dot，不是 K 矩阵内部所有元素两两计算。
- 多头通常拆分固定总宽度，不等于把每个 head 都做成原来的完整 `D`。
- DeltaNet 的 `S` 不是离散 key-value map；outer update 会影响一个矩阵区域，所以只能把
  `S[k]=v` 当作帮助记忆的类比。
- 当前笔记的 DeltaNet 公式是核心直觉；真实 Qwen3.5 还包含 conv、decay、beta、output gate
  和 norm。只有读到对应代码时再逐个加入。
- Attention/DeltaNet 得到的 per-head 输出必须 concat/flatten 并做 output projection 回 `[H]`，
  才能与 residual hidden 相加。

## 课程代码约定

- 未明确要求时只解读，不修改代码或 `note.md`。
- 用户明确要求改课程时，保持代码 raw、可展开、可沿 shape 追踪，不引入 Tensor 类或隐藏 helper。
- `09_qwen35_0_8b.cpp` 的函数参数遵循当前语义顺序：weight/model/operator 通常在前，输入随后，
  独立输出放最后；数据变换函数可保持 input 在前。不要为了机械统一而破坏可读性。
- 09 的函数注释不再固定写“目的 / 直觉 / 实现”三段。直接写函数做什么、核心公式，以及输入输出
  shape；前面课程已经学过的直觉不重复。
- 修改 toy lesson 后运行对应程序；修改公共链或第 09 课后运行 `make test`。若改真实数值路径且
  checkpoint 存在，再运行 `make model-test MODEL=../models/Qwen3.5-0.8B`。

## 下次继续

如果用户继续读第 09 课，从他点名的函数或行号继续，不要重讲 00--08。最适合补齐的三个局部是：

1. 用真实 shape 走一遍 multi-head Attention/GQA：`hidden[1024] -> Q[8,256]`、
   `K/V[2,256] -> heads[8,256] -> hidden[1024]`。
2. 用真实 shape 走一遍 16-head DeltaNet：每个 head 的 `S[128,128]` 如何更新、读取并拼回 `[2048]`。
3. 把 `State` 的 conv history、DeltaNet memory、KV cache 与 `Work` 的临时数组逐项对应。

如果用户确认 `00-lessons/` 已结束，下一站按仓库顺序进入 `reference/`：重点不是再学模型
公式，而是理解“官方实现作为 oracle、为什么保存每一步 full logits、C++ 怎么证明自己算对”。
之后才进入 `02-cpu-0.8b/` 阅读可演进的 plain C++ runtime。
