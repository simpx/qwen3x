# 00

标量（scalar）：rank0，一个数字 3.14
向量（vector）：rank1，有顺序的数字，只有一个轴 [3]
矩阵（matrix）：rank2，行列的数[3, 4]，行为3，列为4
张量（tensor）：rank可以是任意，更一般的统称，多维数组

dot点积：两个向量相乘
dot([1, 2], [3, 4]) = 1*3 + 2*4 = 11
语义上可以理解为“当向量长度相同或已归一化时，结果越大，两个向量越同向”
点积为较大的正数 → 越同向
点积接近 0       → 越接近垂直
点积为较小的负数 → 越反向

矩阵-向量乘法：matrix-vector multiplication
[[1, 2], [3, 4]] @ [3, 4] = [dot([1,2], [3,4]), dot([3,4], [3,4])] = [11, 25]
W [V, H] @ hidden [H] -> logits [V]
cuBlas库里的gemv就是这个
线性层、线性投影（linear projection）都是做这个操作
H维的hidden空间，通过W，投影到了logits空间，所以是线性投影

logits：模型给某个候选token打出的原始分数
所以，logits的形状一定是[vocab_size]，有词表数量个数值
这数值是任意范围
采用greedy decoding，就直接选最大值
采用随机sampling，就先执行softmax，变成概率
probability[i] = exp(logit[i]) / Σ exp(logit[j])，然后再根据概率采样

lm_head：模型最后一层
head好像是模型的一个惯用说法，接在任务输出的“输出头”
模型的hidden size是模型内部定义的
因为最终要输出logits，所以lm_head里要乘的矩阵形状一定是[HIDDEN_SIZE, VOCAB_SIZE]
出于惯例，或者是为了逻辑上可以理解为“为每个token准备一个weight”，所以权重形状实际上是[V, H]
[H] x [V, H].T = logits[V]

tied_lm_head：qwen 3.5-0.8b使用tied lm head，直接用embedding作为lm_head的权重
实际上就是直接复用embedding，而不是真的训练一个W[V, H]
反正embedding的shape是[V, H],而lm_head需要的weight shape也是[V, H]

## 01

linear：线性层
其实就是一个weight，用来做线性变换的，就是做要给mv
W[X, H] @ [H] -> [X]

qwen_rmsnorm：归一化

## 02
这一节是FFN内部结构

FFN：Feed-Forward Network
每一个qwen layer都是
hidden
  │
  ├─ mixer：Attention 或 DeltaNet
  │    负责读取前文、让 token 之间交换信息
  │
  └─ FFN：SwiGLU MLP
       负责独立加工当前 token 的特征

hidden
  ↓ RMSNorm
Attention / DeltaNet
  ↓ residual add
hidden
  ↓ RMSNorm
FFN
  ↓ residual add
下一层 hidden

所以，每一层里，做一些归一、升维的操作，然后就可以做activation，最后再通过一些linear操作，把shape变回hidden
简单的FFN可以是：FFN(x) = W_down @ activation(W_up @ x)

通常ffn最后会有一个residual net
这里MLP和FFN是同一个东西

每一层由 Mixer 和 FFN 两个主要分支组成。Mixer 是 Attention 或 DeltaNet，负责读取前文；FFN 负责每个 token 自己的非线性特征变换。Activation 是 FFN 或 Mixer 内部使用的非线性函数，不等于 Attention。

## 03
skip
给Q/K加位置编码

## 04
attention
先说直觉，attention就是要“拿着最新的一个token/hidden，去历史中找到值得注意的信息”

Query：当前 token 发出的“检索请求” —— 想找什么？
Key：每个历史 token 的“可检索标签” —— 历史中有什么，是否和请求匹配？
Value：每个历史 token 真正携带的内容 —— 历史中真正的信息
Q 和 K 只负责决定分数，最终取回的是 V

当前 token：
h_t @ Wq → Query，表示“我想找什么”

所有历史 token：
h_i @ Wk → Key，表示“我可以怎样被匹配”
h_i @ Wv → Value，表示“关注我后可以取走什么”

Query 与所有 Key 打分
→ softmax 分配注意力权重
→ 对所有 Value 加权求和
→ 得到当前 token 从前文读取的信息

我的最朴素认知
以decode为例，整个序列是[T, H]，先只考虑head数量1
1. 先拿最后一个token的hidden states，[H]
2. [H] @ wq[H, D] -> q[D] ，相当于拿到这个token的一种表示，直觉理解为“待查询的信息”
3. [T, H] @ wk[H, D] -> k[T, D]，相当于拿到所有token的一种表示，直觉理解为用来计算“哪些值得查询”的一种信息
4. [T, H] @ wv[H, D] -> v[T, D]，相当于拿到所有token中准备提取出来的信息的表示
5. q[D] @ k[T,D].T -> [T]，相当于拿到了每一个token的分数，也就是说，一会儿我们准备在V里，每一个位置，对应取信息
6. 做一个softmax，因为我们要的是概率，而不是用[T]加权。因此p[T] = softmax[T]
7. 最终，我们从V中提取数据，p[T] @ v[T, D] -> output[D]
其中，q是拿着[H]算的，所以不需要历史，而k和v都是[T, H]算的，历史cache可以存下来

prefill的时候，需要用一个causal mask，避免信息泄露
这里是decode，没信息可泄露，所以也没有causal mask

## 05

KV cache只保存K和V，不保存Q：旧的Q用完就没用了，但旧K/V还要给未来token读取。

decode位置t的严格时间线：

1. 进入这一步时，历史cache保存位置0到t-1的K/V。
2. 当前hidden h_t分别投影出q_t、k_t、v_t。
3. 把当前k_t、v_t append进cache。
4. 此时可见KV = 历史KV cache + 当前token的k_t/v_t，也就是位置0到t。
5. q_t和全部可见K计算score，softmax后再从全部可见V中加权读取。

公式：

cache_after_t = append(cache_before_t, k_t, v_t)
scores[T] = q_t[D] @ K_cache[T,D].T / sqrt(D)
p[T] = softmax(scores[T])
head_output[D] = p[T] @ V_cache[T,D]

05课的toy中一共有两个输入token。当前query可以理解为第二个token的q_1，cache已经
包含[k_0,k_1]和[v_0,v_1]；得到的是token 1的attention output。完整模型还要继续经过
output projection、后续layer和lm_head，最后才预测token 2。

GQA：本例QH=2、KVH=1，两个Q head共享同一个KV head。共享K/V不代表输出相同，因为
两个Q不同，算出的attention score和读取结果仍然可以不同。

`gqa_decode(query, cache, output)`只负责“当前Q读取已经append好的可见cache”，不是完整
decode step；append发生在调用它之前。因此这里的cache是包含当前token K/V的完整可见前缀。
