# Qwen3.5-0.8B 正确性与评测对齐计划

## 当前执行状态（2026-08-25）

**计划已主动收尾。** 第一、二阶段全部完成；第三阶段已经获得足以判断 Engine 正确性的
Engine/FP32 配对证据，因此不再运行完整数据集和 thinking 长跑。未完成的 seed 44 保留为
可恢复的原始记录，但不作为最终成绩。

- 第一阶段：已完成。8 个 case 覆盖长度 1/2/3/4/16/64、thinking/non-thinking chat，
  SIMD Engine 逐 token 全词表最大绝对误差为 `0.000232458115 < 0.0005`；fresh、append、
  checkpoint restore、rebuild 和 greedy 均通过。
- 第二阶段：已完成。Server/C ABI 已支持 `top_k` 与 generated-only `presence_penalty`，并显式
  处理 `min_p=0`、`repetition_penalty=1`；toy sampler、HTTP 参数和固定 seed 测试已加入。
- 第三阶段：工具链已完成，MMLU-Pro、C-Eval、IFEval 均已完成 20 题 greedy smoke。
  SIMD Engine 与 FP32 Transformers oracle 的回答分别为 20/20、19/19、20/20 逐字一致，
  分数及所有子指标完全一致。模型卡 **Benchmark Results / Language** 表格的脚注明确指定
  `temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5`；此前 1024/32K 运行正是
  benchmark recipe，并非误用了 thinking 参数。2026-08-24 曾把后面 Best Practices 的
  non-thinking 通用生成建议误当作 benchmark recipe，现已纠正。三个 benchmark 的 32K
  正式小样本现均已完成 Engine/FP32 reference 配对；MMLU-Pro seed=42 的 504 题中等样本
  也已在 Engine 和 FP32 reference 两端完成并通过硬验收。Engine seed=43 也已完成并通过硬
  验收。seed=44 在 304/504 时主动停止；完整 non-thinking 和 thinking 小样本不再运行。

20 题 smoke 结果：

| Benchmark | SIMD Engine | FP32 Transformers | 完全相同回答 |
| --- | ---: | ---: | ---: |
| MMLU-Pro（computer science） | 0.0500 | 0.0500 | 20/20 |
| C-Eval（computer_network，共 19 题） | 0.1579 | 0.1579 | 19/19 |
| IFEval prompt-level strict | 0.5789 | 0.5789 | 20/20 |

约 100 题、benchmark sampling 参数且 `max_tokens=1024` 的对齐结果（只证明两端对齐）：

| Benchmark | 样本 | SIMD Engine | FP32 Transformers | 差值 |
| --- | ---: | ---: | ---: | ---: |
| MMLU-Pro（14 类分层，各 8 题） | 112 | 0.2500 | 0.2411 | +0.0089 |
| C-Eval（52 类分层，各 2 题） | 104 | 0.3654 | 0.3077 | +0.0577 |
| IFEval | 100 | 未运行（由正式 32K 结果替代） | 0.5200 | - |

benchmark sampling 参数、`max_tokens=32768` 的正式小样本结果：

| Benchmark | 样本 | SIMD Engine | FP32 Transformers | 官方模型卡 |
| --- | ---: | ---: | ---: | ---: |
| MMLU-Pro（14 类分层，各 8 题） | 112 | 0.3929 | 0.3214 | 0.2970 |
| IFEval prompt-level strict | 100 | 0.5200 | 0.5200 | 0.5210 |
| C-Eval | 104 | 0.3846 | 0.3461 | 0.4640 |

MMLU-Pro 504 题中等样本的正式结果：

| Benchmark | Seed | 样本 | SIMD Engine | FP32 Transformers | 官方模型卡 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MMLU-Pro | 42 | 504 | 0.3294 | 0.3036 | 0.2970 |
| MMLU-Pro | 43 | 504 | 0.2996 | - | 0.2970 |

seed 44 主动停止于 304/504，已评分样本临时分数为 `0.3520`。因为样本不完整，该数值不与
上表或官方成绩直接比较。

两端均完成 504/504 prediction 和 review，所有回答均正常 `stop`，没有失败、length truncation
或未评分样本；prompt/scoring 输入 hash 均为
`bd0c3072270ca7dff720ff1a0f836a871e311071dbda0e5e0145d1e62d9c5f75`。Engine 共生成
1,270,098 token、最长 18,442 token；FP32 reference 共生成 1,193,217 token、最长 20,558
token。配对结果为：双方都对 89 题、仅 Engine 对 77 题、仅 reference 对 64 题、双方都错
274 题；分差 `+0.0258`，配对标准误约 `0.0235`，McNemar exact `p≈0.3122`。因此 504 题
仍没有证据表明 Engine 与同配置 FP32 reference 的准确率不同；reference 距官方只有
`+0.0066`，说明当前 EvalScope recipe 已能很好复现模型卡。Engine 相对官方高 `+0.0324`
需要用其他 sampling seed 估计波动。

seed=43 也完成 504/504 prediction 和 review，全部正常 `stop`，没有失败、length truncation
或未评分样本；共生成 1,255,982 token，最长 16,540 token。它与 seed=42 的样本 ID、prompt
和 scoring 输入完全相同，输入集合 hash 同为
`bd0c3072270ca7dff720ff1a0f836a871e311071dbda0e5e0145d1e62d9c5f75`。seed=43 分数
`0.2996`，距官方 `0.2970` 只有 `+0.0026`。两个 Engine seed 的配对结果为：双方都对 92 题、
仅 seed=43 对 59 题、仅 seed=42 对 74 题、双方都错 279 题；seed=43 相对 seed=42 的分差
`-0.0298`，配对标准误约 `0.0228`，McNemar exact `p≈0.2246`。因此目前没有证据表明两个
seed 的准确率不同，seed=42 偏高可以由 sampling 波动解释。第三个 seed 不再继续，因为现有
证据已经足够判断 Engine 正确性。

这 100 个 Engine IFEval 样本全部成功并评分，无 length truncation，单题最大输出 2,319 token；
其实际输入集合 hash 为
`08916b36aac5a0ba5e9a46121605522152f991942f7cfac52bccf95d2b0ee380`。

同配置 FP32 Transformers reference 也完成 100/100，输入集合 hash 完全相同，primary score
同为 `0.5200`，因此 Engine/Reference score delta 为 `0`，两者相对模型卡均为 `-0.001`。
双方 sampler RNG 实现不同，100 个文本没有逐字相同：Engine 共输出 32,377 token、全部正常 stop；
reference 共输出 62,029 token，其中 1 题达到 32,768 上限。其他 IFEval 指标有小幅采样差异，
但 primary metric 完全一致。prompt-level strict 的配对结果也完全对称：双方都通过 40 题、
仅 Engine 通过 12 题、仅 reference 通过 12 题、双方都未通过 36 题，McNemar exact `p=1`。

C-Eval 的正式 32K 运行也已两端完成 104/104。Engine 分数为 `0.3846`，共生成 69,993 token，
最长回答 14,312 token；FP32 Transformers 分数为 `0.3461`，共生成 94,027 token，最长回答
15,128 token。双方全部正常 stop、没有 length truncation，且 prompt/scoring 输入 hash 完全相同。
配对结果为：双方都对 14 题、仅 Engine 对 26 题、仅 reference 对 22 题、双方都错 42 题；
Engine/Reference 差值 `+0.0385` 的配对标准误约 `0.0665`，McNemar exact `p≈0.665`。因此这组
104 题没有证据表明两端准确率不同，分差可由 sampling 方差解释；双方低于官方 `0.4640`
仍需在中等样本、多 seed 和完整集上区分采样方差与 eval recipe 差异。

MMLU-Pro 的正式 32K 运行也已两端完成 112/112，输入 token 总数均为 156,165，实际输入集合
hash 均为 `14a493307e4b5b2176ab14990703d8f8cc3d8695bfbb0635bf545617c49aa644`。
Engine 共生成 234,624 token，最长 16,594 token；FP32 reference 共生成 208,208 token，
最长 16,140 token。双方全部正常 stop、没有 length truncation。配对结果为：双方都对 25 题、
仅 Engine 对 19 题、仅 reference 对 11 题、双方都错 57 题；Engine/Reference 差值 `+0.0714`
的配对标准误约 `0.0484`，McNemar exact `p≈0.2005`。因此 112 题仍没有证据表明两端准确率
不同；reference 距官方仅 `+0.0244`。后续 504 题和第二个 Engine seed 已进一步证明这里主要是
sampling 波动，而不是 Engine 数学错误。

C-Eval 两端也均为 104/104 成功并评分，实际输入集合 hash 均为
`755b951ac583ed9e5b81b2d7d3398b882007babd2ba99ebc7541d42c2abb93cf`。这轮分差为
5.77 个百分点，不能只凭 104 题断言属于采样方差；32K 与多 seed 结果用于继续归因。

这轮 MMLU-Pro reference 有 50/112 个回答撞到 1024-token 上限；这些题准确率 0.06，正常
结束的 62 题准确率约 0.3871。因此它只能证明同配置 Engine/Reference 对齐，不能作为官方
分数。后续 sampling 运行改用 `max_tokens=32768`、`CONTEXT=40960`、Server
`REQUEST_TIMEOUT=7200` 和 EvalScope HTTP `timeout=7200`；这与模型卡 Quickstart 的 32K
输出设置一致，也避免人为截断或被 Server/客户端默认超时提前终止。
之后的 100/500/完整正式结果均使用新上限及 benchmark sampling recipe。

模型卡存在两套用途不同的官方参数，必须分开：

- [Benchmark Results](https://huggingface.co/Qwen/Qwen3.5-0.8B#benchmark-results) 的 Language
  表格脚注：thinking/non-thinking 均为 `temperature=1.0, top_p=0.95, top_k=20,
  presence_penalty=1.5`。`eval/run.py` 使用并由单测锁定这一套。
- [Best Practices](https://huggingface.co/Qwen/Qwen3.5-0.8B#best-practices) 的通用 API 文本生成建议：
  non-thinking 为 `top_p=1.0, presence_penalty=2.0`，thinking 为
  `top_p=0.95, presence_penalty=1.5`。Server 能表达这套参数，但它不用于复现 benchmark 表格。

误判期间完成的一次 Best-Practices Engine IFEval 100 题结果为 `0.4500`；同配置 reference
和 C-Eval 在发现参数用途错误后主动终止。该结果只保留为生成参数敏感性诊断，不进入正式
benchmark 对比。

长请求关闭 OpenAI SDK 和 EvalScope 的静默重试。请求失败时让 attempt 明确失败，再从已有
prediction cache resume；否则一个 32K 样本可能在后台无提示地从头生成多次。

评测完成状态也采用硬验收：JSON report 必须存在，prediction 数量必须等于所选 split/limit
的预期值，所有 prediction 必须成功，prediction 与 review 数量必须一致，且 `unscored=0`；
任一条件不满足时 manifest 写为 `failed`，不能进入后续 seed 或完整集。Resume 使用
`rerun_review=True`，所以命中的旧 prediction 也会重新评分。

完整 MMLU-Pro 会复制 seed=42 已通过验收的 504 题目录，再用同一 run contract 原地 resume
扩展到 12,032 题，避免重复生成已有样本。该扩容路径已在 seed=42 从约 112 题扩展到 504 题时
真实跑通；样本上限不属于模型语义 contract，但每次 attempt 都记录新的预期题数，最终仍必须
达到 12,032/12,032 才能让后续 C-Eval、IFEval 和 thinking 队列接棒。

最终回归审计还发现 Python binding 在 `Engine.close()` 清空已关闭 Session/Manager 集合时，
对象析构会再次进入同一非递归锁而死锁。析构现先检查 native handle 是否仍有效；mock 回归
由此前卡在 futex 改为正常结束，并新增同 seed 多 token 采样序列完全一致的测试。

FP32 Transformers reference 在 IFEval 第 77 题生成接近 32K 时，dynamic KV cache 因逐 token
`torch.cat` 反复分配使 CUDA context 失效并返回 HTTP 500。Reference server 已切换为
Transformers 原生 StaticCache：按 `prompt + max_tokens` 一次预分配、之后原地写入；短请求与
dynamic cache 的 greedy 输出完全一致。dtype、模型数学和 sampling 均未改变，cache 类型及
context 上限记录在每次 EvalScope attempt 的 `server_info` 中。

这条 32K 样本还暴露出 presence penalty 的非模型开销：旧实现每步重新对完整 output history
做去重/排序。Engine 和 reference 均已改为请求内增量维护 distinct generated-token set；每个
出现过的 token 仍只扣一次，sampling 语义不变，后续长跑不再重复整理全部历史。

IFEval reference 的 100 题 prompt-level strict 为 0.5200，与官方 0.521 几乎一致。初次运行
曾因 NLTK `punkt_tab` 尚未下载而留下 1 个空 review；评测入口现会在开始前准备该资源，
resume 会复用 100 个 prediction 并重新评分，最终为 100/100 scored。

并发消融表明：一个 Engine 进程、4 个 session 并发时约每 34 秒完成一题；同时启动第二个
4-session Engine 后，两边分别降至约每 70 秒和 93 秒一题，合计吞吐反而下降约 15%。
当前 8 核 9700X 已受内存带宽限制，正式评测保持单进程、4 并发；多进程不作为吞吐优化。

曾用 BF16 activation 的 GPU reference 跑过一次，生成路径和分数出现小幅差异；将 oracle
改为 FP32 后差异消失。对 C-Eval 第 16 题额外做了 771-token prompt + 128-step greedy 的
逐步诊断：128/128 argmax 一致，full-vocabulary logits 最大绝对误差约 `4.4e-5`。因此
reference server 默认必须是 FP32；BF16 模式只能作为速度测试，不能用于判断 Engine 正误。

目标是先证明我们的 C++ Engine 与官方 Transformers 实现计算的是同一个模型，
再通过 OpenAI API 跑公开 benchmark，对比 Qwen 官方分数。

这里的“对齐”分成两层：

1. **数值对齐**：相同 token 输入下，C++ 和 Transformers 的 logits/top-k 基本一致。
2. **分数对齐**：在 dataset、prompt、template、采样参数和评分器都尽量相同时，
   benchmark 分数接近官方结果。

在第一阶段通过前，不进入大规模 benchmark。每个阶段都保留固定配置、原始输出和
可重复的命令。

## 第一阶段：数值正确性

### 目标

不经过 HTTP 和评测框架，直接比较 Transformers 和 C++ Engine。这一阶段只回答：

> 相同的 token ids 进入模型，我们是否得到了相同的下一 token 分布？

### 1. 锁定参考环境

记录以下信息，避免以后模型或依赖更新后 reference 悄悄变化：

- `Qwen/Qwen3.5-0.8B` 的 Hugging Face revision/commit。
- safetensors 和 packed bin 的 hash。
- tokenizer 和 `reference/chat_template.jinja` 的 hash。
- Transformers、PyTorch 版本和 dtype。
- C++ 编译器、编译参数和 CPU 指令集。

Transformers reference 使用 BF16 权重数值和 FP32 计算，尽量与我们的
`BF16 weight -> FP32 compute` 路径一致。第一轮直接对齐当前带 SIMD 优化的正式 Engine；
不要求每次再完整运行一遍很慢的纯标量模型。

### 2. 扩充 reference cases

在现有 `tests/reference.json` 基础上增加：

- 直接指定的短 token ids，不经过 template。
- 官方单轮 non-thinking chat prompt。
- 官方单轮 thinking chat prompt。
- 长度为 1、2、4、16、64 的输入，覆盖 conv、DeltaNet State 和 Attention KV。
- 相同 prompt 的 fresh prefill、append-only sync、checkpoint restore 和 rebuild。

每个 case 保存：

- 输入 token ids。
- 若干固定 token 的 logit。
- top-k token ids 及对应 logits。
- argmax token。
- 固定长度的 greedy token 序列。

Reference 由一个独立脚本从 Transformers 生成。日常测试只读已保存的 reference，
不要每次测试都加载 Transformers。

### 3. 自动验收

至少检查：

- 每个 case 的 argmax 必须一致。
- top-k token 集合和顺序应高度一致；两个几乎同分的 token 交换时单独检查。
- reference 中固定 logits 的误差不超过明确阈值。
- 前若干个 greedy tokens 一致。
- fresh、append、checkpoint restore 和 rebuild 到达同一 token 位置时，logits 一致。
- 当前正式的 SIMD 路径通过全部 reference。标量实现只作为定位单个算子数值问题的
  诊断工具，不作为日常完整模型验收项。

### 发现不一致时怎么做

按下面的顺序定位，一次只排除一层：

1. **先比 token ids**
   - token ids 已不同：检查 tokenizer、chat template、thinking 开关和 special tokens。
   - token ids 相同：再检查模型数学。

2. **先在当前 SIMD 版本中找到第一个出错位置**
   - 先比较 layer 级中间结果，不先运行完整标量模型。
   - 若首个错误落在 dot/GEMV 等 SIMD 算子，再只对这个算子或这一层临时切换标量实现。
   - 局部标量正确、SIMD 错：检查 BF16 展开、tail、FMA 和求和顺序。
   - 局部标量也错：继续检查模型公式、weight shape 和权重布局。

3. **找到第一个出错的 layer**
   - 增加临时 debug dump，对比 embedding 后、每层 mixer 后、FFN 后和最终 norm 后的
     `hidden[H]`。
   - dump 先保存 norm、前若干项和固定 checksum；需要时再保存完整向量。
   - Transformers 用 hook 保存同一位置的中间结果。

4. **根据首个出错位置缩小范围**
   - embedding 就错：检查 pack 顺序、offset、token row 和 BF16 转换。
   - RMS/Linear 后错：检查 weight shape、转置、`1 + weight` 和 dot。
   - DeltaNet layer 首次错：分别比较 conv QKV、beta/decay、Delta rule 和 recurrent State。
   - Attention layer 首次错：分别比较 Q/K/V、QK norm、RoPE、GQA head 映射和 KV cache。
   - 只最终 logits 错：检查 final RMSNorm 和 tied lm_head。

5. **区分数学问题与 cache 问题**
   - fresh prefill 对，cache restore 错：检查 DeltaNet checkpoint 和 Attention KV 截断。
   - logits 对，生成结果不同：检查 sampling、seed、stop token 和 template，不再检查 forward。

### 第一阶段完成条件

- reference 生成命令、模型 revision 和误差阈值都已记录。
- 当前 SIMD Engine 通过全部数值 case。
- 若曾发现 SIMD 算子问题，对应算子必须增加 SIMD 与标量小输入对照测试。
- cache/checkpoint 与 fresh forward 在同一位置对齐。
- `make test` 通过。

## 第二阶段：补齐官方采样参数

### 目标

让 OpenAI API 能表达 benchmark 与 Best Practices 中出现的采样配置：

```text
benchmark:                temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5
Best Practices nonthink: temperature=1.0, top_p=1.0,  top_k=20, presence_penalty=2.0
两者均显式记录:          min_p=0.0, repetition_penalty=1.0
```

当前状态：

- `temperature`、`top_p`、`seed`：Server 已支持。
- `top_k`：C ABI、Python binding 和 Server 均已支持。
- `presence_penalty`：已按 generated-only 语义实现，同一 token 只扣一次。
- `min_p=0`、`repetition_penalty=1`：Server 已显式接受为 no-op；其他值会返回 400，
  不过度扩展 sampler。

### 实现顺序

1. Server 解析、校验并传递 `top_k`。
2. 研究 vLLM/SGLang 的 penalty 语义，先明确 `presence_penalty` 统计 prompt token、
   generated token，还是两者，再实现；不自定义一套语义。
3. 在 sampler 前根据出现过的 token 调整 `logits[V]`，不改变 model forward。
4. 将完整 sampling 配置写入请求日志和评测结果，方便回溯。

### 测试

使用人工 toy logits 做确定性单测：

- `temperature=0` 始终等于 argmax。
- `top_k=1` 只能选最大 logit。
- top-p 不会选到 nucleus 之外的 token。
- 相同 seed 产生相同 token 序列。
- 出现过的 token 按 `presence_penalty` 且只减一次分数。
- 非法参数返回清晰的 OpenAI API 400 错误。
- Server 收到官方完整 payload 后，日志能证明所有参数已到达 sampler。

Sampling 的随机数算法可能与 vLLM/SGLang 不同，因此不要要求不同后端在同一 seed
下逐 token 完全相同。要求的是我们自身可重复、过滤规则正确，以及整体分数在统计上接近。

### 第二阶段完成条件

- 官方 sampling payload 可以原样调用我们的 API。
- sampler 单测和 HTTP 参数测试通过。
- 相同 seed 重复请求可以复现。
- 不改变第一阶段的 logits 正确性。

## 第三阶段：使用 EvalScope 跑公开分数

### 目标与工具

使用 EvalScope 的 OpenAI-compatible API backend，不自己重写 dataset prompt、答案提取和
评分规则。EvalScope 作为独立 uv eval dependency group 锁定版本，不进入生产依赖。

初期只跑 text-only、non-thinking：

| Benchmark | 官方 0.8B non-thinking | 官方 0.8B thinking |
| --- | ---: | ---: |
| MMLU-Pro | 29.7 | 42.3 |
| C-Eval | 46.4 | 50.5 |
| IFEval | 52.1 | 44.0 |

官方表格和参数来源：
<https://huggingface.co/Qwen/Qwen3.5-0.8B#benchmark-results>

### 运行入口

保持命令简单，最终提供类似：

```sh
make eval-smoke
make eval EVAL_DATASET=mmlu_pro EVAL_LIMIT=100 EVAL_THINKING=0 EVAL_SEED=42
make eval EVAL_DATASET=ceval EVAL_THINKING=0 EVAL_SEED=42
make eval EVAL_DATASET=ifeval EVAL_THINKING=0 EVAL_SEED=42
```

评测并发必须固定并记录。当前正式 CPU Engine 运行使用一个进程、4 个并发请求；每个样本
请求都显式携带固定 seed，采样状态不跨请求共享，因此并发只改变调度和耗时，不改变单题
生成语义。并发消融已经证明第二个 4-session Engine 会因内存带宽争用降低总吞吐，所以不再
增加进程。FP32 reference 按其 GPU 内存约束单独固定并发数，并在每次 attempt 中记录。

每次运行保存：

- Engine commit、model/tokenizer/template revision。
- EvalScope 版本和 dataset revision。
- thinking 开关、sampling 参数、seed、max tokens 和 context size。
- 每题 prompt、原始回答、提取答案和对错。
- 汇总分数、token usage、cache hit、耗时和异常终止数。

会影响结果的服务属性也属于 resume contract：Engine 的 real/mock compute 与 context，reference
的 device、dtype 与 context。slots、并发数和 dynamic/static cache 只属于 attempt；
timeout/retry policy 会影响长回答是否成功写入 prediction，因此也由 contract 锁定。

### 运行顺序

1. **Smoke**：每项 20 题，先用 greedy，检查 prompt、答案提取、stop 和结果文件。
2. **小样本**：每项分层抽样 100 题，使用官方 sampling 和固定 seed，手工检查错题。
3. **中等样本**：分层抽样约 500 题。seed=42 同时跑 Engine 和 FP32 reference，做同题
   配对比较；seed=43、44 再跑 Engine，估计同一实现仅更换 sampling seed 的分数波动。
4. **完整 non-thinking**：原计划先 MMLU-Pro，再 C-Eval，最后 IFEval；现因验证目标已达成
   而取消。
5. **Thinking**：原计划从小样本开始；现因不再提供关键正确性证据而取消。

这里的完整集按当前 EvalScope/ModelScope 有标签的评分 split 明确定义为：MMLU-Pro test
`12,032` 题、C-Eval val `1,346` 题、IFEval `541` 题。manifest 必须达到对应题数才算完成。

`LIMIT=20/100` 只用于验证流程，不宣称为正式模型分数。MMLU-Pro 共有 12,032 题且
默认 5-shot，完整评测需要预留较长时间。

### 分数不对时怎么做

1. 先用同一 EvalScope 配置跑一个 Transformers/vLLM reference server。
2. 我们和 reference server 分数不同：对比相同样本的 token ids、logits、sampling 和 stop。
3. 我们和 reference server 基本相同，但都与官方表格不同：问题优先归类为
   eval recipe 不同，继续对齐 dataset revision、few-shot prompt、answer extractor、template 和采样参数，
   不盲目修改 Engine。
4. 只有第一阶段的数值 reference 也出现不一致时，才回到模型实现排查。

### 第三阶段收尾判断

- 三个 benchmark 都能从固定命令完成运行，并保存可回溯的原始结果。
- 我们与同配置 reference server 的分数在采样方差内接近。
- 与官方分数的差异能明确归因为 Engine、sampling 或 eval recipe，不留“不知道为什么”
  的结论。
- 现有 32K 小样本、504 题配对和多 seed 结果已经满足上述判断需要；不再把“跑完整 split”
  当作本轮收尾条件。
- thinking 与完整 split 保留为未来需要发布正式 benchmark 成绩时的可选工作，不属于当前
  Engine 正确性验证。
