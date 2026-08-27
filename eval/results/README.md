# 已保留的评测结果

这里只提交能够说明 Engine 与 FP32 Transformers reference 对齐情况的精简对比结果。完整的
prediction、review、HTML report 和运行日志体积较大，仍保存在本机 `eval/results/`，但被
Git 忽略。

| 数据集 | 样本 | Engine | FP32 reference | 官方模型卡 |
|---|---:|---:|---:|---:|
| IFEval | 100 | 0.5200 | 0.5200 | 0.5210 |
| MMLU-Pro | 504 | 0.3294 | 0.3036 | 0.2970 |
| C-Eval | 104 | 0.3846 | 0.3461 | 0.4640 |

对应 JSON 不只保存总分，还保存输入 fingerprint、token/时延统计、配对正确性统计和
McNemar exact p-value：

- `comparisons/ifeval-100-32k.json`
- `comparisons/mmlu-pro-504-seed42-32k.json`
- `comparisons/ceval-104-32k.json`

这些是有限样本结果。IFEval 已非常接近官方分数；MMLU-Pro 和 C-Eval 的 Engine/reference
差异在当前配对样本上没有显示出显著的 Engine 数学错误。详细解释和停止完整长跑的原因见
上一级 [README.md](../README.md) 与 [plan.md](../plan.md)。
