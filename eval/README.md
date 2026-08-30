# Qwen3.5-0.8B 评测

这个目录集中保存公开 benchmark 的复现工具、执行记录和评测结果：

```text
run.py       通过 OpenAI API 调用 EvalScope，并保存可回溯 manifest
compare.py   对比 Engine 与 FP32 Transformers 的同题结果
plan.md      三阶段验证计划、执行过程和详细统计
pyproject.toml / uv.lock  与生产 Runtime 隔离的 EvalScope 环境
results/     提交精简配对结果；忽略本机原始 prediction/review/report
```

## 最终结论（2026-08-25）

本轮评测已经达到“验证 Engine 正确性”的目的，主动停止完整数据集和 thinking 长跑：

- 数值正确性：8 个 case 覆盖长度 1/2/3/4/16/64、thinking/non-thinking、fresh、append、
  checkpoint restore 和 rebuild；SIMD Engine 全词表最大绝对误差为
  `0.000232458115 < 0.0005`。
- 采样链路：`temperature`、`top_p`、`top_k`、`presence_penalty`、固定 seed，以及显式
  `min_p=0`、`repetition_penalty=1` 均已实现并测试。
- IFEval 100 题：Engine `0.5200`，FP32 reference `0.5200`，官方 `0.5210`。
- MMLU-Pro 504 题：seed 42 Engine `0.3294`，同题 FP32 reference `0.3036`；seed 43
  Engine `0.2996`，官方 `0.2970`。配对检验没有发现 Engine 与 reference 或两个 Engine
  seed 之间存在显著分数差异。
- C-Eval 104 题：Engine `0.3846`，FP32 reference `0.3461`；样本量不足以精确复现官方
  `0.4640`，但同题配对结果没有显示 Engine 数学错误。

seed 44 在 `304/504` 时主动停止，临时分数 `0.3520`，只作为不完整记录，不进入最终成绩。
完整 MMLU-Pro 12,032 题、C-Eval 1,346 题、IFEval 541 题和 thinking 评测均不再运行；它们
主要增加统计置信度，不再值得消耗当前 CPU 时间。

## 使用

先启动 Engine：

```sh
./build/qwen35 --listen \
  --session-slots 4 --session-context 40960 --log-level info
```

另一个终端运行：

```sh
make -C eval smoke
make -C eval run EVAL_DATASET=mmlu_pro EVAL_LIMIT=100 EVAL_SEED=42
make -C eval run EVAL_DATASET=ceval EVAL_LIMIT=100 EVAL_SEED=42
make -C eval run EVAL_DATASET=ifeval EVAL_LIMIT=100 EVAL_SEED=42
```

默认正式 sampling 严格采用 Qwen 模型卡 Benchmark Results / Language 表格脚注：
`temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5`。结果写入 `results/`；每次
运行的 manifest 会锁定模型、tokenizer、模板、采样参数、dataset 和服务端语义配置。

与 FP32 reference 对比：

```sh
make -C reference serve
make -C eval reference EVAL_DATASET=mmlu_pro EVAL_LIMIT=100
make -C eval compare \
  ENGINE_RUN=eval/results/<engine目录> \
  REFERENCE_RUN=eval/results/<reference目录>
```

更完整的配置、数据 provenance、配对统计和历史诊断见 [plan.md](plan.md)。
