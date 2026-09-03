# Qwen3.5 评测

这个目录集中保存公开 benchmark 的复现工具、执行记录和评测结果：

```text
run.py       通过 OpenAI API 调用 EvalScope，并保存可回溯 manifest
compare.py   对比 Engine 与 FP32 Transformers 的同题结果
plan.md      三阶段验证计划、执行过程和详细统计
pyproject.toml / uv.lock  与生产 Runtime 隔离的 EvalScope 环境
results/     提交精简配对结果；忽略本机原始 prediction/review/report
```

## 0.8B 历史结论（2026-08-25）

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

### 大改后的 smoke test

smoke 固定运行 MMLU-Pro、C-Eval、IFEval 各 2 个 case。它只减少 case 数，不改变
EvalScope prompt/few-shot，并采用 Qwen 公开推荐参数：thinking、`temperature=1.0`、
`top_p=0.95`、`top_k=20`、`min_p=0`、`presence_penalty=1.5`、
`repetition_penalty=1.0`、`max_tokens=32768`、seed 42。

每个模型有 30 分钟硬截止；超时即失败，不报告残缺分数。先在一个终端启动评测服务，再在
另一个终端运行对应 smoke：

```sh
make serve-eval-4b
make -C eval smoke-4b

make serve-eval-9b
make -C eval smoke-9b

make -C eval smoke-report
```

两个 `serve-eval-*` 使用 65,536 token Session，保证固定题目的 prompt 加 32,768 token
输出预算不会触及容量上限。模型的原生 context 仍是 262,144；本机 RTX 4080 SUPER 无法
为单 Session 分配完整容量，因为当前 FP32 KV cache 单独就需要 16 GiB。Session 容量不参与
RoPE 或 forward；只要没有触顶，65,536 与更大的预分配容量计算相同。报告同时记录实际
context、最大 prompt 和剩余 headroom。

Qwen 没有公开模型卡成绩使用的完整 harness 和逐 benchmark prompt，因此不能声称严格复现
其内部评测。`smoke-report` 输出本地 4B BF16、9B Q8_0、官方完整集分数和差值；smoke 只有
6 题，且使用 EvalScope 的公开 prompt，差值用于快速发现明显退化，不是对官方完整集的统计
复现。

### 自定义评测

先启动 Engine：

```sh
./build/qwen35 --listen \
  --session-slots 4 --session-context 40960 --log-level info
```

另一个终端运行：

```sh
make -C eval run EVAL_DATASET=mmlu_pro EVAL_LIMIT=100 EVAL_SEED=42
make -C eval run EVAL_DATASET=ceval EVAL_LIMIT=100 EVAL_SEED=42
make -C eval run EVAL_DATASET=ifeval EVAL_LIMIT=100 EVAL_SEED=42
```

默认 sampling 采用 Qwen 模型卡公开推荐的 thinking general 参数：
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
