# Qwen3.5-9B Q8_0 macOS CPU baseline 验收

## 机器与产物

- 机器：MacBook Pro，Apple M5 Pro，15 CPU cores，48 GB unified memory。
- 系统：macOS 26.3.2，Apple Clang 21.0。
- 官方权重：`Qwen/Qwen3.5-9B` revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。
- qwen3x model bin：`build/qwen35-9b-q8_0-model.bin`，9,514,418,816 bytes
  （8.861 GiB），SHA-256
  `bdcc3beb2e94d161142cb96c1b0ec1bc3b87ca3ec3b944f6f3fc3983ae9893b7`。

本次验收显式使用原生 CPU binary `build/qwen35`。CPU Q8_0 dot 使用 Arm NEON，大矩阵
按行通过系统线程池并行。后续加入 Metal backend 后，Darwin 上的 `make serve-9b` 和
`make serve-eval-9b` 已改为自动使用 `build/qwen35-metal`；本文只记录 Metal 之前的 CPU
容量、正确性和性能基线。

## 容量与性能

65,536-token Session 的 `vmmap -summary`：

```text
virtual total          13.9 GiB
resident total          8.9 GiB
mapped model file       8.9 GiB virtual / 7.9 GiB resident
MALLOC_LARGE            2.1 GiB virtual / 0.9 GiB resident
MALLOC_LARGE reserved   2.0 GiB virtual / 0 resident
physical footprint      0.9 GiB
swap                     0
```

模型文件页是只读映射且可回收；KV cache 和 recurrent state 按实际 token 触页。48 GB
机器有充足余量，24 GB 是保守的最低建议。warm Session benchmark：

```text
./build/qwen35 --model build/qwen35-9b-q8_0-model.bin \
  --bench 64 16 --session-context 65536
prefill 64: 5.865 s, 10.912 tok/s
decode 16:  1.604 s,  9.973 tok/s
```

这个短上下文 decode 数字不能外推长 thinking。真实 smoke 中 3K prompt 的 CPU prefill
需要数分钟，输出增长后 attention 也会降低 decode 吞吐；本机适合 correctness、研究和
低并发 PoC，不适合把 32K thinking 当成低延迟服务。

真实 OpenAI-compatible HTTP 请求使用 temperature 0、thinking off，要求严格输出
`hello`，返回内容完全匹配；17-token prompt 的端到端延迟为 1.72 秒。

## Eval smoke

配置保持项目 smoke contract：EvalScope 1.10.0、thinking、seed 42、Qwen 公开 sampling
参数、每个数据集 2 题、`max_tokens=32768`、65,536-token Session。macOS 使用 Hugging Face
数据源，因为本机到 ModelScope 的 dataset HEAD 请求超时；manifest 记录 hub、dataset ID、
revision、tokenizer、model bin、chat template 和实际输入 hash。

| 数据集 | 样本 | 得分 | 输入 token | 输出 token | 最大输出 | 完成状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MMLU-Pro | 2 | 0.5 | 3,461 | 7,618 | 6,053 | 2 stop |
| C-Eval | 2 | 1.0 | 4,182 | 18,291 | 9,161 | 2 stop |
| IFEval prompt strict | 2 | 1.0 | 135 | 11,903 | 7,085 | 2 stop |

IFEval 的 instruction strict/loose 和 prompt strict/loose 四项均为 1.0。总计 6/6 prediction
成功、6/6 review 可评分、0 failed、0 length limited。这里只抽了六题，分数不能与官方完整集
作统计比较；smoke 的意义是验证真实 model、render、HTTP、长生成和 evaluator 数据流。

结果目录：

```text
eval/results/smoke/9b/20260903-162714-mmlu_pro-thinking-smoke-seed42
eval/results/smoke/9b/20260903-175825-ceval-thinking-smoke-seed42
eval/results/smoke/9b/20260903-192714-ifeval-thinking-smoke-seed42
eval/results/smoke/latest.json
```

MMLU-Pro 第一次运行暴露服务默认 600 秒 request timeout；完成的首题通过 EvalScope cache
保留，服务改为 7,200 秒后 resume 只重跑失败题。macOS 整套 smoke 的硬截止改为 12 小时，
以覆盖六题各自可能使用完整 32K 输出预算；Linux/CUDA 仍保持 30 分钟默认值。

## Pi

用户级安装使用 Node 22.19.0 和 `@earendil-works/pi-coding-agent` 0.84.4。
`scripts/pi.sh --list-models qwen3x` 能读取本地 `/v1/models` 并列出 4B、9B；9B
服务运行时自动选择 `qwen3.5-9b`。

无文件写入的真实 agent smoke 限定只启用 bash 工具：模型生成一次
`bash({"command":"pwd"})`，pi 执行成功并把 `/Users/simpx/qwen3x` 回送模型，第二轮最终只
输出该路径。第一轮 802-token prefill 为 86.277 秒，生成 55 tokens 并以 `tool_calls`
结束；第二轮命中 857-token live cache，只 prefill 24 tokens，生成 47 tokens 并正常
`stop`。这覆盖了 pi provider/model 选择、thinking、流式 tool call、工具结果回传和 Session
cache。

## 回归

```text
make test                         pass
make -C eval test                 10/10 pass
make -C tests render-test         9/9 pass
make -C tests http-test           pass
non-NEON fallback compile         pass
git diff --check                  pass
```

`make test` 包含新增的 1,024-row Q8 matrix 回归，实际进入 macOS row-parallel 路径。
