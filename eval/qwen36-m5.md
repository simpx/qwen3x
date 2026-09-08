# Qwen3.6-35B-A3B：M5 Pro MLX C++ 验证记录

状态：这是 2026-09-08 夜间实现的历史验收记录，不是当前主线的验收报告。
原实现保存在 `backup/mlx-before-review-20260908`；各次实验的实际源码与二进制身份
以 `build/qwen36-night/` 中的逐文件 SHA 和产物记录为准，备份分支也包含后续诊断改动。

评审后的核心已提交为 `89dc923`，启动与 Pi 配置为 `ead356c`。评审移除了 MLX 专用
取消回调和可恢复错误状态；forward 异常现在记录后终止，普通 State 操作直接执行。
当前主线没有 chunk 取消或失败后继续服务的能力，后续设计见 [TODO.md](../TODO.md)。
真实模型长数值测试、性能测量及 30 分钟 soak 尚未在该版本重跑。

当前核心已通过 `make test`、MLX binary/library 构建、`mlx-test` 小型 GPU fixture、
forward 异常与 checkpoint 不变量失败测试；CPU/MLX HTTP mock 回归在评审期间通过。
最近简化 try/catch 后重新通过构建、基础测试与 MLX fixture，未重复 HTTP 回归。
这些检查不替代真实模型验收。日志在 `build/review-tool-budget/`：
`simple-catches-test.log`、`fatal-errors-http-cpu.log`、`fatal-errors-http-mlx.log`。

以下数值、性能与服务数据均属于夜间历史版本，保留用于复现与后续比较。
该版本的 64K 默认配置通过 30 分钟、125 项检查的稳定性验收；
128K 独立通过数值、近满容量生成、内存和 HTTP 功能检查，未做 128K 长时间 soak。
128K decode 为社区参考的 86.4%，未达到 90% 调优目标；下文保留差距与失败实验。
验收后的真实 pi 会话仍发生提前 EOS 导致的未完成工具调用，尚未定位根因。指定
数值/服务测试通过与复杂编码任务的生成可靠性分别记录，不能据此宣称后者已达标。

## 环境与输入

- 本机 M5 Pro，15 CPU / 16 GPU cores，48 GiB；macOS 26.3.2，Xcode 26.6。
- MLX 0.32.2，源码 commit `1f8e74e3f12f31365464a6867c6579f0e9b29d85`。
  静态链接，预编译 Metal kernels，`MLX_METAL_JIT=OFF`。
- 参考环境 Python 3.12.14、mlx/ mlx-metal 0.32.2、mlx-lm 0.31.3、
  transformers 5.16.1；冻结文件 `build/qwen36-night/mlx-requirements.lock.txt`。
- 社区 snapshot：`mlx-community/Qwen3.6-35B-A3B-4bit`，revision
  `38740b847e4cb78f352aba30aa41c76e08e6eb46`。
  affine4/group64，按 checkpoint 保留 router/shared gate 等 affine8 例外。
- 单文件文本权重：`build/qwen36-35b-a3b-mlx-affine4-model.bin`，
  1,757 tensors，19,509,034,944 bytes，SHA256
  `45f99b175ae9aef4ea5978f78222094b3123c0d19be0e53db39eb826b2b3f8db`。
  pack 仅复制原始整数、scales/biases 与 dtype，不重新量化。vision/MTP 不打包。

数据、模型和测试产物均在 `build/`。大模型测量串行，C++ 与 Python 参考不同时常驻。
系统在任务开始前已有约 6 GiB swap 使用量；不能把最终非零 swap 自动归因于本程序，
也不能仅凭 allocator 峰值宣称没有系统内存压力。

## 夜间正确性证据（历史版本）

`tests/mlx_reference.py` 用同一权重、tokens、dtype、chunk/checkpoint 边界与
最后位置 LM head 形状比较 full-vocabulary logits。

- 五个短场景共 74 个 teacher-forced token，逐 token logits 全部 bit-identical。
- chunk、reset/rebuild、checkpoint restore、恢复后重新计算 suffix、cache-only
  同样精确相等；五组 greedy 12-token 续写相同。
- snapshot 与打包文件分别通过同一 oracle。
- chunk2048 跨界长度 257/1025/2049/4097/8193，含中间 checkpoint，所有采样位置
  `max_abs=0`、argmax 相同。
- 长位置 61,415 / 126,976 tokens 的 prefill、恢复中间 checkpoint、suffix 重算，
  六组 full logits 均 `max_abs=0`，argmax 和 top-8 完全相同。
- 原生 affine4/8 小算子、DeltaNet 多步输出/FP32 状态 fixture、加载异常边界、
  旧 runtime create/forward/save/restore/reset 失败回归、pack 字节保留与截断检测已通过。
  旧 runtime 失败回归随恢复机制移出主线；它不证明真实 GPU 错误可恢复。

原始证据：`reference/`、`packed-check.log`、`range2048/`、
`range-boundary2048/`、`range-long/range-check.json`（均相对 `build/qwen36-night/`）。

曾定位并修复两类差异：静态 MLX JIT 的 BF16 sigmoid 与 wheel 预编译版本细微不同；
未匹配的 activation 编译边界和 BF16 scalar promotion 会传播到 MoE 路由。
通过最早分歧层与最小 sigmoid probe 定位，修正后再固定 exact-equality 契约。
没有放宽阈值掩盖误差。

参考自身的逐 token 与整段 BF16 reduction 顺序不同，数值可以不同；原始 oracle
记录了这一差异。完整序列 LM head 的 GEMM 与只算最后位置的 GEMV 也可能有舍入差异。
对照必须匹配实际计算范围，不能把它们当作同一数值路径。

## 夜间性能与优化记录（历史版本）

协议：一轮热身、三轮正式测量，下面为中位数，128 个输出 tokens（首 token 算入
prefill，后 127 个计 decode）。纯吞吐使用固定非特殊 token 序列、greedy、忽略 EOS，
与 HTTP 行为测试分开。chunk2048 为当前 C++ 默认值。

缓存策略有一项明确差别：C++ Session 默认在 prompt 末尾保存 checkpoint，Python
`generate_step` 不保留快照。下面的 C++ 数字包含实际服务所需的这部分开销；这不是
完全相同缓存配置的比较。保留社区原始生成速度作为参照，不通过增加参考开销来达标。
如要隔离计算路径，C++ benchmark 可显式传 `--no-checkpoint`，结果必须另列。
两者输入 token IDs、输出长度、权重、dtype、chunk 相同；各自 greedy 生成，历史吞吐
记录未保存完整输出 token IDs，不能宣称续写序列相同。新 benchmark 已补记录。
参考的生成器还会异步预调度下一步。

| 输入 tokens | C++ chunk256 prefill / decode | Python chunk256 prefill / decode |
| --- | --- | --- |
| 512 | 1534.6 / 86.1 | 1392.7 / 91.8 |
| 2048 | 1593.5 / 84.8 | 1470.3 / 90.7 |
| 8192 | 1475.9 / 79.3 | 1336.0 / 84.0 |

单位均为 tok/s。这是初始相同 chunk 基线；最终比较使用下方已完成的 Python chunk2048，
不拿优化后的 C++ 与更小 chunk 的参考直接宣称胜出。

已补相同 chunk2048 的社区短档：2K prefill 2025.6 / decode 88.9 tok/s，8K
1943.3 / 84.1 tok/s。保留 checkpoint 的 C++ 2K 吞吐约为参考的 115% / 95%，
8K 约为 116% / 94%。长档结果如下。

| C++ chunk | 2K prefill | 8K prefill | 8K decode | 8K peak active GiB |
| --- | --- | --- | --- | --- |
| 256 | 1593.5 | 1475.9 | 79.3 | 未记录 allocator |
| 512 | 1845.8 | 1729.4 | 78.7 | 19.06 |
| 1024 | 2143.1 | 1974.7 | 79.1 | 19.37 |
| 2048 | 2324.9 | 2246.2 | 78.8 | 20.09 |

主要收益来自有界批量 prefill：增大 chunk 摊薄调度并提高矩阵运算效率。
8K prefill 相比 chunk256 提升约 52%，decode 基本不变；代价为略高临时内存。
在 chunk2048 下重新通过跨块/checkpoint 的精确数值检查后，才将其设为默认。

| 容量 | 输入 / 输出 | C++ prefill tok/s | C++ decode tok/s | peak active GiB |
| --- | --- | --- | --- | --- |
| 32K | 32640 / 128 | 1699.6 | 61.8 | 20.52 |
| 64K | 65408 / 128 | 1271.8 | 48.3 | 21.15 |
| 128K | 130944 / 128 | 832.3 | 32.3 | 23.31 |

三档均已完成一轮热身和三轮正式测量。64K 最慢 prefill 52.42 秒、最低 decode
48.03 tok/s；128K 最慢 prefill 157.81 秒、最低 decode 30.62 tok/s。

长序列使用完整 KV，无旋转/丢弃、无静默截断。上述内存为 MLX 活跃分配峰值，
不包含 allocator cache 与整个进程的其他占用，不是 phys_footprint。
其中 C++ decode 期间保留 prompt checkpoint，64K/128K 因而比无快照参考多保留约
1.25/2.5 GiB KV；这份已知状态不能误判为图增长泄漏。
首次长档位基线采用单个完整三档实验 2400 秒上限；后续 HTTP 单请求固定 600 秒上限，
依据初测 64K 约 50 秒、128K 热身约 161 秒，保留加载/编译及系统波动余量。
超时需报告失败，不能无限重试。

相同 chunk2048 的社区长档已完成三次正式测量：

| 容量 | 社区 prefill / decode tok/s | C++ / 社区 prefill | C++ / 社区 decode |
| --- | --- | --- | --- |
| 32K | 1578.4 / 67.8 | 107.7% | 91.1% |
| 64K | 1208.1 / 53.6 | 105.3% | 约 90.0% |
| 128K | 817.1 / 37.4 | 101.9% | 86.4% |

64K decode 比值实为 0.8999886，处于测量噪声范围，不把四舍五入当作严格超过目标。
128K decode 未达到 90% 调优目标。保留上述社区无快照生成成绩，不通过降低参考速度达标。

后续诊断与未保留实验：

- 在 64,960-token 前缀上，保留/不保留 checkpoint，逐步喂入相同 token，分别热身一次、
  连续测三段 128-token decode。不保留快照中位数 50.37 tok/s，保留为 51.01 tok/s，
  最终 logits 与预测 token 相同。没有证据支持“快照导致每 token 持续复制并拖慢”这一假设。
  该诊断不是每轮重新 prefill 的标准吞吐协议，不能替换上表。decode-only active 峰值
  为 19.68 / 20.98 GiB，差值符合额外快照状态。证据：`checkpoint-diagnostic/`。
- 重复 CPU argmax 测量约 0.12 ms，不能解释全部长 decode 差距。
- 独立编译每层 MoE 的 decode 路径，在实验 dylib 加载模型时出现 SIGBUS，未进入数值
  校验，也没有性能收益证据。实验仅在 `compiled-moe/`，没有改动或替换交付实现。
  原始命令、退出码 -10 与崩溃摘要保留；根因未确认，不能归因于 MLX 本身的稳定性。
- C++ 同步生成与 Python 异步预调度的区别仍是待研究的调度成本；这是剩余假设，
  不是已经证明的全部差距来源。当前保留数值通过的实现，不以未验证实验替换它。

完整 chunk2048 档位的中位数 / 最低吞吐如下（tok/s；每项三次正式测量）：

| 输入 tokens | C++ prefill | C++ decode | 社区 prefill | 社区 decode |
| --- | --- | --- | --- | --- |
| 128 | 924.04 / 918.07 | 85.75 / 85.64 | 418.59 / 416.48 | 87.41 / 86.68 |
| 512 | 1635.66 / 1618.88 | 83.51 / 82.67 | 1497.82 / 1464.02 | 90.07 / 88.15 |
| 2048 | 2324.94 / 2308.40 | 84.45 / 81.88 | 2025.56 / 2013.92 | 88.88 / 88.61 |
| 8192 | 2246.23 / 2234.41 | 78.78 / 78.72 | 1943.26 / 1897.87 | 84.12 / 83.55 |
| 32640 | 1699.56 / 1696.08 | 61.79 / 60.79 | 1578.42 / 1525.43 | 67.81 / 67.78 |
| 65408 | 1271.85 / 1247.80 | 48.28 / 48.03 | 1208.06 / 1206.49 | 53.64 / 51.69 |
| 130944 | 832.28 / 829.74 | 32.29 / 30.62 | 817.12 / 808.55 | 37.37 / 36.24 |

原始数据：`cpp-short2048/`、`cpp-chunk2048/`、`cpp-long/`、`python-chunk2048/`。
各目录的 run=0 是单列热身，包含首次执行影响，不计入上表。服务进程的加载/ready
另记在各 `server.json`。未在这里把首次编译与加载混入热身后的吞吐。

Session 计算耗时按每轮已测 `prefill_seconds + 127 / decode_tps` 汇总，
包含一次 prefill 和全部 128 个输出，不含加载、HTTP、休眠或测试后的诊断：

| 输入 tokens | C++ 秒，中位 / 最长 | 社区秒，中位 / 最长 | 社区 peak active GiB |
| --- | --- | --- | --- |
| 128 | 1.619 / 1.622 | 1.753 / 1.772 | 18.44 |
| 512 | 1.822 / 1.852 | 1.748 / 1.790 | 18.79 |
| 2048 | 2.385 / 2.438 | 2.440 / 2.442 | 19.98 |
| 8192 | 5.259 / 5.266 | 5.725 / 5.836 | 20.09 |
| 32640 | 21.250 / 21.333 | 22.552 / 23.271 | 20.52 |
| 65408 | 54.058 / 55.018 | 56.508 / 56.581 | 21.15 |
| 130944 | 161.264 / 161.961 | 163.616 / 165.348 | 22.44 |

prefill 临时数组可能比最终 cache 更大，所以全程 active 峰值的差不能直接当作
checkpoint 大小；前述 checkpoint 诊断专门重置了 decode 阶段的峰值计数。

复现吞吐时先停其他模型进程，以下命令顺序运行，使用新的输出目录保留旧证据：

```sh
build/mlx-venv/bin/python tests/mlx_reference.py bench --backend cpp \
  --model build/qwen36-35b-a3b-mlx-affine4-model.bin --chunk 2048 \
  --lengths 128 512 2048 8192 32640 65408 130944 --decode 128 --repeats 3 \
  --output build/qwen36-night/reproduce-cpp
build/mlx-venv/bin/python tests/mlx_reference.py bench --backend python \
  --model build/models/mlx-community-Qwen3.6-35B-A3B-4bit --chunk 2048 \
  --lengths 128 512 2048 8192 32640 65408 130944 --decode 128 --repeats 3 \
  --output build/qwen36-night/reproduce-python
```

## 夜间构建与公共路径回归（历史版本）

- 夜间通过旧 `scripts/build_mlx.sh` 构建固定版本静态库与 shader；评审后已改为
  `make mlx-deps`，该目标也已实际构建安装通过。旧脚本不再属于主线。
- `make -j2 test` 通过，默认 CPU 构建未引入 MLX 依赖。
- `make -C tests http-test PROGRAM=/Users/simpx/qwen3x/build/qwen3x-mlx` 通过；
  此项用 mock 计算覆盖认证、JSON/SSE、错误和公共 Session/HTTP 行为，不能替代真实模型验收。
- `make -C tests render-test PYTHON=/Users/simpx/qwen3x/build/mlx-venv/bin/python`
  通过 9 项官方 tokenizer/template 差分测试。
- 原生 Metal/CUDA 实现未修改；本次没有重新运行 CUDA GPU 测试。

日志分别为 `repro-build-script.log`、`final-cpu-tests.log`、
`final-http-regression.log`、`final-render-regression-venv.log`。

## 夜间服务验收（含已移除的取消实现）

本节描述备份版本的 runner 与服务行为。当前 `tests/mlx_service.py` 已移除取消场景，
不再对应旧 runner 的 39/125 项计数。下列取消延迟和 soak 结论不能套用于当前主线。

夜间版本的 `tests/mlx_service.py` 预先固定 20 个短任务与客观条件，另测工具 fixture、SSE/usage、
thinking、stop/EOS/max_tokens、取消、前缀隔离、上下文超限和长上下文续答。
`tests/mlx_behavior_reference.py` 用同一生产 renderer 的 token IDs 跑固定 mlx-lm。

`tests/mlx_long_fixture.py` 生成变动库存长文，在 1%/50%/99% 放三个可核对记录，
分别测试 64K/128K 检索；64K 实际 61,415 tokens、128K 126,976 tokens，余量用于
生成与工具结果追加。容量吞吐测试另用 C−128 输入，不混淆两种验收。

社区参考已通过 20/20 短任务、工具调用/结果续答，以及 64K/128K 三处长文检索。
实际输入分别为 61,415/126,976 tokens，fresh TTFT 为 50.47/155.94 秒。
对应原始记录为 `behavior-reference/results.json`。

短任务逐项对照保存在 `behavior-comparison.json`：两边均 20/20 通过，19 项文本完全
相同，只有“解释递归”的措辞不同。已用 `behavior-divergence/` 独立定位，而非只凭语义
合理接受差异：社区 generate_step 将 24-token prompt 分成 23+1，本项目一次处理 24。
Python 按这两种调度分别复现了社区和 C++ 的原答案。第 10 个输出 token 首次分歧：
token 101692 / 3709 的 logits，在整段路径为 18.75 / 18.375，在分段路径为 18.5 / 18.75。
这来自 BF16 计算调度不同；在各自匹配的调度下，C++ 的全部 86 组 full logits 都与
Python 精确相同，argmax 全部一致。另测 raw logits 与 logprobs 取 argmax，未导致本例
差异。没有改变精度、输出或放宽数值契约。

64K 功能轮 `service64-cancel-fix/` 已通过全部 39 个场景，短任务为 20/20：

| 场景 | 实测 |
| --- | --- |
| 8K fresh（8,179 tokens） | TTFT 3.887 s；总耗时 4.132 s |
| 64K fresh（61,415 tokens） | TTFT 46.341 s；总耗时 46.862 s；三处记录正确 |
| 相同长请求恢复 checkpoint | 总耗时 0.388 s；cached_tokens=61,415；答案相同 |
| 长历史追加问题 | TTFT 0.162 s；cached_tokens=61,433 |
| 长历史追加工具结果 | TTFT 0.312 s；cached_tokens=61,459；结果与前段记录正确 |
| 取消长 prefill 后新请求 | 1.159 s 内恢复；cached_tokens=0 |

初轮 `service64-trial/` 暴露两个问题：thinking 在 1,024-token 输出预算耗尽时没有可见
正文（finish_reason=length），以及取消长 prefill 后仍占用 slot 47.64 s。前者是预算
截断，改用 4,096 预算后实际生成 2,161 tokens 并正常 EOS；README 说明 reasoning 也占
输出预算。后者通过每个 MLX chunk 前检查连接、取消后令状态失效解决，复测为 1.159 s。
早期失败保留，没有从记录中删除。

回调只在一次 generate 调用期间借用，作用域退出即清除；runtime 在 reset 后重新安装，
防止长 prefill 忽略取消。错误与取消均不能让部分更新状态继续充当有效 checkpoint。
相应 runtime fixture、CPU/MLX 测试、HTTP 回归与修改后短 logits exact 检查全部通过，
日志为 `cancel-build-test.log`、`cancel-http-regression.log`、`post-cancel-short-check.log`。

该历史功能轮已在仅含系统 PATH 与显式 metallib 路径的环境重启，readyz/healthz 通过，
启动 3.56 s。没有 Python/MLX dylib 运行依赖；仍需外置 metallib。
128K 独立 HTTP 功能轮 `service128-trial/` 也已通过全部 39 项，20/20 短任务；
服务启动 3.03 s，测试结束后已停止。本轮结果：

| 场景 | 实测 |
| --- | --- |
| 128K fresh（126,976 tokens） | TTFT 141.763 s；总耗时 142.638 s；三处记录均正确 |
| 恢复同一长请求 checkpoint | 总耗时 0.613 s；cached_tokens=126,976；答案相同 |
| 长历史追加问题 | TTFT 0.261 s；cached_tokens=126,994 |
| 长历史追加工具结果 | TTFT 0.538 s；cached_tokens=127,020；结果与前段记录正确 |
| 取消长 prefill 后新请求 | 1.119 s 内恢复；cached_tokens=0 |
| 物理内存 | `sample` 进程峰值 24.7G；请求后回落到 19.4G |
| MLX allocator | active 峰值 25,660,682,232 bytes（23.90 GiB），最大 cache 约 1.26 GiB |

系统 swap used 起止均为 7,268.44 MiB；vm_stat 的 Swapouts 没有增加，Swapins
增加 488 页（7.625 MiB），Pageouts 增加 3,009 页。这些是全系统计数，不能单独归因
于模型，亦不宣称完全没有分页。功能轮未出现 crash/OOM/GPU error。
当时依据独立长位置数值、近满容量生成、内存与上述功能检查提供 128K 配置；没有执行
128K 的 30 分钟 soak，不用 64K 稳定性结果替代它。

64K 默认配置的 `service64-soak/` 已完成 **1,800.003 秒、125 项检查，零失败**。
服务日志有 128 对完整的 access started/completed（包含取消与忙碌重试），没有 error
级别日志、crash、OOM 或 GPU error。包括 20 个短任务、工具调用/结果续答、SSE、
thinking、停止条件、取消、前缀隔离、溢出，以及后续串行长短混合请求。

- 四次 fresh 61,415-token 长检索均正确，总耗时分别为 46.070、46.055、45.955、
  46.027 秒；初始长历史追加、工具结果追加与 checkpoint 恢复也全部通过。
- 125 项请求检查的耗时中位数 0.271 秒，P95（nearest-rank）1.494 秒，最慢
  46.070 秒。这是特定混合任务的分布，不能当作所有长输入的延迟。
- MLX active 分配峰值 22,643,392,192 bytes（21.09 GiB），最大 allocator cache
  约 1.06 GiB；RSS 8.04–8.10 GiB。RSS 不等于整个 GPU 统一内存工作集。
- `sample` 的物理内存峰值 22.1G，长请求后约 21.7G，结束时回落到 19.3G。
  三次重复长请求后的峰值均为 22.1G，没有观察到持续增长；低于默认服务 36 GiB 目标。
- 系统 swap used 从 7,260.44 降至 7,164.38 MiB；Swapouts 增量为 0，Swapins
  增量 6,193 页（96.77 MiB），Pageouts 增量 2,013 页。13 组周期采样和最终计数
  未显示持续 swap-out/in 抖动。它们是全系统指标，不把后台应用的分页全部归因于模型。
- 最终测试退出码 0；runner 随后发送 SIGINT 停止自己的服务，未留下后台生产服务。

本轮与 128K 功能轮使用同一二进制，SHA256 为
`41f246436541f23bb74c0d172d64f1fbd62148583e3cad0b396bd7bda91d0921`。
夜间源码基于 `53aca12` 加当时未提交实现；逐文件 SHA、模型标识、四个产物大小与实际动态
依赖在 `delivery-artifacts.json`，没有用 Git HEAD 代替未提交实现的身份。

旧实现取消的验收范围是 SSE 连接断开；非流式 HTTP 请求尚未接入 chunk 取消，可能继续
处理到本次请求结束。测试覆盖本机单用户、单 slot 文本服务，不代表并发或长期无人值守
服务的无限期稳定性。CPU 35B PoC、原生 Metal MoE、三文件分发和 128K 长时间 soak
均没有冒充完成。

### 夜间性能目标结论（不代表当前主线重新验收）

| 目标 | 结果 |
| --- | --- |
| 2K prefill/decode 达社区 90% | 约 115% / 95%，达到 |
| 2K decode 至少 20 tok/s | 84.45 tok/s，达到 |
| 8K fresh HTTP TTFT 不超过 60 秒 | 约 3.89 秒，达到 |
| 64K 相对吞吐争取 90% | prefill 105.3%；decode 89.99886%，边界值，不宣称严格达标 |
| 128K 相对吞吐争取 90% | prefill 101.9%；decode 86.4%，decode 未达目标 |
| 64K 正式默认容量与工作集 | 数值/行为/缓存/取消/30 分钟验收通过，物理峰值约 22.1 GiB |
| 128K 进阶容量 | 独立数值/吞吐/内存/HTTP 通过，提供配置；未做 30 分钟 soak |

## 验收后补充：pi 大文件工具调用的输出预算

实际 pi 使用暴露了初始工具验收的覆盖不足：短工具 fixture 没有覆盖一次写入完整
HTML 文件。35B 的 pi 配置起初沿用了 4,096-token 输出上限，在“单文件推箱子游戏”
任务中复现 `incomplete generated tool call`。这与 64K 总上下文容量是两个不同限制。

- `scripts/pi-models.json` 的 35B 单次输出上限改为 16,384；其他模型不变。
  捕获实际 pi 请求确认其发送 `max_completion_tokens:16384`，覆盖服务默认预算。
- 预算耗尽导致的未完成工具调用现在返回 `max_tokens_exceeded`，带实际预算与拆分写入
  提示。非流式返回 HTTP 400，SSE 返回结构化错误事件；没有伪造缺失参数或闭合标签。
  没有触及预算的格式错误仍保留原来的 retry 行为。
- 独立完整 HTML 输出在 4,096 时截断，增大预算后 11,410 tokens 正常结束。
  使用 pi 工具定义及项目目录的 greedy 重放，在 4K 和 16K 均出现未闭合调用，
  因此提高预算不是任意生成都能成功的保证。
- 按 pi 默认 temperature=1.0、固定 seed=123 验证，12,573 tokens 正常 EOS。
  原始输出诊断把工具说明放回 system 文本，并逐字确认 rendered prompt 完全相同；
  仅绕过 HTTP 的工具缓冲以观察原始生成。随后用项目 C++ 工具解析器验证完整
  `write(path, content)`，HTML 参数约 37.7 KB，解析前后内容一致。
  示例 HTML 的 JavaScript 语法检查未通过（生成代码第 674 行），没有将它当作已验证
  的游戏交付，也没有执行模型生成的工具；这里只验证预算和参数传输。
- `make test`、CPU/MLX HTTP 回归、9 项 render 差分测试，以及真实 pi 重试测试均通过。
  新增回归确认预算错误不会让 pi 以相同参数重试，普通格式错误仍能重试。

证据在 `build/pi-tool-limit/`，包括捕获的请求、SSE、原始输出、工具解析与测试日志。
上述新增错误提示版本单独链接为 `fixed-qwen3x-mlx` 做 HTTP mock 回归；未替换用户
正在运行的服务，也未重跑此前的 30 分钟 soak。前面列出的二进制 SHA 对应原验收版本。

## 验收后补充：提前 EOS 与压缩后的失败

审计日志 `build/qwen3x-mlx-audit.log` 记录了两次未耗尽输出预算的失败：

| 请求 ID 后缀 | 输入 tokens | 输出预算 | 实际输出 | 缓存命中 | 停止 token |
| --- | --- | --- | --- | --- | --- |
| `005e` | 46,628 | 15,361 | 1,719 | 45,466 | 248046 |
| `006a`（pi 压缩后） | 17,854 | 16,384 | 2,438 | 0 | 248044 |

两次都开始了 `bash` 的 heredoc 脚本，但没有闭合参数、function 和 tool_call；第二次
甚至停在 `if(`。两个停止 token 都在固定社区模型的 EOS 配置内。解析器拒绝了不完整
调用，未发送给客户端执行。不能用增大输出预算解释或修复这两次失败，也不能将第二次
归因于复用了原长会话缓存。

对 `006a` 的请求，将历史工具参数从 JSON 字符串转成原生模板要求的对象后，C++ 与
固定社区 tokenizer 渲染的完整 prompt 逐字一致。固定 seed=123 重放时，默认参数
（temperature=1、top_p=1、top_k=0）生成 2,009 tokens 后返回完整 bash 调用；社区
generation_config 的参数（temperature=1、top_p=0.95、top_k=20）生成 2,496 tokens
后也返回完整调用。两次重放都命中 17,854-token prompt 缓存，不能作为 fresh/cache
数值对照，也不足以证明采样参数改善了成功率。诊断未执行生成的工具。

原始失败请求未指定 seed，服务也未记录实际随机 seed，不能精确重放当时的采样过程。
这份真实会话尚未完成社区逐位置 logits 对照。证据保存在
`build/pi-compaction-debug/`；提前 EOS 的根因仍开放，不把格式容错、重试成功或短数值
fixture 通过当作修复证明。

## 结构评审后的主线状态

- `arch/mlx/engine.cpp` 展示完整模型计算；主体 `engine.cpp` BF16 教学路径保持原样。
  MLX 类型与 C++20/异常处理位于该后端，主体继续 C++17/no-exceptions/no-rtti。
- `internal.h` 只补充 MLX 后端文件名注释，没有 `Q3X_MLX`、状态错误查询或取消接口。
  `runtime.cpp` 无 MLX 专用改动；`main.cpp` 保留独立提交的工具调用错误分类。
- 模型加载沿现有 factory 接口返回初始化错误；forward 捕获异常、记录原因并终止。
  State 创建、重置和 checkpoint 拷贝直接执行，无逐函数 try/catch。
  错误恢复与请求取消作为独立待办，不假定真实 GPU 失败后可安全继续。
- MLX 的 Layer/Cache/loader 直接表达此后端的完整数据流，没有提取通用 tensor 框架。
  CPU 数学对应关系及不同权重/精度路径见 [后端说明](../arch/mlx/README.md)。
- 原生量化 packer 仅复制 tensor 字节；CPU/Metal 不读取这个容器。
  固定版本依赖通过 `make mlx-deps` 构建，不再使用外部 build 脚本。
- 静态 MLX 仍需要外置 `mlx.metallib`，部署为四个文件；三文件分发暂缓。
  Python 只用于离线工具和参考，不在生产推理进程中。
- 仅 chunk 优化有夜间本机前后收益记录；依赖安装与代码整理不是性能优化。
- CPU 35B PoC、原生 Metal MoE 本夜未做；未修改 CUDA，本机无 CUDA 验证环境。
- 当前主线的真实模型数值、性能与长时间稳定性复验仍待进行；本次仅整理历史证据。
