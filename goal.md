# Goal：让 pi + qwen3x 完成简单 coding agent 任务

## 如何使用本文档

本文档是下一阶段的执行入口。新 session 开始工作时：

1. 先阅读 `AGENTS.md` 和本文档。
2. 检查工作区、当前实现和已有测试，确认“当前基础”仍然成立。
3. 从“当前任务”开始，按阶段和决策门推进。
4. 每轮实验记录配置、原始结果和结论，并更新本文档中的状态。
5. 每次只引入能够改善正确性、任务效果、性能或 pi 兼容性的最小改动。

代码修改遵循 `AGENTS.md`；commit 和 push 由用户明确触发。
当用户把本文档交给新 session 并说“开始”或“继续”时，直接执行“当前任务”；到达决策门、
需要改变模型路线或缺少外部条件时，再汇报证据并与用户确认。

## 最终目标

让本机上的 pi 直接使用 qwen3x 作为模型后端，完成以下日常任务：

1. 查看项目，分段读取源码并解释关键数据流。
2. 阅读 `git diff`，给出有文件和代码依据的 review。
3. 定位一个小 bug，生成最小修改，运行测试并总结结果。

最终仍保持 qwen3x 的产品形态：

```text
pi
  -> OpenAI-compatible HTTP
  -> qwen35 C++ binary
  -> Qwen render / runtime / CUDA engine
  -> model.bin
```

部署路径使用一个 C++ binary 和一个模型文件。Python 只用于开发脚本、reference 和 eval。

## 完成标准

### Agent 效果

- pi 能直接连接 qwen3x，无需常驻协议代理。
- `inspect`、`review`、`bugfix` 三个固定场景均可稳定完成。
- 工具选择、参数和调用顺序有效；回答以实际读取的代码和命令结果为依据。
- bugfix 的修改范围符合任务要求，并通过指定测试。

### 正确性

- 选定模型在 qwen3x 与官方/reference 后端上的 top-k、greedy token 和短文本行为一致。
- CUDA prefill、decode、chunk 边界、checkpoint restore 和 Session cache 有固定回归测试。
- 数值差异能够被归类；任务行为一致后即可停止逐元素 logits 深挖。

### 性能

- 单 Session、4K 实际 prompt 的 TTFT 不超过 10 秒。
- 4K context 后 decode 高于 30 tok/s。
- 记录 16K 和更长输入的数据，用真实 pi 负载决定后续长上下文优化。
- 模型、Session、workspace 和 CUDA runtime 能放入本机 RTX 4080 SUPER 16 GiB。

### 工程形态

- 核心推理继续由 `qwen35` 单 binary 完成。
- 新型号沿用清晰的 `main -> parser/render -> runtime -> engine` 数据流。
- 量化或第三方算子只实现最终路线需要的一种清晰路径。
- README 包含模型准备、qwen35 启动、pi 连接和能力边界。

## 当前基础

仓库目前已经具备：

- Qwen3.5-0.8B CPU correctness engine。
- Qwen3.5-0.8B CUDA engine：BF16 权重、128-token chunk prefill、CUDA Graph decode。
- 0.8B 的 reference vectors、逐步 logits 和 checkpoint 对齐测试。
- OpenAI Chat Completions、普通文本 SSE、非流式 tool calls、usage 和请求生命周期日志。
- `scripts/agent.py` 提供 `read_file`、`write_file`、`bash` 的最小非流式 agent loop。
- `scripts/chat.py` 用于构造和检查 OpenAI 请求。

当前 CUDA 基线记录在 `eval/cuda.md`：

| 场景 | Prefill | Decode |
|---|---:|---:|
| 512 + 128 tokens | 926 tok/s | 175 tok/s |
| 4096 + 32 tokens | 814 tok/s，TTFT 5.03s | 49 tok/s |

0.8B 适合继续承担 correctness、协议和工具链 smoke test；此前真实 agent 实验已经表明它的
任务效果不足以成为最终后端。下一阶段需要先找到能够完成目标的最小模型。

## 核心路径

```text
建立固定 agent eval
  -> 用 0.8B 分离协议问题和模型能力问题
  -> 用参考后端比较 2B / 4B / 9B / 27B
  -> 选择最小可用模型
  -> 在 qwen3x 支持该型号
  -> 达到显存和速度门槛
  -> 按真实请求补齐 pi 兼容
  -> 使用 pi 完成真实 bugfix
```

模型、量化和性能路线由同一套任务结果决定。先证明模型有效，再承担对应实现复杂度。

## Stage 1：建立最小 agent eval

目标是用很低的成本回答两个问题：模型能否完成任务，以及失败发生在哪一层。

### 固定场景

- `inspect`：读取一个超过单次 `read_file` 范围的源码文件，解释指定函数和上下游数据流。
- `review`：对一个包含 2～3 个已知问题的固定 diff 做 review，检查是否找到问题和准确位置。
- `bugfix`：修复一个小型确定性 bug；最终以测试结果和 git diff 自动判定。

fixture 保持小而稳定，避免 qwen3x 自身持续变化使历史结果失去可比性。inspect/review 使用简短
人工 rubric；bugfix 使用确定性测试。

### 两种运行方式

1. **固定 agent loop**：复用或小幅扩展 `scripts/agent.py`，用于跨模型、跨后端公平比较。
2. **真实 pi**：运行相同任务，用于检查 pi 的 prompt、工具 schema、消息轨迹和协议行为。

固定 loop 与 pi 使用相同的基础工具能力。固定 loop 增加机器可读 trace，至少保存：

- endpoint、模型和生成参数。
- 每轮 messages、assistant response、tool call、tool result 和 finish reason。
- prompt/cached/completion tokens、TTFT、总耗时和错误分类。
- 最终 diff、测试结果和任务 verdict。

开发 trace 放在 `eval/` 的 gitignore 结果目录；生产日志继续避开 prompt、代码内容和工具输出。

### Stage 1 决策门

- 0.8B 固定 loop 能完整走完 read/write/bash 多轮协议，即工具链成立。
- pi 能发起请求；每个阻塞点都有保存的请求/响应和最小复现。
- 三个场景可以用同一命令重复运行，并生成可比较的结果。

### Stage 1 状态（2026-08-31）

Stage 1 的评估闭环已经完成；以下是进入 Stage 2 前保存的历史基线：

- `eval/agent/fixtures/` 固定了 inspect/review/bugfix。runner 将 fixture 复制到临时 Git
  仓库，review 预置 3 个已知问题，bugfix 用先失败后通过的测试与改动文件白名单判定，
  inspect 强制至少分两段读完 257 行文件并核对函数位置。
- `scripts/agent_eval.py all` 用同一命令运行三个场景。每个场景保存完整 messages、原始
  assistant response、tool call/result、finish reason、usage、客户端 TTFT、总耗时、diff、
  测试和 verdict。原始结果放在 Git 忽略的 `eval/results/agent/`。
- `scripts/agent.py` 增加了开发 trace、HTTP 请求超时和 `read_file` 单次行数上限。连接失败、
  HTTP 错误、响应错误和 turn limit 会保留分类结果；生产服务日志没有增加 prompt 或工具输出。

0.8B CUDA baseline 使用 `build/qwen35-cuda`、40960 Session context、temperature 0、
non-thinking、最多 16 轮和每轮 512 completion tokens。原始结果在本机
`eval/results/agent/stage1-20260831/fixed-loop/`：

| 场景 | 结果 | 轮数 / 总耗时 | 首要失败 |
|---|---|---:|---|
| inspect | fail | 5 轮 / 12.34s | 第二段 `start_line` 生成为字符串，改用 bash 后虽读完但回答因 length 截断且行号错误 |
| review | fail | 16 轮 / 12.85s | 违反 cwd 提示，重复 `cd / && git ...` 直到 turn limit，未报告 3 个问题 |
| bugfix | fail | 16 轮 / 49.27s | 修改并反复重写测试而非 `retry.py`，测试未通过且达到 turn limit |

单独的协议 smoke 中，0.8B 连续生成并成功执行了 `read_file`、`write_file` 和 `bash`，对应
tool result 也进入后续请求，证明非流式工具协议和 loop 成立；模型随后仍重复失败命令直到
turn limit，归类为模型任务/终止能力不足，而不是 qwen35 工具协议错误。

真实 pi 探针锁定 `pi 0.84.4`，使用隔离的 `PI_CODING_AGENT_DIR` 注册
`qwen3x/qwen3.5-0.8b`，并对 inspect fixture 运行非交互 JSON 模式。pi 发出的工具请求使用
streaming，qwen35 在 parse 阶段返回：

```text
400: invalid chat request: streaming tool calls are not supported yet
```

pi 记录 `stopReason=error`、零 usage、无 tool result，并且不重试。原始 pi JSONL 与 qwen35
request lifecycle log 保存在 `eval/results/agent/stage1-20260831/`；最小服务端复现当时由
`tests/test_http.py` 中 `tools + stream: true` 的 400 断言覆盖。本机系统 Node 是 20.20.2，
而 pi 0.84.4 要求 Node >=22.19；本次用临时 Node 22.19.0 完成探针，正式使用前需升级本机
Node 或锁定官方 `legacy-node20` 版本。

Stage 1 决策结论：固定 eval、0.8B 工具链分层和 pi 首个协议阻塞点都已有可重复证据；0.8B
没有通过任何效果场景，与此前“只用于 correctness/protocol smoke”的定位一致。下一决策是
按 Stage 2 用同一 eval 比较 2B/4B/9B/27B 的参考后端，不应先根据 0.8B 失败移植型号，也不在
Stage 1 提前实现 streaming tool calls。

## Stage 2：选择最小可用模型

在 qwen3x 适配新型号前，先使用官方 Transformers 或成熟 runtime 跑同一套 eval：

1. 依次测 2B、4B、9B、27B。
2. 能放入显存的型号先测 BF16。
3. 放不入显存的型号使用一个成熟、来源明确的预量化版本做效果探针。
4. 每个型号记录 inspect/review/bugfix 结果、工具错误、TTFT、decode、峰值显存和模型来源。
5. 选择稳定完成三个场景的最小型号。

这一阶段允许使用外部 runtime 做决策实验；它们不会进入 qwen35 的生产路径。

### Stage 2 决策门

- **2B 或 4B 通过**：优先实现该型号的 BF16 路径。
- **9B 或 27B 才通过**：量化和成熟 GEMM 进入实现范围。
- **只有某个量化版本通过**：记录其精度、格式和关键配置，qwen3x 只实现这一条候选路径。
- **27B 仍无法完成**：保留失败轨迹，重新评估模型或验收场景，而不是先扩展推理框架。

## Stage 3：在 qwen3x 支持选定模型

只适配 Stage 2 选出的型号：

- 扩展 pack 与 model metadata，明确记录模型 shape 和数据布局。
- 用 CPU baseline 验证短序列 forward 和模型结构。
- 扩展 CUDA backend 的具名 forward、prefill 和 decode。
- 覆盖 chunk 边界、checkpoint、Session cache 和完整 greedy continuation。
- 用官方/reference 后端比较有限步骤的 logits、top-k、greedy token 和文本。
- 用 agent eval 比较 qwen3x 与参考后端的任务行为。

### 正确性时间盒

首次数值诊断以半天到一天为一个时间盒，优先检查：

1. 权重和 shape。
2. 首 token 与逐层关键中间值。
3. recurrent state、attention cache 和位置边界。
4. top-k、argmax 和 greedy continuation。

当误差阈值稳定、argmax/greedy 一致、agent 行为与 reference 相当时进入下一阶段。只有数值差异
实际改变 token 或任务结果时，才继续增加中间向量和逐层诊断。

## Stage 4：达到显存和性能门槛

先建立所选型号的 BF16 或候选量化基线，再按 profiler 证据推进：

1. **容量**：记录权重、模型状态、单 Session、context cache 和 workspace 的显存。
2. **现有结构内优化**：调整 prefill chunk、buffer 复用、CUDA Graph 和明确的 launch 热点。
3. **成熟算子**：主要矩阵乘成为瓶颈时，在 CUDA backend 内使用 cuBLAS/cuBLASLt。
4. **量化**：显存容量或吞吐仍未达标时，实现 Stage 2 已验证的一种 weight-only 格式。

量化包含完整闭环：离线 pack、model bin metadata、load、CUDA kernel、reference/effect eval 和
benchmark。先支持一个型号、一种格式、一条 CUDA 路径，再根据真实收益决定扩展。

每轮优化在 `eval/` 记录：commit/工作区状态、命令、三次稳态数据、正确性结果、峰值显存、
profiler 依据、保留或回退结论。优化在下一步需要明显改变可读数据流、而当前性能已经通过目标时
收敛。

## Stage 5：按真实行为完成 pi 兼容

先记录 pi 实际发送的数据，再实现对应语义和回归测试：

- endpoint、headers、model 字段和错误响应。
- system/user/assistant/tool 多轮消息。
- `tools`、`tool_choice`、多个 tool calls、`tool_call_id` 和 `content: null`。
- `finish_reason`、usage、停止、客户端取消和 context 超限。
- 长文件分段读取、长工具输出和工具执行失败。
- pi 的流式模式。

当前 qwen35 已支持普通文本 SSE 和非流式 tool calls。先确认 pi 是否能使用非流式工具调用；
如果真实 pi 请求要求流式工具调用，则实现 tool-call SSE chunks，并用捕获的 pi 请求建立
HTTP/parser/render 回归测试。

兼容范围以 pi 的实际闭环为准，每个新增行为都保留一个最小 fixture。

## Stage 6：真实使用并收敛

按风险逐步扩大 dogfood：

1. 用 pi 解释 qwen3x 的 `runtime.cpp` 或 CUDA engine。
2. 用 pi review 一个范围受控的真实 diff。
3. 在 fixture 仓库完成 bugfix。
4. 在自己的项目完成一个真实小 bugfix，并人工 review 后运行测试。
5. 连续使用一段时间，按重复失败决定下一轮工作。

完成后：

- 更新 README 中的 pi 使用方式和已知边界。
- 把最终选择的模型准备流程收敛到 `scripts/` 和 Makefile。
- 清理未进入最终路线的临时适配层、模型分支和实验脚本。
- 记录最终模型、量化格式、性能、显存和三个验收场景结果。

## Stage 2～6 完成记录（2026-08-31）

### 模型调查与选择

复核范围不只看参数表，也检查官方发布、社区蒸馏来源和 coding-agent 实测：

- 官方 Qwen3.6 只有 27B 和 35B-A3B，没有适合 16 GiB BF16 的小模型。
- 没有找到官方 Qwen3.7 open-weight 发布。
- 官方 Qwen3.8 open-weight 从 27B 起，coding 能力强，但 BF16 不适合本机容量目标。
- 社区 `Qwen3.8-4B/9B-Distill` 实际是以 Qwen3.5 架构做的 full-parameter
  SFT/off-policy distill。模型卡只给出自报的 MMLU/GSM8K，缺少 coding-agent 工具闭环
  证据，其中 GSM8K 还相对 base 回退。因此不以“3.8”命名替代可复现验证。
- 官方 Qwen3.5-4B BF16 在修正 eval 的两个假失败后，reference agent eval 的 inspect、
  review、bugfix 全部通过，且零 tool error；它是已证明可用的最小候选。

两个假失败分别来自 `read_file` 没有输出行号，以及 review verifier 只接受单数 `line`
和字面量 `HandlerRegistry.dispatch`。修复 evaluator 并增加测试后，最终 reference 原始结果在：
`eval/results/agent/stage2-20260831/qwen3.5-4b-bf16-final-run1/`。

最终选择官方 Qwen3.5-4B BF16，不引入社区蒸馏或量化依赖。checkpoint revision 为
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。

### qwen3x 4B、正确性与性能

- model.bin v2 header 明确保存 16 个 shape/config 字段；同一 binary 按 model ID 选择
  0.8B 或 4B 主流程，并拒绝不匹配的模型。packer 支持官方多 shard checkpoint 流式打包。
- CPU/CUDA 都能编译和加载 4B。reference 覆盖 8 个序列/聊天 case，以及 fresh decode、
  chunk、checkpoint、cache 和四步 greedy；行为契约全部通过。
- 修复了 0.8B 因 `VH == KH` 掩盖的 DeltaNet value-head 映射 bug。
- cuBLAS BF16 Tensor Core batch GEMM 和并行 decode attention 将 4096+32 从
  32.476s / 27.334 tok/s 改善到三次稳态 prefill 2.818～2.847s、decode
  56.307～56.377 tok/s。16K+32 为 33.649s / 35.743 tok/s，32K+32 为
  129.416s / 24.063 tok/s；长上下文退化边界也已明确记录。完整结果见
  `eval/cuda-4b.md`。
- 优化前和优化后的 qwen3x agent eval 都是三个场景全通过、零 tool error。最终结果在
  `eval/results/agent/stage4-20260831/qwen3x-4b-cublas-run1/`。

### pi 协议与真实使用

- qwen35 支持 pi 实际需要的 streaming tool-call chunks、多个 tool calls、tool result
  多轮、usage 和 `finish_reason: tool_calls`；生成期间仍检查客户端断开。
- 真实 pi 0.84.4 使用 Node 22.19.0、40960 context 直接连接 qwen3x，无协议代理。
- inspect 用 bash `nl -ba` 分段读完 257 行并准确引用函数行号；内置 `read` 不带行号，
  是第一次引用错误的原因。
- fixture bugfix 中 pi 先复现 1 个失败测试，只修改 `retry.py`，然后 3 个测试全部通过；
  同一轮还成功执行多个流式 tool calls。
- 让 pi 在当前大 dirty worktree 中修复 4B binary 默认路径时，它误改了 `sibling_file` 并
  陷入自我纠正；该尝试按失败记录并被人工停止。独立小 bugfix 通过不代表 4B 可在任意
  大上下文中无人监督修改代码，真实项目仍需要范围约束和人工 review。

### 完成标准核对

- [x] pi 直接连接 qwen3x，无常驻代理。
- [x] 固定 inspect/review/bugfix 在 reference 和 qwen3x 4B 均通过。
- [x] CUDA/reference 的 top-k、greedy、chunk、checkpoint 和 cache 回归通过。
- [x] 4K TTFT < 10s，decode > 30 tok/s；40960 context 能在 16 GiB GPU 实际运行。
- [x] tool-call SSE 和真实 pi 多轮工具闭环通过。
- [x] README、scripts/Makefile、reference 和性能记录已收敛到 Qwen3.5-4B 路线。

## 当前任务（目标已完成，后续按真实失败迭代）

默认生产路线是 Qwen3.5-4B BF16；0.8B 保留 correctness/protocol baseline。下一轮不再按
Stage 1～6 重跑全流程，而是从真实 pi 的重复失败出发，决定是否需要更强模型、上下文策略
或新的性能优化。

常用基础命令：

```sh
make test
make cuda -j4
./build/qwen35-cuda --listen --host 127.0.0.1 --port 8000 \
  --session-slots 1 --session-context 262144
scripts/agent.py -y "<task>"
```

如果 262144 context 的显存占用阻碍所选模型加载，记录权重、Session 和 cache 的实际占用，
再在 Stage 4 决定 lazy allocation、较小工作 context 或 cache 量化方案。
