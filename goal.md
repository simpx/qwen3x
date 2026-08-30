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

## 当前任务

新 session 从 Stage 1 开始，只完成评估闭环，不先移植新模型：

1. 阅读 `scripts/agent.py`、当前 HTTP tool-call 测试和 pi 的本地配置方式。
2. 实现最小的 inspect/review/bugfix fixture 和判定规则。
3. 为固定 agent loop 增加机器可读 trace 和统一结果记录。
4. 启动现有 CUDA server，用 0.8B 跑三个场景并保存基线。
5. 连接真实 pi，运行至少 inspect 场景，记录第一个协议或效果阻塞点。
6. 更新本文档的 Stage 1 状态和结果，再讨论 Stage 2 的模型比较方式。

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
