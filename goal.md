# Goal：今晚跑通并优化 Qwen3.6-35B-A3B 的 MLX C++ 后端

> 状态：用户已于 2026-09-08 通过 active goal「完成goal.md」授权执行。正在实现和验证，
> 实时进度与原始证据在 build/qwen36-night/；下列验收未全部完成前不宣称交付。

## 今晚的唯一主线

在本机 **M5 Pro 48 GiB（15 核 CPU、16 核 GPU）** 上，让 Qwen3.6-35B-A3B
通过本项目的 **MLX C++ 后端**完成真实推理，接入现有 CLI/HTTP，把主要时间用于
prefill、decode 与内存优化。明早交付可直接启动的本地服务、可信的性能数据和可评审的代码。

执行顺序：

```text
社区 MLX 原生量化权重与本机参考
  → C++ MLX 完整 forward 与数值验证
  → 现有 Session / CLI / HTTP 跑通
  → 找出瓶颈并调优
  → 服务验收与明早交接
```

**Mac CPU 35B PoC、原生 Metal MoE、跨后端统一权重格式，全部延后。**
它们不是 MLX 的前置条件或今晚的完成条件。不要为它们继续下载官方 70GB BF16
checkpoint、打包 Q4_0 或开发新 kernel；已有 CPU/Metal 功能保持可用。

本夜投入优先级：**正确性底线 → 跑通 C++ 服务 → 性能调优 → 结构整理**。
用户明确要求今晚性能优先，项目结构明天再评审。允许直接的局部实现、必要的重复代码、
专用优化和少量构建适配；不要统一所有 backend、设计通用框架或做与性能无关的重构。
不能用错误输出、丢弃上下文、跳过状态更新或偷偷换模型来换取速度。

本任务明确覆盖 Qwen3.6-35B-A3B，并允许 MLX 后端局部使用 C++20/异常。这些是用户
为本次任务确定的范围与选择；与 AGENTS.md 的长期范围或默认风格不一致处以此为准。
其他约定继续适用，尤其是单个 C++ 进程、复用产品数据流、不自动 commit/push。

## 已确定的技术选择

### C++ 后端与异常边界

- 模型实现在 `arch/mlx/engine.cpp`，用 MLX C++ API 展示完整 Qwen forward。
- 复用 Engine/Session、render、tokenizer、sampling、CLI、HTTP 和 SSE。
  不启动 Python worker，不调用 mlx-lm.server 作为本项目后端。
- 主体保持 C++17、`-fno-exceptions -fno-rtti`；仅 MLX 库及适配翻译单元使用
  C++20 和必要的异常选项。MLX 头文件及数组类型不进入主体公共 header。
- MLX 对外入口捕获异常并转换为现有错误返回，包括加载、状态操作、惰性求值和同步。
  异常不能穿过 C ABI/无异常代码；失败后不能继续使用部分更新的 Session。
- 接口不能表达失败时，做最小显式补充并测试；不能 catch 后返回成功。
- 优先复用已构建的静态 MLX，提供可复现的 `make mlx`、测试与 benchmark 入口。
  默认 CPU 构建不要求安装 MLX，也不被升级为 C++20。

### MLX 原生量化

固定输入：

```text
repo       mlx-community/Qwen3.6-35B-A3B-4bit
revision   38740b847e4cb78f352aba30aa41c76e08e6eb46
目录       build/models/mlx-community-Qwen3.6-35B-A3B-4bit
仓库大小   20,429,169,263 bytes（完整 snapshot，含非文本部分）
主量化     affine，4-bit，group size 64
例外       router / shared expert gate 等按 checkpoint 配置保留 8-bit
```

逐 tensor 核对量化参数、权重、scales、biases 和未量化张量 dtype。
**放弃此前 Q4_0 重排为 MLX group-32 的路线，不从 Q4_0 解码后重新量化。**
使用 checkpoint 自带量化数据，让 MLX 量化算子直接消费。

目标是离线打包为 `build/qwen36-35b-a3b-mlx-affine4-model.bin`：
只改变容器/必要布局，保留量化整数与 scales/biases 的值和 dtype，不重新量化。
准确保留混合 4/8-bit 配置；文件名 affine4 不代表所有 tensor 都是 4-bit。
格式有明确标识、版本、shape、dtype、量化参数、边界与 EOF 检查，不与旧 Q4_0 混淆。
使用本模型需要的最小格式，不设计通用模型容器。

为尽快测通，初版可以直接读固定 snapshot。若单文件打包影响核心调优进度，明早明确
记录目录依赖和延期项，不因此停住 C++ 推理，也不宣称已满足三文件分发。
官方 checkpoint 保持原样，开发数据和生成权重放 build/。

Python mlx-lm 仅用于离线准备和参考。阅读固定版本 qwen3_5_moe 及其依赖，核对
sanitize、gate/up 拆分和 vision tensor 过滤，不引入视觉推理。
模板沿用本项目行为；数值对照显式使用相同 token IDs，避免模板差异干扰。

## 产品范围与交付底线

- 文本聊天、中文、代码、工具调用；本机单用户、一个 Session slot。
- **64K（65,536 tokens）是默认交付容量，必须验收；128K（131,072 tokens）是本夜
  必须尝试并报告的进阶目标，通过后提供配置。**8K 仅作开发与快速回归，32K 作中间档位。
- 容量指模板、历史、工具结果、当前输入与生成输出的总 token 数；预留输出空间，
  不能用填满容量的 prompt 再额外生成来测试，也不能只调大参数就宣称支持。
- OpenAI Chat Completions、SSE、cache/checkpoint、取消和错误返回可用。
- 不做 vision、MTP、投机解码、训练、并发吞吐工程或新增模型家族。
- 不为今晚新增 CPU/Metal 对 MLX 权重的支持，不改 CUDA。

真实权重通过 C++ 后端推理并完成正确性与服务验收，才算“跑通”。
只有本机可复现的前后对比才算“性能优化”。安装成功、合成 probe、社区 Python 服务
或漂亮的单次 tok/s 不能替代交付。性能未达目标时保留真实结果并说明差距。

### 长上下文可行性与实现约束

模型原生支持 262,144 tokens，64K/128K 无需 RoPE 扩展。40 层中 10 层为完整
Attention，每层 2 个 KV heads、head dimension 256；其余 30 层保存 DeltaNet 固定状态。
按单 Session、BF16/FP16 KV 计算，每 token KV 为
`10 × 2（K/V）× 2 heads × 256 × 2 bytes = 20 KiB`：

| 总上下文容量 | 单份 KV cache |
| --- | --- |
| 8K | 160 MiB |
| 32K | 640 MiB |
| 64K | 1.25 GiB |
| 128K | 2.5 GiB |

DeltaNet 主要 FP32 递归状态另约 60 MiB。上述为 shape 推算，未包含权重、conv state、
checkpoint 副本、分配器缓存与临时张量；不是进程峰值内存实测。社区 snapshot 约
19 GiB，文件大小也不等于运行内存。48 GiB 容量支持认真争取 128K，但仍需实际峰值验收。
初版保留 BF16/FP16 KV，独立于权重 4-bit；不把 KV 量化作为长上下文前置任务。

- 从第一版采用有界 chunk prefill，分块求值并释放旧图；不一次构建整个 128K 计算图。
- 使用 MLX 高效 Attention 路径，避免显式物化完整 T×T scores/mask；分块临时内存也需测量。
- 服务 prefill 只对需要的位置计算 logits。128K 所有位置的 BF16 logits 约 60.6 GiB，
  不得为了测试或输出接口整段保留；长序列参考按选定位置流式采样 full-vocabulary logits。
- 记录 checkpoint 保存/恢复与 KV 扩容的复制峰值；128K 完整 KV 副本会再增加 2.5 GiB。
  优化共享/复制策略时保持状态隔离，不能以污染 checkpoint 换内存。
- 长上下文的关键风险是 prefill 时间与临时内存，不能仅凭最终 KV 大小宣称可用。

## 阶段一：补齐 MLX 输入与本机参照（约 1 小时）

1. 读工作区 diff 和 `build/qwen36-night/progress.md`，复核已有依赖、文件与进程。
   不重建已通过的工具链，不覆盖用户新增改动。
2. 只恢复社区 MLX snapshot，按固定 revision、大小和 SHA 逐文件校验。
   现有 `build/qwen36-night/download_models.py` 原本下载两个模型且官方在前；
   **先改为只选社区 repo，或使用独立下载入口，不能原样启动旧脚本。**
3. 用固定 mlx-lm 跑真实短生成，确认模型支持、tokenizer、量化加载和 GPU 正常。
   记录 dtype、量化参数、权重 hash 与内存。
4. 固定 raw token 输入，导出短序列逐步 full logits 和必要中间状态，建立 C++ 参考。
   测社区实现的本机速度；关闭投机解码等本任务未实现的额外能力。
5. 社区与 C++ 大模型串行运行，退出后释放权重，不同时常驻或竞争 GPU。

参考至少覆盖 512、2048、8192 输入 tokens、128 decode tokens。
先完成短序列 oracle 与 2K 基线，让 C++ 主线继续推进；随后补 32K、64K 与 128K
容量档位的真实长输入参考，预留生成空间。128K 参考失败同样记录原因，不作为跳过
C++ 尝试的理由。长序列仅保存选定位置的 logits，避免 oracle 本身耗尽内存。
下载有限重试，不等待交互登录或全局安装，不把部分 shard 标为 ready。

## 阶段二：完整推理与服务接入（约 2 小时）

直接实现真实 35B 所需数据流；单算子使用小 fixture，不强制先开发另一款 MLX 模型。
复用 ModelConfig 的 shape，逐项核对 checkpoint 和参考实现：

```text
embedding → layer loop
  → norm → DeltaNet / Attention → residual
  → norm → router / selected experts / shared expert → residual
→ final norm → logits
```

- 使用 quantized_matmul、gather_qmm 等 MLX 算子；只计算选中 experts。
  路由与索引留在 GPU，不在每层用 .item() 读回。
- 精度先匹配固定 mlx-lm 的实际计算路径，包括敏感的 norm、softmax、路由与
  DeltaNet 状态累积。无需额外开发全模型 FP32 模式作为前置条件。
- Attention 因果 mask、KV offset、DeltaNet recurrent/conv state 正确。
  checkpoint 不被后续数组更新污染，reset 与不同前缀复用语义明确。
- 正确处理 MLX 惰性图与生命周期，避免每个 token 留住全部历史 graph。
- 最早可用版本就接 CLI、Session benchmark 和 HTTP；不等最后才验证接口与模板。
- logits 回读、sampling 和停止条件沿用项目数据流，有瓶颈证据再做最小调整。

正确性验收：

1. 验证 packing、scale/bias、group、expert 索引与命名映射，覆盖混合 4/8-bit。
2. 同权重、tokens、状态和 dtype，对照固定 mlx-lm 的短 prefill、多步 teacher-forced
   decode full logits。记录 max/mean absolute error、相对误差、top-k 和 argmax；
   不一致时定位首个发生差异的层/算子。
3. 首次调优前固定数值阈值与向量。阈值依据 dtype 和等价计算路径的重复测量写入测试，
   不沿用不同 Q4_0 权重的 5e-4 契约，也不能看完失败再随意放宽。
4. 比较逐 token、chunk prefill、checkpoint restore、reset/rebuild、cache hit。
   相同计算顺序要求一致；改变 reduction 顺序使用预先固定的误差界。
5. 固定 greedy 续写和短行为集；argmax 分歧必须解释，不能只凭“看起来合理”验收。

## 阶段三：性能主攻（约 3.5 小时）

保存正确版本的数据和可恢复补丁，再调优。**不要达到 20 tok/s 就停止；今晚要找出并
消除主要瓶颈，尽量接近同机社区 MLX 实现。**不把单次测量噪声当收益。

按 profiler/分段计时决定顺序，优先检查：

1. **调度与同步**：逐层 eval、CPU/GPU 往返、重复编译、临时分配、图构建开销。
   调整 eval/async_eval 或编译边界；计时覆盖对应计算完成。
2. **prefill**：真正批量/chunk 的矩阵路径，正确处理 DeltaNet 时间依赖和 Attention
   cache。不能把逐 token 循环改名为 batch。比较 chunk size 和峰值内存，不能跨
   runtime 指定的 checkpoint 边界。
3. **MoE**：selected-expert gather、量化 gate/up/down、布局与路由调度。
   先用 MLX 已有高效实现，再考虑融合 gate/up 和减少中间张量。
4. **decode**：状态/工作区复用、图大小、同步点、logits 回读和 sampling。
   对固定 shape 重复计算尝试 MLX 编译，确认状态捕获不会复用旧 cache。
5. **内存**：加载峰值、权重副本、KV/recurrent state、allocator cache、长序列图保留。
   不整模型解量化，不缓存全部 experts 的浮点副本。

有证据时允许在 arch/mlx/ 增加局部自定义 Metal kernel、编译函数和专用快速路径。
不为代码形式整齐放弃明确收益，也不在没有瓶颈证据时重写 MLX 算子。
每个保留优化记录：问题、改法、等条件前后指标、数值检查、内存变化。
无收益或不稳定实验撤回，恢复已验证版本，保留实验结论。

### 测量协议与性能目标

- 固定 checkpoint、tokens、dtype、输出长度、上下文、单 slot。
  社区与 C++ 不同精度/算法单列，不混为等条件结果。
- 快速档位为 512、2048、8192 输入 tokens，各 128 decode tokens；128 输入测短请求。
  长档位按 32K → 64K → 128K 逐级测试；容量 C 的吞吐测试使用 C−128 个实际输入
  tokens，再生成 128 tokens，报告精确长度。纯吞吐可忽略 EOS，产品测试正常处理 EOS。
- 每档至少一次热身、三次正式测量，记录中位数与最差值；加载/首次编译单列。
  不把未执行的 MLX 计算图构建速度当成推理速度。
- 分别报告 prefill tok/s、decode tok/s、Session 耗时、HTTP TTFT/总耗时。
  fresh prompt 与 cache hit 分开，输入长度取实际 token 数。
- 长档位分别测 fresh prefill、已有长前缀后追加输入、checkpoint 恢复后的续答，以及
  接近满容量时的 decode；为追加与输出预留空间，不能拿 2K decode 代替长上下文成绩。
- 记录源码/二进制/模型标识、命令、线程/slot、chunk、内存及运行条件。
  大模型测试串行，测量时暂停自己的下载、打包和其他 GPU 任务。

**调优目标**：2K 档 decode 和 prefill 吞吐分别争取达到同机固定 mlx-lm 参考的
90% 或更高；超过参考则继续按 C++ 自身瓶颈优化。90% 是目标，不是已测值或保证。
8K fresh HTTP TTFT 目标不超过 60 秒；2K decode 最低实用目标为 20 tok/s。
这两个短档指标不能替代长上下文验收。64K/128K 单列 fresh TTFT、追加输入 TTFT、
decode 与峰值内存，分别争取同长度社区参考吞吐的 90%；参考未完成时明确缺失。
长输入首次处理可能需要数分钟，不套用 8K 的 60 秒目标，也不承诺未经实测的绝对速度。
在首次取得长档基线后、调优前固定测试超时，并记录选择依据；超时视为失败，不能无界等待。
相对目标和绝对指标同时报告，不能降低参考速度或只报 cache-hit 来达标。
未达目标时继续处理最大的可解决瓶颈，在收尾解释剩余差距。

## 阶段四：服务验收与收尾（至少 1.5 小时）

不要把全部时间用在性能循环。最终推荐配置必须完成：

- 默认 127.0.0.1、单 slot、64K context，复用既有认证与访问控制。
- 预先固定至少 20 个短任务，覆盖中文、代码、上下文引用、工具调用及结果续答。
  明确客观 pass 条件，目标通过率至少 80%，逐项保留与社区参考的差异。
- 验证 JSON/SSE、thinking/non-thinking、EOS/stop/max_tokens、finish_reason、
  usage、工具参数和多轮续答；工具用无副作用 fixture。
- 覆盖取消后下一请求、checkpoint/cache、不同会话前缀隔离、上下文超限及错误返回。
- 64K 必须测试真实长输入，不能只设置容量后跑短请求。固定长文/代码 fixture，在前段、
  中段、末段放可核查信息，验证检索与跨段引用，并与社区同权重参考逐项对照。
  加入长历史后的工具结果追加、checkpoint 恢复续答、取消长 prefill 后的新请求。
  128K 按相同方法尝试；短序列数值通过不等于长位置计算与检索质量通过。
- 一轮至少 30 分钟且累计至少 100 个串行请求，混合短长请求、多轮、工具和取消。
  使用 64K 默认配置，包含接近容量的真实请求与长前缀复用，不要求所有请求都填满 64K。
  记录失败数、延迟、RSS/phys_footprint、MLX allocator、系统 swap 前后值。
- 无 crash/OOM/GPU error、状态污染、无界图增长或持续 swap 抖动。
  默认服务工作集目标不超过 36 GiB，给 macOS/IDE 留余量；区分 allocator cache 与泄漏。
- 重启后能离线启动；从不依赖 Python 环境变量的终端启动 C++ 产物，核实实际运行依赖。
- 128K 通过内存、长输入行为、缓存/恢复与实际生成检查后才推荐；记录其独立验收范围，
  不能把 64K 稳定性结果冒充 128K。不得靠丢弃/旋转 KV 或静默截断冒充支持。
- 若 64K 未通过，可交付已验证的小容量版本供排查，但必须标明本夜正式交付目标未完成。

回归按最终 diff 执行，不省掉仓库要求：

- 基础 C++ 改动运行 make test，bug fix 补最小回归。
- render/HTTP 边界改动运行相应 render-test/http-test。
- 新增 MLX 数值、Session、真实 35B 集成与性能测试。
- 修改原生 Metal 才触发其 GPU fixture 和规定的真实 reference；默认不修改它。
- 验证默认 CPU 构建不依赖 MLX；无 CUDA 的本机明确记录 CUDA 未重验。

## 无人值守执行与明早交接

上述约 8 小时分配用于控制投入，不是到点伪造完成的理由。开始时记录时间，尽早产出
可运行版本，保留至少 1.5 小时做验收。64K/128K 测试进入主线，不能全留到收尾。
时间紧时延后打包完善和次要微优化，保住 64K 验收；128K 至少完成有内存监测、
有界超时的真实尝试并报告结果，失败后可停止重复调优。不得悄悄降回 8K/32K 目标，
不转去开发 CPU/Metal，不跳过基本正确性与服务检查。

- 持续更新 build/qwen36-night/progress.md：已通过项、当前命令/进程、瓶颈、下一步。
  恢复上下文先读进度与 diff，不重做已完成工作。
- 获得执行授权后，普通可逆实现、build 内安装、下载、编译、测试和上述调优自主推进。
  外部命令有限重试；15 分钟无有效进展则定位或换已验证路径，不无限重启下载。
- 单项受阻时记录原因、推进独立工作，不能把社区 Python 服务当作 C++ 完成。
- 不改系统 GPU 限额、不自动 sudo、不关闭用户应用制造成绩，不上传模型或用户数据。
- 未明确要求不 commit、不 push；保留用户改动和可 review 的工作区 diff。

最终交付：

1. 真正运行的 MLX C++ 后端，以及可复制的构建、启动、健康检查、停止命令。
2. README 中本机 64K 默认配置、经验证的 128K 配置（若通过）、输出空间预留说明、
   文本/stream/tool 示例和完整运行依赖。
3. eval/qwen36-m5.md：来源、正确性、社区/C++ 对比、每项优化前后数据、内存、
   稳定性、失败项和限制。原始证据放 build/qwen36-night/。
4. 报告中的“明日结构评审”清单：文件职责、必要重复、临时适配、接口变更、
   权重容器/分发欠账、哪些优化有实测收益。今晚只记录，不为美化结构重做实现。
5. 列明每项性能目标是否达到，64K 正式验收与 128K 尝试的独立结果、失败原因和限制；
   CPU 35B PoC 和原生 Metal MoE 本夜未做。
6. 收尾检查工作区与遗留进程，停止自己启动的测试服务；未经要求不后台常驻生产服务。

## 已有准备：复用，不冒充模型完成

准备起点 7575868，执行前重新确认 HEAD。历史 Q4_0 CPU/CUDA 结果见
[eval/qwen36-35b-q4.md](eval/qwen36-35b-q4.md)，不能冒充本机 MLX 结果。
此前合并已通过基础测试、CPU_OPT=0、Metal fixture 与 0.8B CPU reference。
CPU 已支持 MoE，原生 Metal 尚未支持，MLX 后端尚未实现。

已核对 macOS 26.3.2、Xcode 26.6、48 GiB。Metal recommendedMaxWorkingSetSize
为 40,200,896,512 bytes（约 37.44 GiB），maxBufferLength 为 30,150,672,384 bytes。
这些是设备报告值，不是可同时占用的额外内存。开跑时重测空闲内存和 swap。

- [x] Metal Toolchain、shader 重新编译与真实 GPU fixture。
- [x] Python MLX GPU 小算子 probe。
- [x] C++ MLX GPU probe：wheel 动态链接与源码静态链接。
- [x] 固定社区 repo/revision，确认无 gated 登录要求；保留部分下载。
- [x] 社区 snapshot 完整校验与真实短推理。
- [x] 同权重短序列参考 logits 与 512/2K/8K 本机性能基线；长档位另行验收。
- [ ] C++ MLX 完整 forward、Session 与 HTTP。
- [ ] 原生量化路径正确性、调优和服务验收。
- [ ] 64K 真实长输入、性能、缓存/恢复与默认服务稳定性验收。
- [ ] 128K 真实尝试与独立结果报告；通过才推荐。

环境 build/mlx-venv/：Python 3.12.14、MLX/MLX Metal 0.32.2、mlx-lm 0.31.3、
Transformers 5.16.1、CMake 4.4.3、Ninja 1.13.2。冻结清单
build/qwen36-night/mlx-requirements.lock.txt；不升级滚动 main，不修改系统 Python。

源码 build/mlx-src/：tag v0.32.2，commit
1f8e74e3f12f31365464a6867c6579f0e9b29d85；静态安装 build/mlx-install/。
当前使用 Release、deployment target 26.3、MLX_METAL_JIT=OFF、BUILD_SHARED_LIBS=OFF，
关闭 tests/examples/benchmarks/Python bindings/GGUF，构建显式使用 TOOLCHAINS=Metal。
原准备阶段的 JIT=ON 小算子 probe 不足以发现完整模型的差异：其 BF16 sigmoid 与
wheel 预编译 kernel 有细小舍入差异。已通过最小实验定位，静态库改用预编译 kernel 后
与 wheel 一致；不要恢复 JIT=ON 而跳过数值回归。

MLX 0.32.2 静态 CMake export 未自动导入 jaccl::jaccl：在 find_package(MLX) 前
include 安装前缀的 lib/cmake/jaccl/jacclTargets.cmake。已验证例子在
build/qwen36-night/cpp-probe/CMakeLists.txt，不用重新排查。

静态 MLX 仍需 mlx.metallib，通过 mlx::core::metal::set_metallib_path 指定资源。
本夜可随产物附带并明确说明，内嵌资源/三文件分发留待结构评审。
运行不依赖 Python，不能因静态链接就声称不需要 shader 资源。

已验证的小程序命令（本次编辑不执行）：

```sh
build/qwen36-night/cpp-probe/static-build/mlx-probe \
  "$PWD/build/mlx-install/lib/mlx.metallib"
xcrun --toolchain Metal -sdk macosx metal --version
```

旧 mlx_probe.py/C++ probe 含 Q4_0 重排 fixture，只证明依赖与部分算子可用，
**不证明原生 affine4/group64、混合 8-bit 路径正确**，需补对应验证。
下载 manifests、断点与日志在 build/qwen36-night/。此前 ModelScope 大文件与固定 HF
snapshot 的 LFS SHA 匹配；可复用镜像，仍需验证固定 revision/文件 hash。
旧双模型 models-ready.json 不适合新主线，为社区 snapshot 记录独立完整性状态。

## 调研来源

实现以本地固定版本源码和实测为准，滚动网页仅作补充：

- [MLX C++ 接入](https://ml-explore.github.io/mlx/build/html/dev/mlx_in_cpp.html)
- [MLX 构建要求](https://ml-explore.github.io/mlx/build/html/install.html)
- [MLX 量化说明](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
- [MLX C++ 算子](https://ml-explore.github.io/mlx/build/html/cpp/ops.html)
- [MLX LM](https://github.com/ml-explore/mlx-lm)
- [固定社区量化配置](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit/blob/38740b847e4cb78f352aba30aa41c76e08e6eb46/config.json)
- [官方模型](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [M4 Pro gate/up 融合实验](https://github.com/fbzz/qwen3.6-mlx-inference)
- [MLX 长 prefill OOM 报告](https://github.com/ml-explore/mlx-lm/issues/1480)：同一模型
  snapshot 在较旧 MLX 0.31.2 上的报告；提示临时内存风险，不代表本机 0.32.2 必然复现。

此前社区 M5 Pro 成绩使用 20 核 GPU，本机是 16 核，不能作为本机性能保证。
本夜依据固定参考在本机的测量和 C++ 自身前后对比进行优化。
