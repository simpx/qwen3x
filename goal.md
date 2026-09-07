# Goal：完成 qwen3x Apple Silicon 原生适配

## 今晚目标

在本机 Apple M5 Pro、48 GB 统一内存上，把 qwen3x 从 Linux/CUDA 项目完整适配为可开发、
可验证、可使用的 macOS 原生推理程序。

核心交付是：

```text
Qwen3.5-0.8B   CPU correctness baseline
Qwen3.5-0.8B   Metal correctness baseline
Qwen3.5-4B     BF16 Metal 推理、HTTP、eval
Qwen3.5-9B     Q8_0 Metal 推理、HTTP、eval、pi、q3x
Qwen3.8-27B    Q8_0 Metal、HTTP、pi、q3x 完整验证（下一阶段 stretch）
```

其中以下两个 Agent 目标属于核心完成条件，不是可选项或时间允许才做的 stretch：

1. **pi 端到端跑通**：pi 连接本机 9B Metal HTTP 服务，在隔离 fixture 中实际读取文件、
   修改指定文件、执行 bash/测试、根据测试结果继续修复，并由独立测试确认最终结果正确。
2. **q3x 端到端跑通**：编译后的独立 `build/q3x` 连接同一 9B Metal HTTP 服务，完成真实
   inspect、review、bugfix 和 continuation smoke，tool call、Session 恢复和最终独立验收通过。

只通过模拟 HTTP、单元测试、模型发现、服务启动或简单聊天，不算这两个目标完成。pi 与 q3x
必须分别留下请求 trace、工具调用、文件 diff、测试输出和最终 verdict；任何一个未通过，今晚
核心目标都标记为未完成。

最终日常使用方式保持简单：

```sh
make model-4b
make serve-4b

make model-9b
make serve-9b

make q3x
build/q3x

scripts/pi.sh
```

只有 0.8B、4B、9B 主线全部通过后，才进入 27B 下一阶段并增加、运行：

```sh
make model-27b
make serve-27b
```

## 固定方案

### 直接使用 Metal

- macOS GPU backend 直接使用 Objective-C++ Metal API 和仓库内 MSL kernel。
- `arch/metal/engine.mm` 负责 Metal device、buffer、pipeline、command queue 和调度。
- `arch/metal/kernels.metal` 负责模型数学和 GPU kernel。
- shader 编译后嵌入最终 executable；部署不依赖外部 `.metallib`。
- 保持一个 `qwen3x` C++ 程序完成 CLI、HTTP、render、runtime、sampling 和推理。
- Linux/CUDA 和通用 CPU 路径保留，不因 macOS 适配删除或分叉公共数据流。

本轮不链接或嵌入 MLX。MLX 不作为 build dependency、runtime dependency、模型格式或
内部服务，也不为了 benchmark 临时改变 qwen3x 的实现方式。

### 只使用 BF16 和现有 Q8_0

- 0.8B、4B 沿用当前 BF16 matrix 路径。
- 9B 沿用当前 Q8_0 matrix 路径。
- 27B 如果进入实现，只使用同一份现有 Q8_0，不增加新量化格式。
- Q8_0 继续保持每 32 个权重一个 FP16 scale 和 32 个 `int8_t`，沿矩阵行分 block。
- 官方 BF16 safetensors 是唯一权重来源；`pack_weights.py` 直接生成 qwen3x 顺序固定的
  `model.bin`。
- 不读取或分发 GGUF、MLX quantized safetensors、GPTQ、AWQ 或其他第三方模型容器。

只有在 27B Q8_0 实测因容量或带宽无法使用，并保存了 benchmark 和内存证据后，才另开目标
讨论简单 Q4_0。今晚不实现 4-bit、5-bit、K-quants 或通用 quantization framework。

### 模型范围

0.8B、4B、9B 继续使用已有 Qwen3.5 模型结构、chat template 和 model bin format。

27B stretch 使用官方 `Qwen/Qwen3.8-27B` text backbone，固定 shape 为：

```text
V=248320  H=5120  I=17408  N=64
AI=4      AH=24   KVH=4    AD=256  RD=64
KH=16     VH=48   KD=128   VD=128  CK=4
tied_embeddings=false
matrix_type=Q8_0
```

Qwen3.8 继续使用 `qwen3_5` architecture，但 attention output gate 为 `swish`。这个差异必须
由具名 `ModelConfig` 字段表达并通过 reference 验证，不能把 3.8 权重伪装成 3.5-27B。

27B 仍然只支持 text：

- vision encoder 和 mmproj 不进入 model bin。
- MTP layer 和 speculative decoding 不进入本轮。
- 使用官方 3.8 tokenizer/chat template 生成独立 render 数据，不能默认复用 3.5 render。
- 若正式支持 3.8，需要同步更新 `AGENTS.md` 中当前仅支持 Qwen3.5 的型号范围。

### 简单优先

- CPU 保留一份完整、直接的 Qwen forward。
- CUDA 保留一份 decode forward 和现有 chunk prefill。
- Metal 保留一份完整 decode forward，并增加一条具名 chunk prefill 路径。
- ModelConfig 只记录真实存在的 shape、matrix type、tied embedding 和 gate 差异。
- 不增加 backend class hierarchy、op registry、dtype dispatch framework、模型图解释器或
  自动 kernel generator。
- 性能优化从 benchmark 和 GPU capture 出发，不为了理论复用引入新抽象。

## 当前起点

- 已有初版 Metal backend，支持 BF16/Q8_0 单 token 完整 forward。
- 当前 `state_forward()` 每个 token 仍调用一次完整 forward；每 8 token 共用 command buffer，
  但不是批量矩阵 prefill。
- `make serve-4b`、`make serve-9b` 在 Darwin 已自动选择 Metal。
- 当前 Metal loader 先 mmap model.bin，再用 `newBufferWithBytes` 分 prefix/layer 复制权重，最后
  unmap；9B 可以接受，27B 必须测加载峰值并避免不可控的双份 residency。
- 本机已有 0.8B checkpoint、9B checkpoint 和 9B Q8_0 model.bin；4B、27B 尚未准备。
- 本机已有 pi；尚无 Bun。
- 本机当前只有 Command Line Tools，`xcrun --find metal` 不可用，完整 Metal 真机测试尚未开始。
- 当前工作树干净，`main` 比 `origin/main` 领先初版 Metal 提交。

## 执行规则

- 按下面阶段顺序执行；correctness 未通过时不做性能优化，核心阶段未完成时不进入 27B。
- 可以修改仓库文件、下载官方 checkpoint、创建 `build/` 测试产物、安装仓库本地 Python
  环境和用户目录下的 Bun。
- 完整 Xcode 若需要 GUI、Apple 登录、管理员密码或许可确认，停止并输出明确操作，不尝试
  绕过系统授权。
- 长任务使用 `caffeinate`，每个阶段保存命令、退出码、耗时和关键日志。
- 8000 端口被占用时先确认 PID、命令和所属服务；只停止已确认的 qwen3x 实例，不杀未知进程。
- 不删除已有 checkpoint/model.bin，不改写用户全局 Git 配置，不自动 push。
- 今晚不自动 commit；完成后保留可 review 的工作区 diff，由用户确认后提交。
- 遇到失败先保留最小复现和证据；不能用 skip、放宽误差或 CPU fallback 把失败写成通过。

## 实现顺序

### 0. 环境与资源门禁

- 记录 `uname`、`sw_vers`、芯片、CPU/GPU core、48 GB 内存和空闲磁盘。
- 检查工作树、当前 commit 和已有 build artifact，不覆盖用户改动。
- 检查 `xcode-select -p`、`xcrun --find metal`、`xcrun --find metallib` 和 Metal device。
- 完整 Xcode/Metal toolchain 可用后，编译一个最小 shader，再开始项目构建。
- 安装固定版本 Bun，运行 `bun --version`；确认 pi 可执行文件和版本。
- 检查 8000/8080 端口，记录并清理已确认的旧 qwen3x server。
- 检查电源、sleep 和可用磁盘。27B checkpoint 约需 56 GB，Q8 model.bin 约需 29 GB，
  下载、pack 和临时文件至少预留 100 GB。

Metal compiler 门禁失败时停止 GPU 主线并报告。不能把已有 CI 编译或 CPU 测试当作本机 Metal
通过。

### 1. 0.8B CPU baseline

- 准备 0.8B model/render，构建 `build/qwen3x`。
- 运行完整基础测试、render test 和 HTTP test。
- 用 CPU 跑 CLI prompt、prefill/decode、checkpoint restore、cache hit 和 reset。
- 保存 `--bench 128 32`、`--bench 512 64` 基线。
- 0.8B CPU 只承担 correctness baseline，不做本轮 CPU 性能优化。

### 2. Metal primitive 和 0.8B correctness

- 编译并嵌入 `kernels.metallib`，生成 `build/qwen3x-metal`。
- `MTL_DEBUG_LAYER=1 make metal-test` 必须在真实 Apple GPU 上通过。
- 运行 `make metal-reference` 和 `make metal-smoke`。
- 对齐完整 logits、top-k、argmax、recurrent state、KV、multi-token prefill、checkpoint restore、
  reset、cache hit 和 Session 隔离。
- 保持已有误差契约；误差超过契约时先定位 kernel、同步、offset 或 state，不提高容差掩盖问题。
- 日志必须输出实际 Metal device 和 working-set limit，禁止静默切回 CPU。

### 3. 4B BF16 与 9B Q8_0 真模型

- 下载并 pack 官方 Qwen3.5-4B，检查 config、tensor shape、alignment 和最终 EOF。
- 复用现有 Qwen3.5-9B Q8_0，重新校验 model hash/header 后使用。
- 为 4B 增加与 0.8B/9B 一致的 backend smoke vector/check target。
- 运行 4B BF16 和 9B Q8_0 的 logits/state/session smoke。
- CLI 短生成必须得到正常文本，无 NaN、乱码、重复死循环或 backend error。
- `make serve-4b`、`make serve-9b` 必须在 macOS 上加载 Metal executable，并完成真实请求。

### 4. Metal chunk prefill 与性能

当前逐 token Metal forward 只能作为 correctness 实现，不能作为完整适配的最终 prefill。

- 参考现有 CUDA chunk prefill 的真实数据流，在 Metal 中增加具名 batch BF16/Q8_0 matrix kernel。
- projection 和 FFN 的矩阵计算按 token chunk 批量执行。
- DeltaNet recurrent update 保持 token 顺序，不能改变状态语义。
- Attention 保持 causal mask、RoPE、KV 写入和读取顺序。
- runtime 的 `checkpoint_at` 仍是精确 range 边界，Metal chunk 不跨过该位置。
- 控制 command buffer 生命周期，减少每 token commit/wait；不依赖隐式同步。
- 对 BF16 和 Q8_0 分别调 token tile/threadgroup，但不建立通用 kernel tuning framework。
- profile 后只优化实测热点，保留完整 forward 的可读调用顺序。

性能验收使用 release build、单 Session、预热一次、重复三次取中位数：

```text
prompt/decode: 128/128、512/128、4096/128
context:       40960
models:        0.8B CPU、0.8B Metal、4B Metal、9B Metal
```

每项记录 load time、prefill tok/s、decode tok/s、TTFT、process peak footprint、swap 变化、
Metal working-set limit 和是否发生 memory pressure。

9B 的最低可用门槛：

```text
4096-token prefill <= 5 s（等价于 prefill >= 819 tok/s）
短上下文 decode >= 30 tok/s
4096-token prompt 的 runtime TTFT <= 5 s
Metal prefill/decode 均快于相同模型 CPU baseline
```

这里的 prefill 时间和 TTFT 使用已经加载模型、完成一次预热后的纯 runtime benchmark；模型
加载、HTTP 建连、JSON/render 和首轮 shader/pipeline 初始化另行记录，不混入 5 秒门槛。
decode TPS 使用 prefill 完成后的连续 128 token 测量，排除首 token，并以三次运行的中位数
验收。任一硬门槛未达到时，保存 profile 和实际结果，性能状态标记为未完成。

如果未达到门槛，保存 benchmark/profile，继续针对热点优化；未达标不能称为性能适配完成。

### 5. HTTP、eval 和产品入口

- 覆盖 `/v1/models`、非流式 chat、SSE stream、thinking on/off、错误请求和客户端断开。
- 覆盖正常多轮、Session cache、checkpoint reuse、tool call 和 tool result。
- 普通日志不泄露 prompt、生成内容或 API key；audit log 仍由显式参数开启。
- 4B/9B server 默认只绑定 `127.0.0.1`，单 Session，端口和 PID 行为清楚。
- 运行 4B、9B 的 EvalScope smoke：MMLU-Pro、CEval、IFEval 各 2 题。
- 保存原始预测、manifest、smoke report、耗时和服务日志；smoke 只证明链路可用，不声明模型
  完整能力。
- 更新 README 中 macOS 构建、模型准备、serve、benchmark、pi/q3x 和已知限制。

### 6. q3x 与 pi

先完成确定性测试：

```sh
make q3x-test
make q3x
Q3X_TEST_BINARY="$PWD/build/q3x" make -C agent test
make -C tests pi-retry-test
```

然后让 9B Metal server 跑真实 Agent smoke：

- q3x 对 `inspect-flow`、一个 `review`、`fix-csv` 各跑至少一次。
- q3x continuation 跑一次压缩、重载 Session、继续任务，检查副作用不重复。
- pi 通过 `scripts/pi.sh` 连接 9B，完成一个隔离 fixture：读取文件、修改指定文件、运行 bash
  测试并根据失败修复。
- fixture 在临时 Git 仓库中执行，不允许模型修改 qwen3x 工作树或目录外文件。
- 保存完整 JSONL/trace、diff、测试输出、usage、TTFT、总耗时和 verdict。
- tool-call parse error、未完成 tool call、异常重试、声称完成但独立测试失败，都算失败。

若时间允许，再运行 q3x `daily.ts` 的全部场景；首次 Mac 验收不要求把每项重复三次。

### 7. Qwen3.8-27B Q8_0 下一阶段

只有 0.8B/4B/9B correctness、9B 性能、HTTP、eval、q3x 和 pi 全部完成后才进入。

#### 格式与数学

- 为 Qwen3.8-27B 增加唯一 model ID、固定 ModelConfig 和 packer config 校验。
- 继续使用 model bin 当前顺序格式；只有确有必要时才升级 format version。
- attention output gate 的 `swish` 与 Qwen3.5 的现有行为分别由 config 表达。
- CPU、CUDA、Metal 的同一数学位置都实现该差异；未能运行 CUDA 时至少完成编译/静态检查，
  并明确记录 CUDA 真机未验证。
- 增加 packer 错误 config、错误 gate、错误 tensor、shape、截断和 EOF 测试。
- 增加短 token 的 CPU/Metal logits、state、checkpoint 和 greedy continuation 对齐。
- 官方 BF16 reference 若因 48 GB 容量无法运行，可以使用 disk offload 生成极短 reference；
  如果仍无法得到官方 reference，状态只能标记为 experimental，不能称完整 verified。

#### 内存门禁

- pack 前输出精确 text tensor 参数量、Q8 model.bin 大小和每层 region 大小。
- 记录 `device.maxBufferLength`、`recommendedMaxWorkingSetSize` 和实际 model upload 峰值。
- Metal loader 不保留无必要的完整第二份权重；优先使用简单、明确、可审阅的 mmap/shared-buffer
  生命周期，不引入通用 paging/offload framework。
- 先创建 `session_slots=1, context=16384`，成功后才测试 32768。
- steady-state memory pressure 为 yellow/red、发生持续 swap、GPU allocation 失败或系统明显失去
  响应时，停止扩大 context。
- 今晚不测试 64K、128K、262K，不增加 CPU/GPU 分层 offload。

#### 最小验收

- model.bin 完整 pack、header/schema/EOF 校验通过。
- Metal 成功加载，16K 单 Session 创建成功。
- 运行 16/64/512-token prompt 和 16/32-token decode，无 NaN、崩溃或错误输出。
- 记录 load、prefill、decode、峰值内存和 16K 后剩余 headroom。
- 16K 稳定后尝试 32K；32K 失败不影响 0.8B/4B/9B 核心目标。
- `make serve-27b` 提供与 4B/9B 相同的 OpenAI-compatible stream/non-stream、thinking、
  tool-call 和 Session cache 行为。
- 编译后的 `build/q3x` 连接 27B Metal 服务，至少完成 inspect、review、bugfix 和
  continuation 各一次，并通过 fixture 独立验收。
- pi 连接 27B Metal 服务，在隔离 fixture 中完成一次 read、write、bash、test、根据失败继续
  修复的完整 coding task，并由独立测试确认结果。
- 27B 不要求跑完整 EvalScope 或 q3x daily 三次重复，但 pi、q3x 端到端都是 27B 阶段的硬性
  完成条件；任意一个失败时，27B 只能标记为 engine verified / agent incomplete。
- 27B 暂不设吞吐硬门槛，但必须记录实测 TPS、TTFT 和内存，并明确其交互可用性。

27B 最终必须明确归类为以下之一：

```text
verified at 32K with pi/q3x
verified at 16K with pi/q3x
engine verified / agent incomplete
loadable but not reference-verified or agent-verified
blocked by Metal working set / allocation
not attempted because core goal incomplete
```

## 测试矩阵

核心代码修改后至少运行：

```sh
make test
make -C tests render-test
make -C tests http-test
make -C eval test
MTL_DEBUG_LAYER=1 make metal-test
make metal-reference
make metal-smoke
make metal-smoke-9b
make q3x-test
make -C tests pi-retry-test
```

真模型服务完成后运行：

```sh
make -C eval smoke-4b
make -C eval smoke-9b
make -C eval agent-eval AGENT_SCENE=all EVAL_MODEL=qwen3.5-9b
```

如果增加新的 `metal-smoke-4b`、`metal-smoke-27b` target，也必须纳入最终命令列表。测试命令
因环境依赖无法运行时，要记录 `blocked` 和具体依赖，不能写成 pass。

## 完成定义

### 核心完成

必须同时满足：

- [ ] 0.8B CPU baseline、基础 C++、render、HTTP 测试通过。
- [ ] 本机 Metal shader、executable 和 primitive test 通过。
- [ ] 0.8B Metal reference/smoke 通过。
- [ ] 4B BF16、9B Q8_0 Metal logits/state/session smoke 通过。
- [ ] 4B、9B CLI 与 OpenAI-compatible HTTP stream/non-stream 通过。
- [ ] Metal chunk prefill 已实现，9B 达到最低可用性能门槛。
- [ ] 4B、9B EvalScope smoke 完成并生成报告。
- [ ] q3x 确定性测试、Mac 独立 binary，以及连接 9B Metal 服务的 inspect、review、bugfix、
  continuation 端到端任务全部通过。
- [ ] pi retry test，以及连接 9B Metal 服务的真实 read/write/bash/test 端到端任务通过，并由
  fixture 独立测试确认修改正确。
- [ ] README、Make target、测试说明和 benchmark 报告与实际行为一致。
- [ ] CPU 路径未回归；CUDA 未删除，能够执行的构建/静态回归已完成。

### 27B 完成

27B 是核心完成后的独立下一阶段，不影响 0.8B/4B/9B 核心完成状态。27B 阶段只有在至少
16K Q8_0 model load、CPU/Metal smoke、短生成、HTTP、pi 端到端、q3x 端到端和内存报告全部
通过时才算完成。仅有 engine load/生成不算 27B 完成；没有官方 reference 时还必须保留
experimental 标记。

## 最终产物

- `build/qwen3x`：macOS CPU baseline。
- `build/qwen3x-metal`：嵌入 shader 的 Apple Silicon executable。
- `build/qwen35-4b-model.bin`：4B BF16。
- `build/qwen35-9b-q8_0-model.bin`：9B Q8_0。
- 可选 `build/qwen38-27b-q8_0-model.bin`：27B Q8_0 stretch。
- `eval/metal-macos.md`：硬件、命令、正确性、性能、内存、eval、pi/q3x 结果和限制。
- 可选 `eval/q8-27b-macos.md`：27B 容量、reference 状态和结论。

最终汇报必须列出：

```text
实现了什么
没有实现什么
所有修改文件
所有测试及退出结果
各模型 benchmark
各模型的 pi/q3x verdict 和失败样本
27B 的阶段状态
仍需用户处理的外部依赖或风险
```

不能用“编译通过”“服务启动”或“一次输出看起来正常”代替上述完成条件。
