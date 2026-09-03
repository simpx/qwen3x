# qwen3x 开发约定

本文件记录 qwen3x 的长期目标和个人开发原则，适用于整个仓库。实现新功能时先确定它在
整体数据流中的位置，再选择最直接的实现。

## 项目目标

qwen3x 是一个个人开发的、极简、本地优先的 Qwen C++ 推理引擎，用于教学、研究和 PoC。
型号范围固定为 Qwen3.5-0.8B、2B、4B、9B 和 27B，不在本项目中扩展到其他架构或型号。

项目希望用尽可能少的代码展示一条真实、完整、可运行的推理数据流。读者应当能从
`main()` 出发，一直读到模型 forward：

```text
CLI / HTTP
  -> OpenAI request parser
  -> Qwen chat template
  -> tokenizer
  -> Session prefill / decode / sampling
  -> Qwen forward
  -> token decode
  -> text / JSON / SSE
```

项目优先级是：

```text
correct -> simple -> readable -> usable -> fast
```

## 总体结构

### 一个 C++ 程序

- 一个 `qwen35` 可执行文件完成所有推理相关工作，包括 CLI、HTTP、JSON、chat template、
  tokenizer、session、模型计算、sampling 和流式输出。
- CLI 单次请求、Session benchmark 和 HTTP listen 是同一个程序的不同入口，共用同一套
  render、runtime 和 engine 数据流。
- 部署时只启动这个进程。模型运行不依赖 Python、动态链接的项目库、内部 RPC、额外 worker
  或 tokenizer 服务。
- `qwen35.h` 提供 Engine/Session 的窄 C ABI；正常构建将其实现直接链接进 `qwen35`。为了
  逐 token 数值对齐，`reference/` 可以把同一实现临时编译成 `reference/build/libqwen35.so`
  供 ctypes 驱动。这个 shared library 是测试适配器，不是部署方式或分发产物。
- 理想形态是单文件 C++；当一个文件已经妨碍阅读时，才沿真实数据流拆分。源码文件数量
  保持少，文件边界表达职责，而不是表达框架层次。

当前核心文件按数据流拆分：

```text
main.cpp       main、CLI、HTTP routes、completion 编排
parser.cpp     JSON <-> 简单 C++ 请求/响应结构
render.cpp     Qwen chat template、tokenizer、token decode
runtime.cpp    Session、cache/checkpoint、prefill、decode、sampling
engine.cpp     权重、State/Work、算子和完整 Qwen forward
arch/cuda/     CUDA engine；chunk prefill 与单 token decode forward
arch/metal/    Metal engine；完整单 token forward 与 MSL kernel
log.cpp        进程级日志实现
```

这些文件通过直接函数调用和明确数据结构连接。数据所有权、错误返回和执行顺序应当能从调用
点一路追踪。

### Qwen 定制，而非通用框架

- 模型结构、常量、chat template 和数据格式围绕当前 Qwen 型号直接表达。
- 已经出现且反复存在的需求才提取公共抽象。
- engine 直接展示 Qwen 数学，runtime 直接展示 Session 状态变化，render 直接展示文本到
  token 的过程。
- 新模型优先复用清晰的数据流和工程结构，再针对真实差异扩展；模型兼容性本身不是抽象
  层数的理由。

### 模型抽象原则

- model bin 使用固定布局和 model ID 表达已支持型号；header、tensor 顺序、类型、alignment
  和 EOF 检查保持唯一、明确。
- `ModelConfig` 集中记录每个型号的固定 shape。同一 Qwen3.5 结构共享一份完整 CPU forward 和
  一份 CUDA decode forward；只有计算结构或数值路径不同时才增加具名 forward/prefill。
- forward 从 config 读取 `H、I、N、AH、KVH、VH`，完整展示 embedding、layer loop、
  DeltaNet/Attention、FFN、final norm 和 logits。分支直接留在 backend 入口和 layer loop。
- Metal 独立展示同一完整 forward；系统 API 收敛在 Objective-C++ 平台文件，数学写在
  `.metal` kernel。复用现有 model bin 和 runtime 边界，量化分支集中在 embed/mv。
- File/Reader、固定布局 loader、Model/Layer、State/Work、checkpoint、模型算子和 CUDA kernel
  保持内聚、可组合，由完整 forward 直接编排。
- 新型号依次加入 model ID、`ModelConfig`、packer 和 reference 测试；结构或数值路径不同时再
  增加对应的 CPU/CUDA 主流程。

## 多语言边界

生产推理路径使用 C++。Python 有两个明确用途：

1. `scripts/` 和 `tests/` 中的离线脚本、转换工具、开发客户端和自动化测试。
2. `reference/` 和 `eval/` 中使用 PyTorch/Transformers 等官方生态进行数值对齐与评测。

Python 产出测试向量、模型数据或验证结果；`qwen35` 运行时独立消费最终二进制数据。这样
开发阶段可以利用成熟生态，最终分发仍保持纯 C++、本地和自包含。

reference 的 ctypes 包装只调用上述 C ABI，用于控制 Session、checkpoint 和读取完整 logits；
HTTP、JSON、render 和产品服务接口仍由 `qwen35` 可执行文件统一提供。

JSON 同样只是一种边界语言：`parser.cpp` 将它转换为普通 C++ 数据；render、runtime 和
engine 只认识自身需要的明确类型。

## C++ 风格

项目使用 C-oriented、exception-free C++17，也就是“用 C++ 管理资源，用 C 表达数据流”：

- 构建保持 `-std=c++17 -fno-exceptions -fno-rtti`。
- namespace 用于模块命名，class 和 RAII 用于资源所有权，struct 用于普通数据，函数和显式
  参数用于过程。
- `std::unique_ptr`、`std::vector`、`std::string`、`auto` 和简单 lambda 可以用于让
  生命周期或局部代码更清楚。
- 核心模型计算使用数组、指针、循环和显式 shape，保持接近数学公式和 C 实现。
- 可恢复错误通过返回值、小型 `Status` 或现有错误缓冲区报告。对象 factory 使用非空指针
  表示成功、`nullptr` 表示失败。
- 内部不变量失败时，先记录表达式和运行上下文，再 assert/abort。
- 模板限于简单、局部、一眼可以展开理解的用途，例如类型转换。宏主要用于日志和 assert
  这类统一边界。

“没有隐藏魔法”是判断代码风格的核心标准：控制流在调用点可见，内存所有权有明确对象，
状态变化发生在容易找到的位置。新增封装应当减少理解成本，而不只是减少代码行数。

## 依赖原则

项目尽可能依赖标准库和仓库内代码。确实需要第三方能力时，依赖按以下顺序选择：

1. 成熟、行为稳定、使用广泛。
2. 接口小，源码容易审阅。
3. header-only 或少量源码，可直接 drop in 到 `third_party/`。
4. 构建时无需包管理器，运行时无需额外服务和动态库。
5. 解决明确的非核心问题，例如 JSON、HTTP 和日志。

模型 forward、Session、Qwen render 等项目核心能力保留在仓库代码中。HTTP、JSON 等成熟
但边界情况繁多的通用协议优先使用符合上述条件的小库。

第三方代码固定版本并放在 `third_party/`；对应说明记录来源、版本和用途。引入新依赖时，
同时说明它替代了什么复杂度，以及最终 binary 增加了什么。

## 分发产物

目标分发形态只有两个文件：

```text
qwen35                    平台对应的单个可执行文件
qwen35-0.8b-model.bin     该模型运行所需的单个数据文件
```

可执行文件包含程序逻辑、HTTP 服务、Qwen template 和通用协议实现；model bin 包含权重、
tokenizer 数据及其他随模型变化的数据。使用其他型号时只替换对应 model bin。

开发阶段为了调试和快速迭代，暂时保留：

```text
qwen35-0.8b-model.bin
qwen35-0.8b-render.bin
```

`render.bin` 稳定后合并进 model bin。官方 checkpoint、转换中间文件和测试向量都是开发
产物，不属于最终分发集合。

## 数据与目录

- 根目录保持紧凑，只放核心 C++、公开/内部 header、`Makefile` 和项目文档。
- `scripts/` 放下载、pack、开发客户端等离线工具。
- `tests/` 放 parser、render、runtime、CLI 和 HTTP 回归测试。
- `reference/` 放官方 PyTorch/Transformers 数值参考。
- `eval/` 放评测工具和结果。
- `third_party/` 放固定版本、可直接构建的第三方源码。
- `build/` 统一容纳官方 checkpoint、生成的 model/render、编译结果和测试产物，并由
  `.gitignore` 隔离。
- 官方 checkpoint 目录保持上游原始内容；pack 后的文件放在独立位置。
- 早期教学代码保存在 Git tag `learning`；主线保持当前产品实现所需的最小结构。

## 模型正确性与性能

- 公式 shape 和 checkpoint 实际存储布局分别说明。模型代码使用 `H、I、D、T、V` 等固定
  符号，并让矩阵乘法维度能够直接检查。
- `engine.cpp` 中一个 token 的完整 forward 从上到下展开，作为最容易理解和验证的实现。
- CPU prefill 逐 token forward，作为 correctness baseline。
- CUDA prefill 在 backend 内按 chunk 批量调度；runtime 把 `checkpoint_at` 作为精确 range
  边界，CUDA chunk 不跨过该边界。CUDA decode 保留可直接阅读的单 token forward，并用
  CUDA Graph replay 相同的具名 kernel 顺序。
- checkpoint 恢复后应等价于已经处理相同前缀；cache 优化通过 token、logits 和生成结果
  一致性测试验证。
- 性能优化由 benchmark 或 profiler 数据驱动，并保持模型数据流可读。
- Session benchmark 测量纯 prefill/decode；端到端 benchmark 单独测量 HTTP、JSON、
  template、tokenizer 和 sampling。

## 协议、日志与开发工具

- HTTP 对外兼容 OpenAI Chat Completions，Qwen 特有行为收敛在 parser/render 边界。
- `scripts/chat.py` 是忠实的薄 DSL 翻译器，可以构造正常或反常请求，并通过
  `--dry-run` 生成可直接执行的 curl。
- 每个 HTTP 请求在最外层记录两条生命周期日志：取得 request ID 时记录
  `access started`，最终成功、失败或断开时记录且只记录一条 `access completed`。
- `info` 展示服务负载和请求生命周期，`debug` 展示阶段与 cache 状态，`trace` 展示
  per-token/forward 细节。prompt/benchmark 默认安静，listen 模式默认展示服务状态。
- 日志通过 request ID 关联完整流程并记录指标；prompt、生成内容、API key 等敏感数据留在
  日志之外。
- 显式 `--audit-log` 将完整请求、模型输入输出和 HTTP/SSE 输出写入独立 audit 文件；
  request ID 关联请求，session ID 关联复用的 Session。audit 不进入普通日志且默认关闭。

## 修改、测试与提交

- 解读、review 或讨论请求以读取和解释为主；“改/实现”请求进入代码修改。
- diff 按独立行为或文件拆成小而内聚的单元，方便逐个 review 和提交。
- commit 由用户明确触发。需要先 review 时，完成改动和测试后保留工作区 diff。
- 用户已有改动保持原样；工作区有变化时先区分改动归属。
- 基础 C++ 改动至少运行 `make test`；render 边界运行 `make -C tests render-test`；HTTP
  数据流运行 `make -C tests http-test`。bug fix 配套最小回归测试。
- Metal 改动在 Apple Silicon 上运行 `make metal-test`；真实 0.8B 数值验收运行
  `make metal-reference`。WSL shader 离线编译和无 GPU 的 CI 构建不算 GPU 正确性通过。
- push main 前确认提交内容和测试结果；远端历史出现分叉时停止 push 并向用户说明。
