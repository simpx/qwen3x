# Goal：极简 Qwen3.8-27B Q4_0 本地推理

## 目标

在当前 qwen3x 数据流和项目边界内，完成一条简单、正确、基本可用的
Qwen3.8-27B Q4_0 Apple Silicon 本地推理路径：

1. 保持项目简单、固定、可读，不增加兼容层，不扩展到目标之外。
2. 27B 只使用 Q4_0 权重，并通过真实模型和 Metal 正确性验证。
3. pi 能连接本地 27B 服务，完成一个真实的基础 coding task。
4. 在不破坏简单性的前提下优化 27B，使短、中上下文达到基本交互可用。

开始工作时先审计现有实现和证据。已经满足的部分只复验，不重写；只有存在明确缺口或失败
证据时才修改代码。

项目优先级固定为：

```text
correct -> simple -> readable -> usable -> fast
```

性能不能排在简单和正确之前。

## 固定边界

### 模型与量化

- 目标模型固定为官方 `Qwen/Qwen3.8-27B` text backbone，revision 固定为
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。
- 27B 只支持 Q4_0，不保留 Q8_0 中间方案，不增加其他量化格式。
- 这条限制只针对 27B；不删除或重做已有 0.8B、2B、4B、9B 路径，9B 可以继续使用 Q8_0。
- Q4_0 固定使用 ggml block 布局：每 32 个权重共享一个 FP16 scale，随后是 16 bytes
  packed nibbles，反量化整数范围为 `[-8, 7]`。
- 不使用 MLX 4-bit、GGUF runtime、GPTQ、AWQ、K-quants、混合量化或动态量化策略。
- vision encoder、mmproj、MTP 和 speculative decoding 不在目标内。
- 27B model ID、shape、tensor 顺序、类型和 alignment 都是固定代码，不做通用模型发现。

27B 固定 shape 为：

```text
V=248320  H=5120  I=17408  N=64
AI=4      AH=24   KVH=4    AD=256  RD=64
KH=16     VH=48   KD=128   VD=128  CK=4
tied_embeddings=false
```

### 数据格式与命名

- 系统自身统一使用 `qwen3x`：可执行文件、C ABI、源码模块和共享 render 数据都使用这个名字。
- 模型产物保留官方模型家族名，27B 固定为：

  ```text
  build/qwen38-27b-q4_0-model.bin
  ```

- model bin 只保存该型号的权重，使用唯一固定布局。
- 不保存 format version，不读取旧格式，不迁移旧产物，不做运行时格式探测。
- `qwen3x-render.bin` 独立保存所有支持型号共享的 tokenizer 数据，不与 model bin 合并。
- chat template 固定使用项目选定的 Qwen3.8 template，不为 Qwen3.5/3.8 维护多套模板分支。

### 平台与运行路径

- 27B 运行平台固定为本机 Apple M5 Pro、48 GB unified memory、macOS 26.3+、Metal 4。
- Metal 构建固定使用 Xcode 26.6 对应的 toolchain，不探测旧 Xcode、旧 Metal 或其他 Mac。
- 27B 不提供 CUDA、CPU 产品 fallback 或其他 GPU 的兼容承诺。
- CPU 实现和测试 dylib 可以作为 correctness baseline，但不是 27B 的部署路径。
- 一个 `qwen3x` 进程完成 HTTP、render、tokenizer、Session、sampling 和 Metal forward；
  不增加 worker、RPC、Python 服务或外部 tokenizer 进程。
- 服务固定绑定本地地址、单 Session、32768 context。
- 本地模型 generation 和 HTTP 请求不设置整请求 timeout；由正常完成、明确错误或用户取消
  结束。pi/bash 工具自身的命令 timeout 可以保留，不能把它混同为模型生成 timeout。

### 简单性约束

- 不增加 backend class hierarchy、op registry、模型图、自动 kernel generator 或通用 dtype
  dispatch framework。
- 不增加自动硬件探测、动态 tuning、运行时 tile 选择或多套兼容 kernel。
- Metal decode 保留一条从 embedding 到 logits 的完整单-token forward。
- prefill 只增加真实必要的固定 batch 路径；batch 大小由本机 benchmark 选定后写死。
- Q4 分支集中在权重读取和矩阵计算处；activation、recurrent state、KV cache、workspace 和
  logits 继续使用现有明确类型。
- 优化必须有 benchmark 或 profiler 证据；没有证据的复杂优化不进入项目。
- q3x、完整 EvalScope、其他模型性能和分发签名不属于本目标的完成条件。

## 实现与验证顺序

### 1. 固定 Q4_0 model bin

- 校验官方 checkpoint revision 和 `text_config`。
- packer 只打包 text backbone，输出固定 header、固定 tensor 顺序和严格 EOF。
- 记录 tensor 数、参数量、最终文件大小和 SHA-256。
- 单元测试覆盖 Q4_0 block 的 scale、nibble 顺序、舍入、零值、非有限输入、残缺 block、错误
  tensor、错误 shape 和截断文件。

这一阶段不能为了读取旧产物增加 version 或兼容分支。格式变化时直接重新 pack。

### 2. Metal correctness

- Q4 embedding、matrix-vector 和 batched matrix 与 CPU scalar baseline 对齐。
- 完整 forward 覆盖 DeltaNet、Attention、FFN、final norm 和 logits。
- 验证 recurrent state、KV cache、不同 batch 尾部、checkpoint restore 和 reset。
- batch prefill 必须与逐 token forward 的 position、完整 logits 和 argmax 一致。
- Attention 保持 causal 读取边界；DeltaNet recurrent update 严格按 token 顺序。
- 任何 NaN、越界、Metal validation error、静默 CPU fallback 或容差放宽都算失败。

至少运行：

```sh
make test
make -C tests render-test
make -C tests http-test
MTL_DEBUG_LAYER=1 make metal-test
```

合成小模型测试不能替代真实 27B 对齐；真实模型至少保存一个多 token batch/逐 token 完整
logits 对比结果。

### 3. 真实 27B 服务与 pi

固定使用：

```sh
make model-27b
make serve-27b
```

HTTP 验收至少覆盖：

- `/v1/models` 只返回实际加载的 `qwen3.8-27b`。
- 非流式 completion 正常结束。
- SSE content、finish chunk 和 `[DONE]` 完整。
- 错误 model ID 返回明确错误且不进入 generation。
- 普通日志不记录 prompt、生成内容或 API key；audit 只在显式开启时记录完整内容。
- 多轮 tool result 请求命中 Session prefix cache。

pi 使用仓库内 `scripts/pi.sh` 连接同一个 27B 服务，在隔离临时目录完成一个基础任务：

1. 读取目标代码和测试。
2. 找到一个明确的小 bug。
3. 只修改用户允许的文件。
4. 运行测试。
5. 根据测试结果继续或结束。
6. 给出与实际 diff 和测试一致的最终总结。

验收程序必须独立确认：原始 fixture 失败、最终测试通过、修改范围正确、没有越界文件、没有
timeout/terminated/截断。保存完整 JSONL、工具调用、文件 diff、独立测试和服务端耗时；这些
大体积证据放在 `build/`，结论写入 `eval/`。

一次简单聊天、只读回答、模型发现或模拟 HTTP 不算 pi coding task 通过。

### 4. 27B 性能优化

先测量，再只优化明确热点。优先考虑：

- 一个 SIMD group 协作计算一行 Q4 权重。
- 向量化 Q4 nibble unpack 和 scale 应用。
- 固定小 batch 复用同一组 Q4 权重完成多个 token 的 projection/FFN。
- 在保持状态顺序和 causal 语义的前提下批量处理 elementwise、DeltaNet 和 Attention 阶段。
- 减少不必要的 command buffer commit/wait 和中间内存往返。

禁止为了性能引入其他量化、通用调优框架、模型结构近似、精度降级或不可解释的 fused graph。
每个优化必须同时满足：

1. 合成 CPU/Metal correctness 继续通过。
2. 真实 27B batch/逐 token logits 继续一致。
3. benchmark 可重复证明改善；无改善或回退的实现删除。
4. 完整 forward 的调用顺序仍可从 `engine.mm` 直接阅读。

## 性能验收

使用 release build、单 Session、32768 context。每个 case 预热一次，再运行三次并取中位数；
模型 load 和 Session 创建时间单独记录，不计入 prefill/decode。

```sh
make metal-library
caffeinate -i python3 scripts/bench_session.py \
  --library build/metal/libqwen3x-metal.dylib \
  --model build/qwen38-27b-q4_0-model.bin \
  --context 32768 --prompts 128 512 4096 \
  --decode 128 --repeats 3 \
  --output build/bench-27b-q4.json
```

“基本可用”的最低目标固定为：

```text
128-token prefill   >= 20 tok/s
512-token prefill   >= 20 tok/s
短/中上下文 decode >= 10 tok/s
4096-token prompt 能完整运行并记录 TTFT/decode，不设虚假的短上下文硬门槛
```

同时记录：

- 128、512、4096 prompt 的 prefill、TTFT 和持续 decode。
- model load、Session create、Metal allocation、recommended working set 和 max buffer。
- process peak footprint、运行前后 swap、memory pressure level。
- pi 每轮 prompt/cached/new token、TTFT、总耗时和 decode tok/s。

如果最低目标未达到，先保存真实结果和 profiler 证据，再决定是否继续一个简单优化；不能用
旧 WIP 数字、单次最好值或短 prompt 外推长上下文。

## 分发与使用

本目标的最终运行文件只有：

```text
qwen3x                              分发名；构建产物为 build/qwen3x-metal
qwen38-27b-q4_0-model.bin
qwen3x-render.bin
```

运行已经编译好的产物只依赖 macOS Metal runtime，不依赖 Xcode、Python、Node、动态项目库或
额外服务。重新编译 shader/executable 和运行 pi 分别需要构建 toolchain 与 Node，但这些不是
模型服务的运行时依赖。

## 完成定义

只有同时满足以下条件，本目标才标记完成：

- [x] 27B model bin 是官方 Qwen3.8-27B text backbone 的固定 Q4_0 产物。
- [x] packer、基础 C++、render、HTTP 和 Metal 真机测试全部通过。
- [x] 合成 Q4 CPU/Metal 和真实 27B batch/逐 token 数值对齐通过。
- [x] 32768-token 单 Session 创建成功，没有 allocation failure 或持续增加 swap。
- [x] HTTP stream/non-stream、错误路径和 Session cache 通过。
- [x] pi 完成真实 read/edit/bash/test 基础任务，独立验收通过。
- [x] 128/512-token 性能达到上述最低目标，4096-token 结果完整记录。
- [x] 27B 路径没有其他量化、兼容层、fallback、动态 tuning 或范围外功能。
- [x] README、Make target 和 `eval/q4-27b-macos.md` 与真实行为和限制一致。

失败必须按实际状态报告，例如：

```text
Q4 format incomplete
Metal correctness incomplete
engine verified / pi incomplete
correct but below usable performance target
complete
```

不能用“可以加载”“输出看起来正常”“单元测试通过”代替完整验收。

## 修改与提交规则

- 先审计，再修改；不重做已经通过的实现。
- 不从旧 WIP 整体搬代码，只提取仍符合本目标且有证据支持的最小逻辑改动。
- commit 按逻辑阶段拆分，不把每个几行机械改动单独提交，也不把 model、runtime、Metal 和
  文档全部塞进一个 commit。
- 用户明确要求 commit 前，只保留可 review 的工作区 diff。
- 不自动 merge、push 或修改 main。
- 用户已有改动保持原样；发现重叠时先停止并说明。
