# Goal：完成 Qwen3.5-9B Q8_0

## 今晚目标

在 qwen3x 中完整支持官方 Qwen3.5-9B 的 text model，使用标准 Q8_0 权重运行在
RTX 4080 SUPER 16 GiB 上，并通过 CPU、CUDA、runtime、HTTP 和 pi 的完整测试。

最终使用方式：

```sh
make model-9b
make serve-9b
pi
```

0.8B 和 4B 现有路径保持可用。

## 固定方案

### 模型

- 使用官方 `Qwen/Qwen3.5-9B` BF16 checkpoint 作为唯一权重来源。
- model ID 使用 `9000`，`ModelConfig` 固定记录 9B shape、Q8_0 matrix type 和
  `tied_embeddings=false`。
- text model 保存独立的 token embedding 和 `lm_head`。
- model bin 继续使用唯一、顺序固定、64-byte aligned 的 qwen3x 布局。
- vision encoder 和未参与普通生成的 MTP 权重不进入 model bin。
- model bin 保持 format v2，header reserved 字段保持为零；model ID 唯一决定 matrix type、
  tied 配置和 tensor 布局，不增加运行时可选 dtype 字段。
- 现有 0.8B/4B model.bin 无需重新 pack；旧 binary 对 9B 返回 unsupported model ID，不能把
  Q8_0 数据误读成 BF16。

### Q8_0

Q8_0 与 llama.cpp `quantize_row_q8_0_ref` 保持相同含义：

```cpp
struct Q8Block {
    uint16_t scale;
    int8_t values[32];
};
```

每 32 个 BF16 权重转换为一个 block：

```text
scale = max(abs(weight)) / 127
q[i] = round(weight[i] / scale)
weight[i] ~= fp16(scale) * q[i]
```

- block 沿矩阵行排列，不跨行；所有 Q8_0 matrix 的 `cols` 必须能被 32 整除。
- scale 先以 FP32 计算，再转换为 IEEE FP16 写入 little-endian model.bin；反量化使用文件中
  实际保存的 FP16 scale。
- `round` 与 C `roundf` 一致，halfway case 远离零取整。
- 全零 block 保存 `scale=0` 和 32 个零值。

量化以下矩阵：

- token embedding 和独立 `lm_head`。
- DeltaNet 的 qkv、z、a、b、out。
- Attention 的 q、k、v、out。
- FFN 的 gate、up、down。

RMSNorm、q/k norm、conv1d、dt bias 保持 BF16；A_log 和 DeltaNet norm 保持 FP32。
约 99.988% 的参数进入 Q8_0，text weights 预计从 16.68 GiB 降至 8.86 GiB。

### Forward

- activation、recurrent state、KV cache、workspace、accumulator 和 logits 保持 FP32。
- CPU 和 CUDA 各保留一份完整 Qwen forward。
- `Linear` 明确记录 `MATRIX_BF16` 或 `MATRIX_Q8_0`。
- CPU 在 `embed()` 和 `mv()` 中集中 switch BF16/Q8_0。
- CUDA 在 weight-consuming wrapper 中集中 switch，调用具名 BF16/Q8 kernel。
- DeltaNet、Attention、FFN、layer loop 和 residual 主流程不出现量化分支。
- CUDA Graph capture 后固定为选定型号的 kernel 序列，decode replay 不再执行类型分支。

### CUDA

CUDA 增加以下 Q8_0 计算：

```text
embed_q8
mv_q8 / mv_add_q8
delta_projections_q8
attention_projections_q8
ffn_projections_q8
batch_mv_q8
```

kernel 直接读取 Q8 block，在寄存器中反量化，与 FP32 activation 相乘并以 FP32 累加。
prefill 的 batch kernel 按 token tile 复用读取到的 Q8 block。

4B prefill 继续使用 BF16 cuBLAS；9B prefill 使用 Q8_0 × FP32 kernel。9B 不创建完整
BF16 权重副本，不量化 activation，也不分配 4B 路径使用的 BF16 activation conversion
buffer。

## 代码结构要求

### 文件职责

```text
model_config.h          固定 model ID、shape、matrix type 和 tied 配置
q8.h                    Q8Block 固定布局和必要常量
engine.cpp              CPU Model/load、State/Work、算子和唯一完整 forward
arch/cuda/engine.cu     CUDA Model/load、BF16/Q8 kernel、唯一 decode forward 和具名 prefill
scripts/pack_weights.py 官方 checkpoint 校验、固定 tensor 顺序和 Q8_0 pack
runtime.cpp             Session、cache、prefill/decode 调度，不认识量化细节
```

`q8.h` 只表达一种具体磁盘和内存格式，不发展成通用 quantization API。Q8 CPU 代码留在
`engine.cpp`，Q8 CUDA kernel 在 `engine.cu` 中连续放置；只有实际代码已经妨碍主流程阅读时
才增加具名、内聚的实现文件，不建立 `ops/`、backend class hierarchy 或多层目录。

`engine.cpp` 继续从 Model、State/Work、基础计算读到完整 forward，目标保持在约 500 行。
优先通过删除重复流程和保持小函数直接来控制长度，不为满足行数机械搬移代码。

### 主流程

- CPU 只有一个 `forward()`；CUDA 只有一个 decode `forward()` 和一个供 4B/9B 复用的
  matrix prefill 主流程。
- BF16/Q8 复用同一组 Model、Layer、State 和 Work 数据结构。
- `forward()` 完整展示 embedding、layer loop、DeltaNet/Attention、FFN、final norm 和
  `lm_head`；不通过模板、宏、函数指针或 ops object 生成主流程。
- Q8 只改变 Linear 的存储和计算。runtime、checkpoint、cache、sampling、render、HTTP 和
  pi 数据流不增加量化分支。

### 分支位置

允许的量化分支集中在：

```text
ModelConfig / loader       model ID 决定固定布局
CPU embed / mv             选择 BF16 或 Q8_0 实现
CUDA weight wrappers       选择 BF16 kernel/cuBLAS 或 Q8_0 kernel
CUDA prefill 入口          0.8B FP32 路径或 4B/9B matrix 路径
```

CUDA weight wrappers 只包括 `embed`、`mv`、`mv_add`、三个 fused projection 和
`batch_mv`，并在文件中集中排列。switch 发生在 host wrapper；选中的 device kernel 只处理
一种明确类型。

layer loop 中只保留模型本身的 DeltaNet/Attention 分支，不出现 `if (quantized)`。DeltaNet、
Attention、FFN、norm、residual、state 和 cache 内不传播 model ID 或量化 flag。

### 直接实现

- 使用具名函数：`mv_bf16`、`mv_q8`、`batch_mv_bf16`、`batch_mv_q8`。
- `Linear` 只保存 weight view、rows、cols 和明确的 matrix type，不拥有数据。
- Model 继续唯一拥有 mmap 或 CUDA weight allocation；State/Work 的所有权不变。
- loader 按唯一 tensor 顺序切指针并检查 EOF，不引入 tensor registry、name lookup 或动态
  dtype schema。
- packer 直接把官方 BF16 tensor 转换成最终 model.bin，不经过 GGUF 或中间量化容器。
- 不使用虚函数、继承、`std::variant`、通用 dtype dispatch table、量化模板框架或宏生成
  BF16/Q8 代码。

## 实现顺序

### 1. 固定格式与 pack

- 增加共享的 `Q8Block` 定义和静态布局检查。
- 增加 9B `ModelConfig`、官方 config 校验和独立 `lm_head` 布局。
- packer 分块读取 BF16 safetensors，用 NumPy 生成 Q8_0，不把 checkpoint 整体载入内存。
- pack 完成后检查 tensor 数量、每个 tensor shape、alignment、最终 EOF 和文件大小。
- 增加 `make model-9b`，命令自带项目约定的默认 checkpoint 和输出路径。

### 2. CPU correctness

- loader 按 model ID 绑定 BF16 或 Q8_0 matrix。
- 实现 Q8_0 embedding、dot 和 matrix-vector multiply。
- 让独立 `lm_head` 进入 logits 路径。
- 保持完整 CPU forward 只有一份。

### 3. CUDA decode

- 直接上传混合 BF16/FP32/Q8_0 model bin，不展开 Q8 权重。
- 实现 Q8 embedding、projection、mv 和 mv_add kernel。
- 让同一份 decode forward 和 CUDA Graph 支持 9B。

### 4. CUDA prefill

- 实现 Q8_0 × FP32 batch matrix multiply。
- 复用现有 chunk、checkpoint_at、DeltaNet recurrence 和 Attention KV 数据流。
- 保持 chunk 不跨 checkpoint 边界。
- 用 profiler 数据调整 token tile 和 block shape，不改变模型主流程。

### 5. 产品入口

- 增加 `make serve-9b`，默认使用 40960 context、单 Session slot 和本地 audit log。
- `scripts/pi-models.json` 增加 `qwen3.5-9b`，使用 `contextWindow=40960`、
  `maxTokens=4096`、`reasoning=true` 和 `thinkingFormat=qwen-chat-template`。
- pi 的 model ID 与 server model name 保持一致；保留现有 incomplete tool-call 重试、streaming
  tool-call 和 audit 行为。
- 更新 README 中的模型准备、启动服务、连接 pi、显存需求和当前能力边界。

## 测试与验收

### 格式

- 固定向量的 Q8_0 bytes 与 llama.cpp reference algorithm 一致。
- 覆盖全零、正负极值、rounding 边界和随机 BF16 block。
- packer 拒绝错误 config、错误 dtype、缺失 tensor、错误 shape 和截断 shard。
- loader 拒绝错误 model ID、header、布局、alignment、截断和多余数据。

### 数值

- CPU Q8_0 dot/mv 与 Python reference 一致。
- CUDA Q8_0 embed、mv 和 batch mv 与 CPU Q8_0 结果满足固定误差契约。
- 9B CPU 与 CUDA 覆盖 fresh decode、multi-token prefill、chunk boundary、checkpoint restore、
  Session cache 和多步 greedy continuation。
- 与官方 BF16 reference 比较 logits、top-k、argmax 和短文本；记录量化误差，确认 greedy
  分歧没有来自布局、kernel 或 state bug。
- 0.8B、4B 的既有 reference 和回归测试继续通过。

### 外部 Q8_0 基线

- 固定 llama.cpp 版本、Unsloth `Qwen3.5-9B-GGUF:Q8_0` revision 和运行参数。
- 用 llama.cpp + Unsloth Q8_0 运行相同的短文本与 agent eval，记录结果作为外部行为基线。
- Unsloth 不作为 qwen3x 的权重来源、文件格式或运行时依赖，也不要求两个量化文件逐字节相同。
- qwen3x 结果异常时，用外部基线区分模型能力问题与 pack、render、state 或 CUDA kernel 问题。

### 容量与性能

- 记录 model weights、recurrent state、40960 KV cache、workspace、CUDA runtime 和峰值显存。
- `make serve-9b` 在 16 GiB GPU 上完成加载、40960 Session 创建、prefill 和 decode，无 OOM。
- 4K 实际 prompt 的 TTFT 不超过 10 秒，4K context 后 decode 不低于 20 tok/s。
- 记录 16K prompt 的 TTFT、prefill tok/s、decode tok/s 和峰值显存。
- 如果直接 Q8_0 × FP32 batch kernel 未达到门槛，保存 benchmark 和 profiler 证据并停止扩展；
  未经新的针对性讨论，不改成 activation quantization、完整 BF16 临时矩阵、CUTLASS 或通用
  算子框架。

### Agent

- 9B qwen3x 跑完固定 `inspect`、`review`、`bugfix` eval，保存 trace、usage、耗时和 verdict。
- 三个场景均通过且没有 tool-call parse error、incomplete tool call 或异常重试。
- pi 直接连接 `make serve-9b`，至少完成一次包含 read、write、bash 和测试的 coding task。

### 完成命令

```sh
make test
make cuda -j4
make -C tests render-test
make -C tests http-test
make -C tests pi-retry-test
```

9B reference、CUDA、性能和 agent eval 的具体命令与原始结果记录在 `eval/`。

## 本轮边界

本轮只实现 Qwen3.5-9B 的标准 weight-only Q8_0。Q4、Q6、Unsloth Dynamic、GGUF loader、
activation quantization、KV quantization、vision、MTP inference 和其他模型留给独立目标。

实现保持固定布局和直接数据流；不增加通用 tensor registry、ops abstraction、虚函数、
量化模板框架或运行时任意 dtype 组合。

## 当前状态

- [ ] 固定格式与 pack
- [ ] CPU correctness
- [ ] CUDA decode
- [ ] CUDA prefill
- [ ] runtime / HTTP / Makefile / README
- [ ] reference 与回归测试
- [ ] 16 GiB 容量与性能验收
- [ ] agent eval 与 pi 实测

全部验收通过后更新本节，记录最终 model bin 大小、峰值显存、4K/16K 性能、数值误差、agent
结果和已知边界。commit 和 push 由用户明确触发。
