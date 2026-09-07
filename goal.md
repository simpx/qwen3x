# Goal：支持 Qwen3.6-35B-A3B 的 CPU 与单一 device allocation CUDA Q4_0

## 最终目标

在 qwen3x 中完整支持官方 `Qwen/Qwen3.6-35B-A3B` 的纯文本 backbone，直接从官方
BF16 safetensors 流式 pack 出一个固定布局的 Q4_0 model.bin，并在当前开发机完成 CPU/CUDA
全模型验收。CUDA 只使用一个完整 device allocation，不实现应用层 offload：

```text
同一个 Q4_0 model.bin
  -> CPU 完整 forward / Session / logits / generation
  -> CUDA 单一 device allocation / Session / logits / generation
  -> CPU 与 CUDA 逐 token 数值对齐
```

当前开发机是：

```text
Linux: WSL2
CPU:   AMD Ryzen 7 9700X，8C/16T
RAM:   15 GiB
GPU:   NVIDIA GeForce RTX 4080 SUPER，16376 MiB，SM 8.9
CUDA:  12.8
Disk:  /home 所在文件系统约 788 GiB 可用
```

最终产品目标是 M5 48GB unified memory。Metal MoE/Q4 适配不属于本轮实现；本轮必须留下
同一 model.bin、CPU oracle、测试向量和明确的 Mac 交接记录，使后续在 Mac 上只需增加
Metal Q4/MoE 计算，不再改变模型格式或 CPU 数学。

本目标由用户明确扩展了仓库当前只支持 Qwen3.5 dense 型号的范围。开始实现时先同步更新
`AGENTS.md` 的型号范围和模型抽象说明，但保留项目的总体原则：

```text
correct -> simple -> readable -> usable -> fast
```

实现过程中持续推进到“完成条件”全部满足；不要在写完计划、只通过合成测试、只完成 CPU、
只完成 CUDA kernel 或只成功 pack 后提前停止。遇到长时间下载、pack、编译或推理时继续等待
并检查结果，不因耗时较长自动缩小目标。

## 用户已经固定的决策

- 型号是官方 **Qwen3.6-35B-A3B**，不是旧的 Qwen3-30B-A3B。
- 本轮只实现 text backbone；vision encoder 和 MTP 不进入 model.bin，也不进入 forward。
- 新模型只使用标准 **Q4_0** 作为 weight-only 量化格式，不增加 Q8、Q5、Q6、K-quant、
  IQ、AWQ、GPTQ、activation quantization 或 KV quantization。
- CPU 和 CUDA 必须读取同一个 qwen3x model.bin，不经过 GGUF runtime。
- CPU 先成为 correctness baseline，再实现 CUDA。
- CUDA 使用最简单的单一 device allocation；物理驻留和分页交给 CUDA 驱动。
- 当前不实现 CPU offload、routed expert streaming、managed-memory oversubscription 或自动
  显存规划。这些属于未来单独设计和验收的高级功能。
- M5 上的 Metal 实现在本轮结束后由用户继续；本轮不实现 Metal MoE/Q4 kernel。
- 不重命名 `qwen3x` executable、C ABI 或现有 `q3x_` namespace。本轮用 model name/model ID
  区分 Qwen3.6。
- 不 commit、不 push；完成后保留可 review 的工作区 diff，提交由用户明确触发。

“只使用 Q4_0”表示所有被量化的 tensor 只采用 Q4_0，不表示破坏官方数值要求，把每个小
tensor 都强行压成四位。router 和下文列出的非矩阵参数保持 BF16/FP32；新模型中不出现 Q8。

## 固定上游

唯一权重来源：

```text
repo:     Qwen/Qwen3.6-35B-A3B
revision: 995ad96eacd98c81ed38be0c5b274b04031597b0
license:  Apache-2.0
```

下载目录固定为：

```text
build/models/Qwen3.6-35B-A3B
```

最终 qwen3x 权重文件固定为：

```text
build/qwen36-35b-a3b-q4_0-model.bin
```

继续复用现有 render 数据：

```text
build/qwen3x-render.bin
```

Qwen3.6 与当前 Qwen3.5 的 `vocab.json`、`merges.txt`、`tokenizer.json` SHA-256 已确认完全
一致。packer/test 应把这个事实变成显式校验；本轮不产生第二份 tokenizer/render bin。

## 固定模型结构

官方外层 checkpoint 是 `qwen3_5_moe` multimodal 模型，文本部分是
`qwen3_5_moe_text`。只实现 `model.language_model` 和顶层 `lm_head`。

model ID 固定使用：

```text
36035
```

`ModelConfig` 固定 shape：

```text
name                         Qwen3.6-35B-A3B
V                            248320
H                            2048
I / routed expert hidden     512
N                            40
attention interval           4
attention heads              16
KV heads                     2
attention head dim           256
rotary dim                   64
DeltaNet key heads           16
DeltaNet value heads         32
DeltaNet key dim             128
DeltaNet value dim           128
conv kernel                  4
experts                      256
experts per token            8
shared expert hidden         512
max context                  262144
tie_word_embeddings          false
matrix type                  Q4_0
```

40 层固定为：

```text
10 × {
  3 × (Gated DeltaNet -> MoE)
  1 × (Gated Attention -> MoE)
}
```

也就是 `layer % 4 != 3` 为 DeltaNet，`layer % 4 == 3` 为 full attention。Qwen3.6 的
DeltaNet/Attention shape 与现有 CUDA 固定常量 `AD=256, RD=64, KH=16, KD=128,
VD=128, CK=4` 一致；复用现有模型数学，不复制第二份 mixer forward。

text backbone 参数约为 `34,660,610,688`。vision 和 MTP tensors 即使存在于 shard/index
中也必须忽略，不能把它们误算为缺失或多余的 text tensor。

## Q4_0 固定格式

Q4_0 必须与 llama.cpp `quantize_row_q4_0_ref` 的 block 含义和 byte layout 一致：

```cpp
struct Block {
    uint16_t scale;
    uint8_t values[16];
};

static_assert(sizeof(Block) == 18, "Q4_0 block layout mismatch");
```

一个 block 表示同一行中连续的 32 个 BF16 权重。量化以 FP32 完成：

```text
max = block 中绝对值最大的元素，并保留该元素自身符号
d   = max / -8
id  = d != 0 ? 1 / d : 0
q   = min(15, int8(x * id + 8.5))
dequant(x_i) = (q_i - 8) * fp16(d)
```

布局固定为：

```text
values[j] low nibble  -> element j
values[j] high nibble -> element j + 16
j in [0, 15]
```

要求：

- scale 是 IEEE FP16 bit pattern，以 little-endian 写入 model.bin；scale 允许为负。
- 反量化使用实际落盘的 FP16 scale，不使用量化时未截断的 FP32 scale。
- block 不跨矩阵行；最后一维必须能被 32 整除。
- 二维和三维 expert tensor 都沿最后一维逐行量化。
- 全零 block 保存 `scale=0`，每个 nibble 为 8，也就是 16 个 `0x88` byte。
- NaN/Inf 输入必须让 packer 明确失败。
- rounding、signed maximum 和 nibble 顺序必须用固定向量逐字节对齐 llama.cpp reference。

增加独立 `q4.h`，只包含 block layout、block size 和最少的共享常量；不要把 `q8.h` 扩展成
通用量化框架，也不要引入第三方 GGML/llama.cpp 源码作为运行时依赖。

## Tensor 类型

以下大矩阵使用 Q4_0：

- token embedding 和独立 `lm_head`。
- DeltaNet：`in_proj_qkv`、`in_proj_z`、`in_proj_a`、`in_proj_b`、`out_proj`。
- Attention：`q_proj`、`k_proj`、`v_proj`、`o_proj`。
- Routed experts：`experts.gate_up_proj`、`experts.down_proj`。
- Shared expert：`gate_proj`、`up_proj`、`down_proj`。

以下保持 BF16：

- input/post/final RMSNorm。
- Attention q/k norm。
- DeltaNet conv1d 和 dt bias。
- MoE router `mlp.gate.weight`。
- `mlp.shared_expert_gate.weight`。

以下保持 FP32：

- DeltaNet `A_log`。
- DeltaNet gated norm weight。
- activation、accumulator、logits。
- DeltaNet recurrent/conv state。
- Attention KV cache。

router `[256, 2048]` 保持 BF16 是固定正确性决策：它只比 Q4 多约 28 MiB，却避免 Q4 误差
直接改变 top-8 expert selection。最终 text model.bin 预计约 `18.19 GiB`，实际 byte size 和
SHA-256 在完成后记录。

## model.bin 固定布局

- 使用 `Q3XMODL\0`、固定 72-byte header 和 64-byte tensor alignment。
- 不增加 tensor name table、runtime dtype table 或 GGUF metadata。
- model ID `36035` 唯一决定 Q4_0、untied lm_head、MoE shape 和 tensor 顺序。
- header 的 `INTERMEDIATE_SIZE` 对该型号写 `512`；experts/top-k/shared hidden 由固定
  `ModelConfig` 和 model ID 决定。
- 现有 0.8B/4B/9B model.bin 无需重打包；旧型号的行为和布局保持不变。
- 所有 loader 对 MatrixType 使用显式 `switch`；不能把未知类型或 Q4 默认为 Q8。

固定 tensor 顺序：

```text
model.language_model.embed_tokens.weight
lm_head.weight
model.language_model.norm.weight

for layer 0..39:
  input_layernorm.weight

  if DeltaNet:
    linear_attn.in_proj_qkv.weight
    linear_attn.in_proj_z.weight
    linear_attn.in_proj_a.weight
    linear_attn.in_proj_b.weight
    linear_attn.conv1d.weight
    linear_attn.A_log
    linear_attn.dt_bias
    linear_attn.norm.weight
    linear_attn.out_proj.weight
  else:
    self_attn.q_proj.weight
    self_attn.k_proj.weight
    self_attn.v_proj.weight
    self_attn.q_norm.weight
    self_attn.k_norm.weight
    self_attn.o_proj.weight

  post_attention_layernorm.weight
  mlp.gate.weight
  mlp.experts.gate_up_proj
  mlp.experts.down_proj
  mlp.shared_expert.gate_proj.weight
  mlp.shared_expert.up_proj.weight
  mlp.shared_expert.down_proj.weight
  mlp.shared_expert_gate.weight
```

完整 schema 应为 693 个 text tensors：

```text
prefix                 3
30 DeltaNet layers    30 × 18 = 540
10 Attention layers   10 × 15 = 150
total                 693
```

loader 必须严格检查 header、每个 tensor 的 shape/type/size、alignment 和最终 EOF。packer
必须严格检查官方 config、weight_map、dtype 和 shape；不因为 checkpoint 中还有 vision/MTP
权重而失败。

## MoE 数学

每层 post-attention norm 后执行：

```text
router_logits = BF16 router × FP32 x
router_probs  = softmax(router_logits, FP32, over all 256 experts)
ids, probs    = top-8(router_probs)
weights       = probs / sum(probs)

routed = 0
for slot in 0..7:
    expert = ids[slot]
    gate, up = expert_gate_up[expert] × x
    hidden = silu(gate) * up
    routed += weights[slot] * (expert_down[expert] × hidden)

shared_gate, shared_up = shared expert projections × x
shared_hidden = silu(shared_gate) * shared_up
shared = shared_down × shared_hidden
shared *= sigmoid(shared_expert_gate × x)

output = routed + shared
```

实现必须遵循官方顺序：先对全部 256 logits 做 FP32 softmax，再 top-k，再对选中概率重新归一化。
首版不要利用“只对 top-8 logits 做 softmax”的代数化简，以免改变 rounding。top-k 的测试数据
避免未定义的完全相等 tie；生产实现必须稳定、无越界、无重复 expert ID。

CPU 逐个执行 8 个 routed experts，复用一份 expert scratch。不要展开 256 份 activation，
不要把 expert 权重反量化为完整 FP32/BF16 副本。

## CPU 实现

### 数据结构

允许增加直接表达 MoE 的小结构，例如：

```cpp
struct ExpertLinear {
    const void* w;
    int experts;
    int rows;
    int cols;
    MatrixType type;
};

struct MoeWeights {
    const BF16* router;
    ExpertLinear gate_up;
    ExpertLinear down;
    Linear shared_gate;
    Linear shared_up;
    Linear shared_down;
    const BF16* shared_scale;
};
```

具体命名可以微调，但 ownership、shape 和 expert 偏移必须能在调用点看懂。`Layer` 同时容纳
现有 dense FFN 和 MoE；用 `experts > 0` 或明确的小 enum 选择结构，不使用继承、虚函数、
`std::variant`、tensor registry 或 ops graph。

### 算子

增加并集中放置：

```text
embed_q4
dot_q4
mv_q4
expert_mv_q4
router_top_k
moe
```

- Q4 kernel 直接读取 18-byte block、解 nibble、与 FP32 activation 相乘并 FP32 累加。
- 首版标量 CPU 实现即可；CPU 性能不是本轮完成门槛。
- `forward()` 继续只有一份完整 Qwen 主流程，layer loop 中只出现 mixer 类型和 FFN 类型分支。
- dense FFN 和 MoE 各自是具名函数；量化细节不传播到 runtime/session/render。
- CPU Model 继续 mmap model.bin，不把约 18.19 GiB 文件完整读入 15 GiB RAM。
- mmap/page cache 导致完整模型 smoke 较慢是允许的，但必须真正完成，不以 RAM 小于文件为由
  跳过 CPU full-model 验收。

## CUDA 实现

### Weight ownership

- Qwen3.5 和 Qwen3.6 CUDA loader 都采用一个完整 model.bin device allocation。
- Qwen3.6 loader 以一次 `cudaMalloc` 和一次 host-to-device copy 上传全部 Q4/BF16/FP32 权重。
- 所有 40 层 routed expert pools 与 non-expert weights 都通过 device pointer 访问。
- loader 完成后关闭并解除 host 文件映射；Model 只拥有 device weights。
- 完整 model.bin 是 18.1874 GiB，此外还需要 State、Work 和 CUDA runtime 空间。当前
  WSL/WDDM 驱动允许普通 `cudaMalloc` 超过物理显存并自行分页；qwen3x 不感知或调度分页。
- 不加入 CPU offload、expert streaming、pinned staging、managed memory 或显存自适应策略。

### CUDA Q4/MoE kernels

至少实现：

```text
embed_q4
mv_q4 / mv_add_q4
batch 或逐 token projection_q4
router_top_k
expert_gate_up_q4
expert_swiglu
expert_down_weighted_q4
shared_expert_q4
```

expert kernel 从常驻 pool 的 `[expert_id][row][block]` 直接计算基址，只保留一条数学路径。

`expert_gate_up_q4` 一次 dispatch 处理 top-8 × 1024 output rows；不要循环发射 8 个独立
matrix-vector kernel。`expert_down_weighted_q4` 以 H=2048 output rows 为并行维度，在同一输出
内部遍历 8 个 slot 并按 routing weight 累加，避免 atomic add。

### CUDA prefill

本轮以“完整跑通和数值对齐”为目标，不要求 Qwen3.6 的高性能 chunk/grouped-expert prefill。

- 多 token prefill 首版可以在 backend 内逐 token 调用同一 Q4 decode forward。
- 只在最后一个输入 token 计算 logits。
- `checkpoint_at` 边界、position、DeltaNet recurrence 和 attention KV 语义必须与现有 runtime
  完全一致。
- 不实现 expert grouping、FlashAttention、activation quantization、tensor core INT4 或
  speculative/MTP。

## Packer 与构建入口

### Packer

扩展 `scripts/pack_weights.py`，保持流式读取：

- 校验外层 `model_type=qwen3_5_moe`。
- 校验 `text_config.model_type=qwen3_5_moe_text` 和本目标全部固定 shape。
- 校验 `tie_word_embeddings=false`、BF16 dtype、262144 context。
- 只从 index 打开 693 个 text tensors 所需的 shard。
- rank-2/rank-3 Q4 tensor 按最后一维逐行分块，不把完整 shard/tensor 放入 RAM。
- router/shared gate 和其他 BF16/FP32 tensor原样流式复制。
- 使用临时输出，全部成功后原子 rename；失败时删除临时文件。
- 最后打印 tensor 数、实际 bytes、GiB、模型名、model ID 和 SHA-256。

### Makefile

增加直接入口，名称固定：

```sh
make model-35b
make cpu-35b-smoke
make cuda-35b-smoke
```

`make model-35b`：

- 固定使用上述 Hugging Face revision。
- 复用已经存在且完整的 checkpoint/shards，不重复下载。
- 缺失时下载 config、index、tokenizer 文件和全部必需 safetensors shards。
- pack 到 `build/qwen36-35b-a3b-q4_0-model.bin`。
- 确保现有 render bin 存在。

CPU full-model smoke 等价于：

```sh
./build/qwen3x \
  -m build/qwen36-35b-a3b-q4_0-model.bin \
  -r build/qwen3x-render.bin \
  -p "Hello" --session-context 128 --max-tokens 4
```

CUDA full-model smoke 等价于：

```sh
./build/qwen3x-cuda \
  -m build/qwen36-35b-a3b-q4_0-model.bin \
  -r build/qwen3x-render.bin \
  -p "Hello" --session-context 8192 --max-tokens 4
```

命令可以增加保存 logits、日志和输出目录的参数，但不能偷偷换小模型或不同量化文件。

## 测试计划

### Q4 格式测试

新增/扩展测试覆盖：

- `sizeof(Block)==18`、block=32、little-endian FP16 scale。
- 全零 block，预期 16 个 `0x88`。
- signed absolute maximum 为正/负的两个 case。
- 正负边界、halfway rounding、FP16 scale 截断。
- 随机 BF16 blocks 与独立 llama.cpp reference algorithm 逐 byte 一致。
- rank-2 和 rank-3 row boundary。
- 非 32 整除、NaN、Inf、截断输入明确失败。
- CPU `embed_q4/dot_q4/mv_q4/expert_mv_q4` 与 Python scalar reference 对齐。

增加清楚的测试 target，例如：

```sh
make -C tests q4-cpu-test
```

### MoE 测试

使用小型确定性 fixture 单独覆盖：

- 256-way router FP32 softmax。
- top-8 ID、顺序、概率重新归一化。
- combined gate/up tensor 的切分顺序。
- 8 routed expert weighted sum。
- shared expert 和 sigmoid gate。
- routed + shared + residual。
- reset/checkpoint 后 router 和 MoE 输出一致。

fixture 是测试数据，不作为新的可支持模型写入 `config_for_id()`。

### Packer/loader 测试

- 官方 config 精确匹配时选中 model ID 36035。
- 错误 outer/text model type、shape、dtype、tie、expert 数、top-k、context 被拒绝。
- 缺失任意一个 text tensor、错误 shape、错误 shard offset、截断 shard 被拒绝。
- vision/MTP tensor 被忽略。
- tensor count 固定为 693。
- loader 拒绝错误 magic/version/model ID/header、截断和尾部多余数据。
- CPU/CUDA 不会把 Q4 数据误读为 Q8；Metal 对 model ID 36035 必须明确返回
  `Q4_0/MoE not supported by Metal yet`，不能走当前 Q8 fallback。
- 现有 0.8B/4B/9B packer/loader 测试继续通过。

### 完整 CPU 验收

- mmap 官方完整 Q4 model.bin 成功。
- `Hello` raw prompt 至少生成 4 token。
- 保存 prompt 最后位置和连续 4 个 decode 位置的完整 248320 logits。
- fresh prefill、single-token decode、checkpoint save/restore、reset、Session cache 通过。
- 两次相同输入得到相同 logits/token。
- 记录运行时间和 peak RSS；不设 CPU tok/s 门槛。

CPU smoke/logits 输出固定保存在：

```text
build/qwen36-35b-cpu-smoke/
```

### 完整 CUDA 验收

- 当前 4080 SUPER 使用单一完整 `cudaMalloc` 创建 8192 context Session，不存在应用层
  host expert view 或 per-token weight copy。
- 同一个 `Hello` prompt 至少生成 4 token。
- 保存与 CPU 相同位置的完整 logits。
- CPU/CUDA 每个位置最大绝对误差 `<= 5e-4`。
- 每个位置 argmax 相同，4-step greedy token 完全一致。
- 覆盖 fresh prefill、decode、checkpoint restore、reset 和 cache reuse。
- CUDA memcheck/sanitizer 可用时检查小型 Q4/MoE fixture；完整 35B 不强制在 sanitizer 下运行。
- 记录 model/state/work bytes、峰值显存、prefill 和 decode 时间；不设性能门槛。

CUDA smoke/logits 输出固定保存在：

```text
build/qwen36-35b-cuda-smoke/
```

### 外部/官方参考

- Q4 block bytes 必须严格匹配 llama.cpp `quantize_row_q4_0_ref`。
- 优先使用同一官方 revision 转换的 llama.cpp Q4_0 GGUF 做独立完整模型短 prompt 对照。
- GGUF/llama.cpp 只用于测试，不成为 packer 中间格式、运行时依赖或分发产物。
- 如果完整 Transformers BF16 因 16 GiB RAM/VRAM 无法常驻，允许使用 disk offload 或只在
  M5/其他大内存机器补充 BF16 质量对比。Q4 CPU/CUDA 严格对齐同样留到显存足够的 CUDA
  机器执行，当前 16 GiB 开发机不以 offload 绕过容量要求。
- 外部实现与 qwen3x reduction 不同，不要求逐 byte logits 相同；必须检查 tokenizer、prompt、
  top tokens 和短 greedy continuation，并保存命令、revision 与结果。

### 既有回归

至少运行：

```sh
make test
make cuda -j4
make -C tests render-test
make -C tests http-test
make -C tests q4-cpu-test
make cpu-35b-smoke
make cuda-35b-smoke
```

如果测试 target 的实际名称略有调整，在 README/eval 记录最终命令。基础测试不能因完整模型
尚未下载而失败；full-model smoke target 可以明确提示先运行 `make model-35b`。

## 代码结构要求

优先修改现有数据流文件：

```text
model_config.h           Qwen3.6 config、MoE 固定 shape、Q4 matrix type
q4.h                     Q4_0 固定 block layout
scripts/pack_weights.py  官方 config/schema、rank-3 expert Q4 pack
engine.cpp               CPU Q4、MoE、完整 forward
arch/cuda/engine.cu      CUDA Q4、MoE、单一 device allocation
tests/                   Q4/MoE/packer/loader/runtime tests
reference/               必要的 Q4/full-model 对齐驱动
Makefile                 model/smoke targets
eval/qwen36-35b-q4.md    revision、命令、数值、容量、性能、Mac 交接
```

保持：

- 一个 CPU `forward()`。
- 一个 CUDA Qwen3.6 forward 数学顺序和一种常驻 expert weight view。
- runtime、Session、sampling、parser、HTTP 不认识 Q4 block 或显存容量策略。
- loader 仍按固定顺序切权重，不引入 tensor registry/name lookup。
- 量化分支集中在 embed/mv/expert weight consumer。
- MoE 分支集中在 layer FFN 位置。
- 权重不展开成 BF16/FP32 副本。
- exception-free C++17、无 RTTI、显式错误和资源所有权。

禁止：

- 引入 GGML、llama.cpp、CUTLASS、PyTorch C++ 或新的运行时依赖。
- 通用 quantization API、任意 dtype 混搭、算子对象层次、虚函数或宏生成 forward。
- 为追求性能复制第二份完整模型 forward。
- 在 WSL 使用不受支持的 managed-memory oversubscription。
- CPU offload、expert streaming 或自动显存规划。
- 为 CUDA 容量改用 Q3/IQ2 或量化 KV/state。
- 为本轮顺便实现 vision、MTP、Metal MoE、真正的 grouped prefill 或多 GPU。

当 `engine.cpp` 或 `arch/cuda/engine.cu` 因 MoE 已明显妨碍从完整 forward 阅读数据流时，才可
增加一个具名且内聚的 `moe` 实现文件；不要建立 `ops/` 目录或 backend framework。

## 文档与 Mac 交接

更新 README：

- 支持型号中加入 Qwen3.6-35B-A3B text-only。
- 写明 model revision、Q4_0、文件大小和下载/pack 命令。
- 写明 CPU mmap 的 RAM 特性。
- 写明 CUDA 单一 device allocation，以及当前 WSL/WDDM 的驱动分页行为。
- 写明 vision、MTP、Metal Q4/MoE 尚未实现。
- 写明 tokenizer/render 与现有 Qwen3.5 共用。

新增 `eval/qwen36-35b-q4.md`，至少记录：

- 官方 revision、config 和 model.bin SHA-256。
- 完整 tensor count、文件大小和 pack 时间。
- Q4 reference vectors。
- CPU 完整命令、greedy tokens，以及足够显存机器上的 CUDA 对齐命令。
- CPU RSS/耗时。
- CUDA device allocation bytes、驱动分页现象、峰值物理显存和耗时。
- 已知性能限制。
- Mac 后续需要增加的 `embed_q4`、`mv_q4`、top-8、indexed expert、shared expert 和 loader
  工作，以及可直接复制到 Mac 的 model/vector 路径。

生成可复制到 Mac 的 CPU smoke vectors，目录固定为：

```text
build/qwen36-35b-metal-smoke/
```

至少包含 model SHA-256、token IDs、每个位置完整 logits、argmax、容差契约和生成命令。Mac
后续可以直接用它验收 Metal，而不需要再次下载官方 72GB checkpoint。

## 本轮明确不做

- Qwen3.6 vision encoder、图片/视频输入。
- MTP/speculative decoding。
- Q8、Q5、Q6、Q4_K、IQ、AWQ、GPTQ。
- activation/KV/recurrent-state quantization。
- Metal Q4/MoE 实现或真实 Mac 测试。
- CUDA 高性能 grouped-expert prefill、FlashAttention 或 tensor parallel。
- 256K 性能验收。
- 重命名 executable/C ABI/namespace。
- agent eval、pi 产品能力评测；等 Metal 产品路径完成后另立目标。

## 完成条件

只有同时满足以下条件才算完成：

- [x] `AGENTS.md` 已明确允许 Qwen3.6-35B-A3B text-only 和 MoE。
- [x] 官方固定 revision 已下载且 packer 产出唯一 Q4_0 model.bin。
- [x] Q4_0 bytes 与 llama.cpp reference 固定向量逐 byte 一致。
- [x] packer/loader 对 693 个 text tensors、shape、dtype、alignment、EOF 有严格检查。
- [x] CPU 完整模型完成 prompt、4-token generation、logits、checkpoint/cache 测试。
- [x] CUDA 完成 Q4 mixer、router、routed/shared MoE 数学。
- [x] CUDA loader 使用单一完整 device allocation，不包含 CPU offload 或 expert streaming。
- [x] CUDA 在 4080 SUPER/WSL 驱动分页下创建 8192 context Session 并完成完整模型验收。
- [x] CPU/CUDA 固定位置 logits 最大绝对误差 `<= 5e-4`、argmax 与 greedy tokens 相同。
- [x] 现有 0.8B/4B/9B、render、runtime、HTTP 回归继续通过。
- [x] README 和 `eval/qwen36-35b-q4.md` 记录所有实际命令、revision、SHA-256、数值、容量、
      性能与限制。
- [x] `build/qwen36-35b-metal-smoke/` 足以让后续 Mac 直接验收 Metal。
- [x] Metal backend 对尚未支持的 36035 明确报错，绝不把 Q4 默认为 Q8。
- [x] 工作区只包含本目标相关改动，没有 commit/push。

完成时向用户报告：

```text
model.bin path / bytes / SHA-256
CPU smoke result / time / peak RSS
CUDA device allocation / physical VRAM / fixture result
all test commands and pass/fail
Mac handoff vector path
remaining known limitations
```

任何 performance 数字不达预期但 correctness 已通过时，先如实记录；本轮没有性能门槛，不能
擅自扩展到新的量化格式或复杂优化。只有模型无法正确生成、CPU/CUDA 不对齐、CUDA 数学
fixture 失败或既有回归失败，才属于未完成。

## 当前状态

- [x] 固定格式和 Q4_0 block
- [x] 官方 checkpoint 下载与 pack
- [x] CPU Q4/MoE correctness
- [x] CUDA 单一 device allocation Q4/MoE
- [x] 完整 CPU/CUDA 数值验收
- [x] 既有回归
- [x] 文档与 Mac handoff
