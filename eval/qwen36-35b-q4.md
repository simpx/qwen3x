# Qwen3.6-35B-A3B Q4_0

## 固定输入

```text
repository: Qwen/Qwen3.6-35B-A3B
revision:   995ad96eacd98c81ed38be0c5b274b04031597b0
checkpoint: build/models/Qwen3.6-35B-A3B
model.bin:  build/qwen36-35b-a3b-q4_0-model.bin
```

官方 index 声明 1045 个 tensors、71,903,645,408 bytes 的 tensor payload；26 个
safetensors shard 的实际文件总和是 71,903,776,776 bytes。qwen3x 只选择 693 个 text
backbone tensors，选中 schema 的参数量是 34,660,610,688。tokenizer.json、vocab.json 和
merges.txt 与现有 Qwen3.5-0.8B render 来源逐字节相同：

```text
tokenizer.json 5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42
vocab.json     ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003
merges.txt     a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d
```

## 格式和实现

- model ID 是 36035，使用 `Q3XMODL\0`、固定 72-byte header 和 64-byte tensor alignment。
- 大矩阵是标准 Q4_0：32 values、FP16 signed scale、16 packed nibble bytes。
- router 和 shared expert gate 是 BF16；DeltaNet A_log/gated norm 是 FP32。
- 固定 revision 的 safetensors 实际把 A_log/gated norm 存成 BF16；packer 逐块扩展为上述
  model.bin FP32，不在内存中保留整 tensor 副本。
- CPU 逐 token forward，每层只执行 top-8 routed experts 和一个 shared expert。
- CUDA 一次分配并上传完整 model.bin，全部 routed expert pools 都由同一个 device allocation
  持有；没有 CPU offload、expert streaming、pinned staging 或 managed-memory fallback。
- Qwen3.6 prefill 首版逐 token 调用 decode forward，不使用 CUDA Graph。

## 验证命令

```sh
make test
make cuda -j4
make -C tests q4-cpu-test
make -C tests q4-cuda-test
make -C tests render-test
make -C tests http-test
make cpu-35b-smoke
make cuda-35b-smoke
```

另外使用本机固定的 llama.cpp b10516 `libggml-base.so` 直接调用
`quantize_row_q4_0_ref`，对 4096 个固定 seed BF16 blocks 比较全部 73,728 bytes；结果
byte-exact：

```sh
python3 tests/q4_llama_reference.py \
  --library build/llama.cpp-b10516/build-cuda/bin/libggml-base.so.0.20.2
```

固定输出 SHA-256 是 `735e40577b24f36e3ff4cc1a12bd085e7648b1b0099d0ae0b0700b371a64532c`。

这项检查调用真实 llama.cpp C reference，不是再次调用 qwen3x 的 Python scalar 实现。

`compute-sanitizer --tool memcheck build/q4-cuda-test` 已尝试；当前 WSL/WDDM 驱动返回
`Failed to initialize WDDM debugger interface` / `Device not supported`，因此不把它记作
sanitizer 通过。fixture 本身在同一次运行中通过，正常 CUDA 回归另行记录。

`tests/qwen36_smoke.py` 固定使用 raw `Hello`（token 9419），保存 prompt 位置与连续四个
greedy decode 位置的全部 248320 logits，并覆盖 checkpoint restore、live cache hit 和
reset/rebuild。CPU/CUDA elementwise tolerance 是 5e-4，同 backend 路径 tolerance 是 5e-5，
所有位置还必须有相同 argmax。

## 当前开发机

```text
WSL2
AMD Ryzen 7 9700X, 8C/16T
15 GiB RAM + 4 GiB swap
RTX 4080 SUPER, 16376 MiB, SM 8.9
CUDA toolkit 12.8
```

完整 model.bin 的单次 device allocation 是 19,528,534,784 bytes，大于该 GPU 的 16,376 MiB
物理显存；8192-context State/Work 还需要约 469,547,332 bytes。当前 WSL/WDDM CUDA 驱动仍
允许普通 `cudaMalloc` 成功并自行分页，因此完整 smoke 可以运行。qwen3x 没有 host expert
view、D2H top-k、per-token weight copy 或任何应用层 offload/fallback。

## 完整模型结果

下列数字是当前单一 device allocation 实现的实测结果：

```text
model bytes:            19,528,534,784 (18.1874 GiB)
model SHA-256:           3e1de932b9031bbbfceb2513fa76b2cd2e586043423ee48a8b666d77807c24fa
packed tensors:         693
pack wall / peak RSS:   4:03.35 / 99,560 KiB
CPU check wall:         31.03 s (inference checks 19.791 s)
CPU peak RSS:           3,772,876 KiB (3.598 GiB)
CPU prefill:            2.336 s
CPU decode:             1.874, 1.538, 1.520, 1.577 s
CPU greedy tokens:      [11, 353, 2688, 4313] -> ", I'm trying"
CPU all-position argmax:[11, 353, 2688, 4313, 310]
CUDA model allocation:  19,528,534,784 bytes, one cudaMalloc
CUDA state / work:      469,547,332 / 1,249,600 bytes (context 8192)
CUDA check wall:        19.01 s (inference checks 1.698 s)
CUDA host peak RSS:     13,130,304 KiB (driver-managed pages included)
CUDA prefill:           0.141 s
CUDA decode:            0.076, 0.067, 0.066, 0.069 s
CUDA physical peak:     15,859 MiB; delta from 1,506 MiB baseline is 14,353 MiB
CUDA greedy tokens:     [11, 353, 2688, 4313] -> ", I'm trying"
CUDA all-position argmax:[11, 353, 2688, 4313, 310]
CPU/CUDA max error:     2.765655517578125e-05 (limit 5e-4)
```

CPU oracle 的 `checkpoint restore`、live cache hit 和 `reset/rebuild` 路径内部最大误差是 0；
CUDA 对相同五个位置比较全部 248320 logits，每个位置的 argmax 和连续四个 greedy token
均相同。真实 CPU/CUDA CLI 同样输出 `, I'm trying`。CUDA Q4、router、routed/shared MoE 还由
小型 fixture 单独覆盖。

## 独立完整模型参考

使用 llama.cpp 源码 commit `b95502ba9aa0eb73a2f4fc8878d7fbe6a847a0b9`（运行时报告
`0.1.2-dev`, build 1），从同一官方 revision 独立转换 BF16 GGUF，再量化为 Q4_0：

```sh
python3 build/llama.cpp-b10516/convert_hf_to_gguf.py \
  build/models/Qwen3.6-35B-A3B \
  --outfile build/qwen36-35b-a3b-bf16.gguf --outtype bf16 --no-nextn

LD_LIBRARY_PATH=build/llama.cpp-b10516/build-cuda/bin \
  build/llama.cpp-b10516/build-cuda/bin/llama-quantize \
  --output-tensor-type Q4_0 --token-embedding-type Q4_0 \
  build/qwen36-35b-a3b-bf16.gguf \
  build/qwen36-35b-a3b-llama-q4_0.gguf Q4_0 16

LD_LIBRARY_PATH=build/llama.cpp-b10516/build-cuda/bin \
  build/llama.cpp-b10516/build-cuda/bin/llama-simple \
  -m build/qwen36-35b-a3b-llama-q4_0.gguf -n 4 -ngl 0 Hello
```

转换器识别为 `Qwen3_5MoeForConditionalGeneration` / `qwen35moe`，生成 733 个
GGUF tensors。Q4_0 GGUF 是 19,583,940,640 bytes；`llama-simple` 确认 raw `Hello` 是一个
prompt token，四步 greedy 输出为 `Hello, I'm trying`，与 qwen3x 的续写一致。GGUF 与
qwen3x model.bin 的 tensor 拆分和小 tensor 存储精度不完全相同，因此这项只验证 tokenizer、
prompt、top token 和短续写，不作 logits 逐值容差验收。

## 回归结果

2026-09-07 在上述开发机执行：

```text
make test -j4                                      passed
make cuda -j4 NVCC=/usr/local/cuda-12.8/bin/nvcc passed
make -C tests q4-cuda-test                         passed
make -C tests render-test                          passed, 9 tests
make -C tests http-test                            passed
make cuda-test                                     passed, 8 official 0.8B cases
python3 tests/q4_llama_reference.py ...            passed, byte-exact
make cpu-35b-smoke                                 passed
make cuda-35b-smoke                                passed, one device allocation
```

`make test -j4` 包含 Q4/Q8 CPU、runtime、parser、CLI、19 个 packer 合约测试和其他既有
回归。`make cuda-test` 的现有 Qwen3.5-0.8B 官方 reference 最大误差是
`1.72019005e-4`，低于 `5e-4` 限制。

## Mac handoff

`make cpu-35b-smoke` 同时生成 `build/qwen36-35b-metal-smoke/`：

```text
vectors.json   model SHA、命令所需 token、greedy token、误差契约和平台信息
logits.f32     五个位置的 little-endian FP32 全词表 logits
```

`logits.f32` 固定为 4,966,400 bytes，SHA-256 是
`25632e5fdbcf26d458c80aa823e2eee563632f31601fbdfd1247d623ec02efd1`。`vectors.json`
记录 model/library SHA、raw prompt token 9419、五个 argmax、四个 greedy token、context 128、
`atol=5e-4`、同 backend `path_atol=5e-5` 与完整生成命令。

M5 Metal 实现复用已有 dense Q4 embed/GEMV，需要新增 top-8 router、indexed expert 和
shared expert，但不得改变 model.bin 布局或 CPU oracle。当前 Metal loader 根据
`config.experts > 0` 返回明确的 `MoE not supported by Metal yet`。

## 已知限制

- 不包含 vision encoder 或 MTP。
- KV cache、recurrent state 和 activation 保持 FP32。
- CUDA 使用完整 device allocation，不支持应用层 CPU offload；超过物理显存能否运行取决于
  CUDA driver/platform，当前 WSL/WDDM 会自行分页。
- Qwen3.6 prefill 尚未做 grouped expert 或 chunk matrix 优化。
- Metal MoE 留给 M5 48GB 阶段实现；dense Q4 路径已经存在。
