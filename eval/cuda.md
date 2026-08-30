# CUDA 性能记录

本文记录 qwen3x 从 CPU correctness engine 开始实现 CUDA 的过程。所有数字均来自同一台
机器、同一份 `qwen35-0.8b-model.bin`，benchmark 只测 Session prefill/decode，不包含
tokenizer、HTTP 和采样。

## 环境与口径

- commit 起点：`399946bf859836a8312630f3143d388f7239db72`
- CPU：AMD Ryzen 7 9700X，8 核 16 线程
- GPU：NVIDIA GeForce RTX 4080 SUPER，16 GiB，compute capability 8.9
- CUDA：12.8，driver 591.86
- v1 短口径：`--bench 128 32 --session-context 160`
- v2 起正式口径：`--bench 512 128 --session-context 640`
- 流程：先预热一次，再记录三次；表中给出三次稳态平均值

## 结果

| 版本 | 口径 | Prefill tok/s | Decode tok/s | 说明 |
|---|---|---:|---:|---|
| CPU baseline | 128/32 | 41.447 | 28.999 | AVX-512 BF16 -> FP32 GEMV，单线程 |
| CUDA v1 FP32 weights | 128/32 | 180.450 | 127.628 | 逐 token，自写 FP32 GEMV |
| CPU baseline | 512/128 | 40.974 | 28.680 | 后续正式对照 |
| CUDA v2 BF16 weights | 512/128 | 165.417 | 122.201 | 逐 token |
| CUDA v3 Graph | 512/128 | 234.557 | 142.150 | BF16，CUDA Graph replay |
| CUDA v4 warp reduce | 512/128 | 274.227 | 170.105 | Graph + shuffle reduction |
| CUDA v5 grouped projection | 512/128 | 288.447 | 174.561 | 同源 projection 合并 |
| CUDA v6 projection + residual | 512/128 | 290.902 | 176.279 | projection 直接累加 hidden |
| CUDA v7 chunk prefill | 512/128 | 926.041 | 175.460 | 128-token chunk；decode 不变 |
| CUDA v7 chunk prefill | 4096/32 | 814.149 | 49.105 | 4K context 长序列检查 |

CPU 三次原始结果：

```text
prefill 128  3.083 s  41.517 tok/s   decode 32  1.118 s  28.619 tok/s
prefill 128  3.075 s  41.632 tok/s   decode 32  1.126 s  28.410 tok/s
prefill 128  3.107 s  41.192 tok/s   decode 32  1.068 s  29.967 tok/s
```

后续每轮 CUDA 改动都在这里追加实现方法、正确性结果、性能结果和保留/停止理由。

## CUDA v1：逐 token FP32 engine

第一版直接把 `engine.cpp` 的 Qwen 数据流翻译成具名 CUDA kernel。packed BF16 权重在加载时
展开为 FP32 并常驻显存；Session 的 recurrent state、KV cache、checkpoint 和 Work 也全部
常驻显存。`runtime.cpp` 只把 backend 调用从单 token 扩为 token range，并在
`checkpoint_at` 精确切段。

第一次逐步 logits 对齐在第二个 token 发现最大误差 1.026。原因是 DeltaNet kernel 先用
未衰减的 state 计算 `k^T S`，再衰减 state；正确顺序是先衰减再读取。单 token 冒烟和一句
completion 都没有暴露这个 recurrent 顺序错误，因此后续每轮均保留完整 step-logits 与
checkpoint 测试。

修正后 8 组官方 FP32 vectors 全部通过，覆盖 139 个逐步 logits、64-token prompt、greedy
续写和 checkpoint 恢复；最大绝对误差为 `0.000151634216`，阈值为 `0.0005`。稳态三次
原始性能：

```text
prefill 128  0.789 s  162.200 tok/s   decode 32  0.237 s  135.167 tok/s
prefill 128  0.672 s  190.417 tok/s   decode 32  0.250 s  128.119 tok/s
prefill 128  0.678 s  188.733 tok/s   decode 32  0.268 s  119.598 tok/s
```

## CUDA v2：BF16 权重常驻

v1 把 packed BF16 展开为 FP32，占用两倍显存，也让每次 GEMV 多读一倍权重数据。v2 保留
原始 BF16 权重，GEMV kernel 加载时直接转 FP32 累加。这个变化只涉及 `Linear`、权重上传
和读取点，不改变 forward 结构。

短 benchmark 的 GPU 升降频波动足以掩盖差异，因此从本轮起正式口径改为
`--bench 512 128 --session-context 640`。同口径 CPU 三次平均为 prefill `40.974`、decode
`28.680` tok/s。v2 三次平均为 prefill `165.417`、decode `122.201` tok/s。吞吐没有因
BF16 自动提升，但模型上传从约 1.31 秒降至约 0.40 秒，模型显存占用减半；因此保留。

## CUDA v3：Graph replay

Nsight Systems 对 64+16 token 的 v2 profile 记录了 32,274 次 `cudaLaunchKernel`，约
403 次/token；launch API 占 CUDA API 时间 49.6%。v3 保留完整的具名 kernel 和 top-down
forward，只把普通 forward 与带 lm_head 的 forward 各捕获为一张 CUDA Graph。token 与
position 放在 Session 的显存控制区，range 开始时一次上传，graph 最后显式递增 position。

8 组 reference 的最大误差仍为 `0.000151634216`。三次长口径结果非常稳定：prefill
`234.717/234.492/234.463`，decode `142.175/142.247/142.029` tok/s；相比 v2 分别提升
41.8% 和 16.3%。v3 profile 中普通 kernel launch 降为 810 次，80 个 token 由 80 次
`cudaGraphLaunch` 驱动。

## CUDA v4：warp reduction

GEMV 每个输出行原本用 shared memory 做 8 轮 block reduction。v4 先在 warp 内用 shuffle
求和，再归并 8 个 warp 部分和，把 block 同步减少为 2 次。该 helper 同时服务 GEMV、RMS
和 attention dot，不改变上层数据流。

8 组 reference 继续通过，最大误差 `0.000172019005`。三次长口径平均为 prefill
`274.227`、decode `170.105` tok/s，相比 v3 分别提升 16.9% 和 19.7%。Nsight Compute
硬件计数器被本机 NVIDIA 驱动以 `ERR_NVGPUCTRPERM` 拒绝，因此后续只能使用端到端数据与
Nsight Systems 的 API/launch 统计。

## CUDA v5：合并同源投影

DeltaNet 的 qkv/z/a/b、Attention 的 q/k/v、FFN 的 gate/up 都从同一个 normalized hidden
读取。v5 分别用三个具名 kernel 同时调度这些同源投影，权重、输出和 shape 仍在调用点
显式出现；预计每 token 减少 90 次 graph 内 kernel 调度。

首次实现把区间选择写成连续 `if`，导致 qkv/query 的高位 row 再次落入后续投影区间，首
token 最大误差达到 17.98。修正为基于原始 block index 的互斥区间后重新执行完整对齐。

修正后 reference 最大误差回到 `0.000172019005`。三次平均为 prefill `288.447`、decode
`174.561` tok/s，相比 v4 提升 5.2% 和 2.6%，收益开始递减。

## CUDA v6：projection + residual

mixer out projection 和 FFN down projection 的结果必然立即加回 hidden。v6 用明确的
`mv_add` 直接表达 `hidden += W @ input`，删除每层两次独立 residual kernel，共减少
48 次 graph 内调度。

8 组 reference 继续通过，最大误差 `0.000172019005`。三次平均为 prefill `290.902`、
decode `176.279` tok/s，只比 v5 提升 0.9% 和 1.0%。这说明单 token 路径的局部 launch
融合已经进入明显的边际收益区。

同版本的 4K 实测只有 prefill `92.400`、decode `49.281` tok/s，TTFT 达到 44.329 秒。
原因是逐 token causal attention 每个位置重新串行扫描整个前缀；短 benchmark 隐藏了这个
O(T²) 调度问题。

## CUDA v7：chunk prefill

v7 增加一个显式 `prefill_chunk()`，每次最多处理 128 个 token：

- RMS、projection、FFN 用二维 grid 并行 token 和输出行；
- DeltaNet 每个 head 在一个 kernel 内按 token 顺序更新 recurrent state；
- causal attention 为 chunk 内每个 query/head 启动独立 block，并只读取其合法前缀；
- decode 继续复用 v6 的单 token CUDA Graph。

这不是通用 Tensor/Graph 调度器，而是第二条从上到下展开的 Qwen prefill 数据流。
runtime 仍只在 checkpoint 边界切 token range，chunk 大小完全属于 CUDA backend。

完整 reference 在单 token eval、batch sync、append suffix 和 checkpoint restore 上继续通过，
最大误差保持 `0.000172019005`。三次 512/128 平均为 prefill `926.041`、decode
`175.460` tok/s；prefill 相比 v6 提升 3.18 倍，decode 基本不变。4K prompt 降至
5.031 秒、`814.149 tok/s`，随后 4K context 下 decode 为 `49.105 tok/s`，达到当前目标。

额外用 CPU/CUDA 两个 C ABI engine 对比 257-token 输入，令 `checkpoint_at=128`，同时跨越
两个 128-token chunk。fresh sync 与 restore-then-append 的完整 logits 最大差异都为
`1.52587890625e-05`，argmax 均为 198。单 slot、4096 context 的 listen 进程使整卡显存从
空闲时 2479 MiB 增至 4392 MiB，即该配置约占 1913 MiB（包含模型、Session 与 CUDA
context；WSL 未返回 per-process memory 行）。

## 停止点

当前下一档优化会是 tensor-core tiled GEMM/cuBLAS、FlashAttention 式分块与 online softmax、
按 GPU 型号 autotune chunk/tile，或量化 KV cache。这些都会引入新的矩阵布局、数值契约与
调度抽象，不再是保持当前代码结构的小优化；硬件计数器权限也尚未开放。按“优化直到开始
破坏结构和可读性”的约定，本轮停在 v7。
