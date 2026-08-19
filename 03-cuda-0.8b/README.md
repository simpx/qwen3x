# Stage 3：plain CUDA + cuBLAS 0.8B

这是与 Stage 2 同一个固定 Qwen3.5-0.8B 的 CUDA snapshot。它不增加模型兼容性，也不建立
backend framework：权重格式、layer order、token-id CLI 与 `prefill/decode` 的语义全都固定。

同目录有两份源码，目的是让这个 stage 可独立阅读和编译：

```text
qwen35_cpu.cpp   # 固定 loader、Model/State/Work、scalar oracle helpers
qwen35_cuda.cu   # GPU weights/state、cuBLAS GEMV、直接 CUDA kernels、CUDA main
```

`qwen35_cuda.cu` 在编译时包含同目录的 CPU source，以重用**模型专用**的 mmap loader、weight
struct 和 parser；它不包含 `lessons/` 或别的 stage。没有 virtual backend、Tensor 或 dispatcher。

## 计算和 dtype

| 部分 | 实现 | dtype |
| --- | --- | --- |
| Linear / tied lm_head | `cublasSgemv` | checkpoint BF16 expanded once to FP32 × FP32 activation → FP32 output |
| embedding、RMSNorm、RoPE、residual、SiLU | 直接 CUDA kernels | FP32 activation |
| DeltaNet state、conv state、KV cache | GPU-resident arrays | FP32 |
| argmax | GPU kernel，只下载一个 id/logit | FP32 scores |

Stage 3 的正确性版本不在每个 GEMV 前把 activation 舍入 BF16，也不依赖 cuBLAS 尚不支持的
BF16-matrix/FP32-vector GEMV 组合。它在模型加载时把 linear matrices 由 checkpoint BF16 一次性
展开为 FP32；embedding/norm/conv 等非-linear 参数仍保留 BF16。这样 GPU path 直接对齐 CPU 的
“BF16 权重值 + FP32 activation”数学语义。以后 Tensor-Core BF16-activation 是有价值、但数值语义
不同的性能实验，应留给性能阶段及独立 regression。这里必须满足 Stage 1 的紧误差门槛，并逐 step
保持相同 argmax 与 greedy token ids。

## 编译与验证

```sh
make CUDA_ARCH=89
make cuda-test
make test MODEL=../models/Qwen3.5-0.8B
```

`CUDA_ARCH=89` 是这台 RTX 4080 的默认值；DGX Spark 应显式指定它实际支持的 compute capability。
普通 WSL 若存在旧的 distro `libcuda.so`，Makefile 会优先使用 `/usr/lib/wsl/lib`。

`make test` 依次生成独立的官方 CUDA HF vectors、验证 CUDA 每步 full logits、greedy continuation，以及
`prefill()+decode()` 与连续 `forward_cuda()` 的 GPU KV/GDN/conv state bitwise 一致性。

没有 CUDA driver 时，`make`/`make cuda-test` 必然失败；这是环境事实，不能将 Stage 3 标成已验证。
