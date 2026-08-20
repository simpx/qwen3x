# Stage 5：Qwen3.8-27B preflight

这一阶段不能把 0.8B 的常量直接替换成更大的数字。`02-cpu-0.8b/` 与 `03-cuda-0.8b/` 是有意
固定模型的教材快照；Qwen3.8-27B 需要另写一份同样直接、但尺寸正确的 snapshot。

在开始写它之前，先用官方的两个小 metadata 文件锁定事实：`config.json` 和
`model.safetensors.index.json`。本目录**不会下载任意 safetensors shard**。

```sh
make test                 # 当前 16 GiB 开发 GPU：确认 27B raw BF16 不可能装下
make inspect DEVICE_GIB=128  # 只报告较大机器的容量，不假装已经能运行
```

当前官方 contract 是：64 layers = 48 Gated DeltaNet + 16 full-attention；`H=5120`、`I=17408`；
attention 为 24 query / 4 KV heads、head dim 256；DeltaNet 为 16 key / 48 value heads、各 dim 128。
它和 0.8B 一样是 3:1 hybrid，但不共享尺寸，也**不** tied `lm_head`。

## 为什么这台机器不能继续跑 27B

官方 index 报告的 BF16 weight payload 约 51.75 GiB；本机 RTX 4080 SUPER 只有 16 GiB。即使完全
不算 KV cache、GDN state、scratch 或 CUDA workspace，也不能加载。更重要的是，Stage 3 为 correctness
把 linear matrices 展开 FP32；直接复用那条路径约需 103.5 GiB weights，显然更不适用于 27B。

因此下一个真正的 CUDA implementation 必须在有足够显存/统一内存的目标机器上开发，并保持 BF16
权重，不得复制 Stage 3 的 FP32 matrix expansion。它仍然先做 BF16 correctness，再谈 quantization、
Tensor-Core fusion 或 TPS。

`qwen38_preflight.py` 对 config 的每个模型专用字段、3:1 layer pattern、1199 个总 tensor 中恰好
851 个 text-only tensor 的完整名字集合、safetensors shard 数以及 batch=1 state 内存都做检查。视觉
tower 和 MTP tensor 不会混进未来的 text-only packer。官方 metadata 改变时测试应失败，而不是无声地
对一个“差不多的 Qwen”继续工作。

`make test` 除了跑官方 metadata，还会有意把 `hidden_size`、一个 text tensor name 和 raw weight
byte count 改坏，确认 checker 会逐项拒绝它们。
