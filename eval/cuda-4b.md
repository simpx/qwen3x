# Qwen3.5-4B CUDA 结果（2026-08-31）

## 配置

- checkpoint：官方 `Qwen/Qwen3.5-4B`，revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- model bin：BF16，7.83 GiB
- GPU：RTX 4080 SUPER 16 GiB
- CUDA：12.8，compute capability 8.9
- binary：`build/qwen35-cuda --model build/qwen35-4b-model.bin`
- Session：1 slot，40960 context

4B shape 记录在 model.bin v2 header，并由运行时 model ID 和固定 shape 校验。prefill
使用 cuBLAS BF16 Tensor Core GEMM（FP32 accumulation）；decode 保持 FP32 activation
和 CUDA Graph。DeltaNet 的 32 value heads 映射到 16 q/k heads，不能沿用 0.8B 中
`VH == KH` 时被掩盖的一一映射。

## 正确性

`make -C reference compare-cuda-4b` 对官方 mixed-FP32 oracle 的 8 个 case 全部通过。
fresh decode、chunk、checkpoint restore、Session cache 和四步 greedy continuation
一致；top-10 每步至少重叠 9 项。完整 logits 最大绝对误差为 0.361，出现在 BF16
special-token prefill；checkpoint restore 自一致误差不超过 5e-5。契约细节见
`reference/README.md`。

优化后的 qwen3x 固定 agent eval 原始结果位于本机忽略目录：
`eval/results/agent/stage4-20260831/qwen3x-4b-cublas-run1/`。inspect、review、bugfix
均通过，且没有工具协议错误。

## 4K benchmark

命令：

```sh
./build/qwen35-cuda --model build/qwen35-4b-model.bin \
  --bench 4096 32 --session-context 40960
```

优化前基线：prefill 32.476s（126.1 tok/s），decode 27.334 tok/s。

优化后三次完整独立运行：

| run | prefill | prefill tok/s | decode tok/s |
|---:|---:|---:|---:|
| 1 | 2.847s | 1438.755 | 56.377 |
| 2 | 2.818s | 1453.529 | 56.307 |
| 3 | 2.831s | 1446.910 | 56.367 |

因此 4K TTFT 低于 10 秒、decode 高于 30 tok/s 两个门槛均通过。40960 context
的真实 HTTP 服务也已在该 16 GiB GPU 上启动并供 pi 完成多轮工具任务。WSL 下
`torch.cuda.mem_get_info` 对另一个进程的全局驻留统计不可靠，所以不把它的约 598 MiB
变化误报为模型真实显存；容量结论只表述为该配置实际能够加载并运行。

长上下文单次记录（同一 binary、40960 Session context）：

| prompt + decode | prefill | prefill tok/s | decode tok/s |
|---|---:|---:|---:|
| 16384 + 32 | 33.649s | 486.903 | 35.743 |
| 32768 + 32 | 129.416s | 253.198 | 24.063 |

这说明 16K 后 decode 仍超过 30 tok/s，但 32K 已降至 24 tok/s，且 prefill 的 attention
成本明显增长。40960 是已验证容量，不应被描述成全范围都满足 4K 性能门；真实 pi
日常上下文应尽量依靠 Session prefix cache，并在接近 32K 前整理会话。

## 保留的实现

- 保留 cuBLAS BF16 batch GEMM：4K prefill 提速约 11.5 倍。
- 保留并行 online-softmax decode attention：4K decode 约翻倍。
- decode attention 必须保留 `1 / sqrt(head_dim)` scaling；reference regression 曾捕获
  一次遗漏，修复后才计入上述结果。
- 暂不引入量化：BF16 4B 已满足容量、TTFT 和 decode 门槛，量化会增加当前目标不需要的
  数据格式和 kernel 复杂度。
