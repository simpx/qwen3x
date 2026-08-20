#!/usr/bin/env bash
# 在同一份官方权重上比较 CPU reference 与 CUDA backend。
#
# 这不是“CUDA 能启动”的 smoke test：它把两边 prefill 的 next-token/logit 和八步
# greedy decode 都放在一起检查。性能 CUDA 为 Tensor Cores 把每个 linear 的输入舍入为
# BF16，故允许约 5e-2 的 logit 差；CPU 第 09 课仍是 FP32-activation 的数值 reference。
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 /path/to/Qwen3.5-0.8B" >&2
    exit 2
fi

model_dir=$1
model_bin=$(mktemp /tmp/qwen35-cuda-oracle.XXXXXX.bin)
trap 'rm -f "$model_bin" "$model_bin.tmp"' EXIT

python3 ../00-lessons/09_pack_weights.py "$model_dir" "$model_bin" >/dev/null
cpu_forward=$(../00-lessons/09_qwen35_0_8b --forward "$model_bin" 248044,198,198)
cuda_forward=$(./qwen35_cuda --forward "$model_bin" 248044,198,198)

python3 - "$cpu_forward" "$cuda_forward" <<'PY'
import re
import sys

pattern = re.compile(r"next token:\s*(\d+), logit:\s*([+-]?\d+(?:\.\d+)?)")
cpu, cuda = (pattern.search(text) for text in sys.argv[1:])
if not cpu or not cuda:
    raise SystemExit("could not parse CPU/CUDA forward output")
if cpu.group(1) != "198":
    raise SystemExit(f"CPU reference no longer matches the official token: {cpu.group(1)}")
if abs(float(cpu.group(2)) - 17.276018) > 1e-3:
    raise SystemExit(f"CPU reference no longer matches the official logit: {cpu.group(2)}")
if cpu.group(1) != cuda.group(1):
    raise SystemExit(f"next-token mismatch: CPU {cpu.group(1)} != CUDA {cuda.group(1)}")
if abs(float(cpu.group(2)) - float(cuda.group(2))) > 0.1:
    raise SystemExit(f"logit mismatch: CPU {cpu.group(2)} != CUDA {cuda.group(2)}")
PY

cpu_generate=$(../00-lessons/09_qwen35_0_8b --generate "$model_bin" 248044,198,198 8)
cuda_generate=$(./qwen35_cuda --generate "$model_bin" 248044,198,198 8)
if [[ "$cpu_generate" != "$cuda_generate" ]]; then
    echo "CPU:  $cpu_generate" >&2
    echo "CUDA: $cuda_generate" >&2
    exit 1
fi

echo "cuda oracle: passed ($cpu_forward; $cuda_forward)"
