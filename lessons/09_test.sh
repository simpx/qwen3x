#!/usr/bin/env bash
# Run the teaching capstone on a local official Qwen3.5-0.8B checkpoint.
# The expected values were pinned from the official-weight verification.
#
# 这个脚本不是把 capstone 和另一套 runtime 互相比较：它固定一个短 prompt 的
# next-token/logit 和八步 greedy decode，防止教学代码被重构后悄悄偏离真实权重。
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 /path/to/Qwen3.5-0.8B" >&2
    exit 2
fi

model_dir=$1
# 转换产物约 1.4 GiB，因此放在 /tmp；trap 保证通过或失败都清理它及其临时文件。
course_bin=$(mktemp /tmp/qwen35-course-08b.XXXXXX.bin)
trap 'rm -f "$course_bin" "$course_bin.tmp"' EXIT

# 09_pack_weights.py 做 dtype/shape 检查后按第 09 课的读取顺序写出小格式权重。
python3 09_pack_weights.py "$model_dir" "$course_bin" >/dev/null
course_forward=$(./09_qwen35_0_8b --forward "$course_bin" 248044,198,198)

python3 - "$course_forward" <<'PY'
import re
import sys

# `--forward` 的输出是最后一个 prompt token 后的分布；只解析公开 CLI 文本，
# 因而检查不依赖 C++ 内部函数或额外 Python inference framework。
match = re.search(r"next token:\s*(\d+), logit:\s*([+-]?\d+(?:\.\d+)?)", sys.argv[1])
if not match:
    raise SystemExit("could not parse forward output")
if match.group(1) != "198":
    raise SystemExit(f"next-token mismatch: {match.group(1)} != 198")
if abs(float(match.group(2)) - 17.2760) > 1e-3:
    raise SystemExit(f"logit mismatch: {match.group(2)} != 17.2760")
PY

# 再验证 state 跨八次 decode 持续更新；仅比较 greedy token id，避免 tokenizer 变体。
course_generate=$(./09_qwen35_0_8b --generate "$course_bin" 248044,198,198 8)
expected_generate="generated: 198 198 198 198 198 198 198 198"
if [[ "$course_generate" != "$expected_generate" ]]; then
    echo "course:   $course_generate" >&2
    echo "expected: $expected_generate" >&2
    exit 1
fi

echo "course oracle: passed against pinned official-weight values ($course_forward)"
