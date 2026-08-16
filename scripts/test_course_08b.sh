#!/usr/bin/env bash
# Compare the teaching capstone with the preserved full CPU reference on a
# local official Qwen3.5-0.8B checkpoint. No model is downloaded implicitly.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 /path/to/Qwen3.5-0.8B" >&2
    exit 2
fi

model_dir=$1
course_bin=$(mktemp /tmp/qwen38-course-08b.XXXXXX.bin)
trap 'rm -f "$course_bin" "$course_bin.tmp"' EXIT

python3 convert.py "$model_dir" "$course_bin" >/dev/null
course_forward=$(./qwen38_course --forward "$course_bin" 248044,198,198)
reference_forward=$(./qwen38_08b --forward "$model_dir" 248044,198,198 | tail -n 1)

python3 - "$course_forward" "$reference_forward" <<'PY'
import re
import sys

course, reference = sys.argv[1:]
pattern = re.compile(r"next token(?: id)?\s*:?\s*(\d+), logit\s*:?\s*([+-]?\d+(?:\.\d+)?)")
course_match = pattern.search(course)
reference_match = pattern.search(reference)
if not course_match or not reference_match:
    raise SystemExit("could not parse forward output")
if course_match.group(1) != reference_match.group(1):
    raise SystemExit(f"next-token mismatch: {course_match.group(1)} != {reference_match.group(1)}")
if abs(float(course_match.group(2)) - float(reference_match.group(2))) > 1e-3:
    raise SystemExit(f"logit mismatch: {course_match.group(2)} != {reference_match.group(2)}")
PY

course_generate=$(./qwen38_course --generate "$course_bin" 248044,198,198 8)
reference_generate=$(./qwen38_08b --generate "$model_dir" 248044,198,198 8 --temperature 0 | awk '/^generated:/ {line=$0} END {print line}')
if [[ "$course_generate" != "$reference_generate" ]]; then
    echo "course:    $course_generate" >&2
    echo "reference: $reference_generate" >&2
    exit 1
fi

echo "course oracle: passed ($course_forward)"
