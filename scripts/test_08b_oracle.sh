#!/usr/bin/env bash
# Run the fixed Qwen3.5-0.8B CPU oracle against the pinned Transformers trace.
#
# Development-only: the model checkpoint and reference/.deps are intentionally
# ignored from git.  The release binary has neither dependency.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model=${1:-"$root/models/Qwen3.5-0.8B"}
ids=${QWEN38_TEST_IDS:-248044,198,198}
cpp_trace=$(mktemp -d "${TMPDIR:-/tmp}/qwen38-cpp-trace-XXXXXX")
reference_trace=$(mktemp -d "${TMPDIR:-/tmp}/qwen38-reference-trace-XXXXXX")

if [[ ! -f "$model/config.json" || ! -f "$model/model.safetensors-00001-of-00001.safetensors" ]]; then
    echo "qwen38: expected a local Qwen3.5-0.8B checkpoint at: $model" >&2
    exit 2
fi
if [[ ! -d "$root/reference/.deps/transformers" ]]; then
    echo "qwen38: install the development oracle first:" >&2
    echo "  python3 -m pip install --target reference/.deps -r reference/requirements.txt" >&2
    exit 2
fi

make -C "$root" qwen38_08b
"$root/qwen38_08b" --inspect "$model"
"$root/qwen38_08b" --trace "$model" "$ids" "$cpp_trace"
PYTHONPATH="$root/reference/.deps" python3 "$root/reference/dump_qwen35_trace.py" "$model" \
    --ids "$ids" --out "$reference_trace" --dtype fp32
PYTHONPATH="$root/reference/.deps" python3 "$root/reference/compare_traces.py" \
    "$cpp_trace" "$reference_trace" --atol 0.001 --quiet

echo "qwen38: Qwen3.5-0.8B CPU oracle passed for token IDs: $ids"
