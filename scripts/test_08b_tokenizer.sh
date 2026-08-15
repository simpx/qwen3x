#!/usr/bin/env bash
# Check the small C++ tokenizer adapter against Hugging Face's local tokenizer.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model=${1:-"$root/models/Qwen3.5-0.8B"}

if [[ ! -f "$model/tokenizer.json" ]]; then
    echo "qwen38: expected tokenizer.json under: $model" >&2
    exit 2
fi
if [[ ! -d "$root/reference/.deps/transformers" ]]; then
    echo "qwen38: install the development oracle first:" >&2
    echo "  python3 -m pip install --target reference/.deps -r reference/requirements.txt" >&2
    exit 2
fi

cmake -S "$root" -B "$root/build"
cmake --build "$root/build" --target qwen38_08b -j2
PYTHONPATH="$root/reference/.deps" python3 "$root/reference/test_tokenizer.py" "$root/build/qwen38_08b" "$model"
