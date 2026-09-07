#!/usr/bin/env python3
"""Compare qwen3x BF16->Q4_0 bytes with llama.cpp's C reference."""

import argparse
import ctypes
import hashlib
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pack_weights import quantize_q4_0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=4096)
    args = parser.parse_args()
    if args.blocks < 3:
        parser.error("--blocks must be at least 3")

    random = np.random.default_rng(20260907)
    values = random.normal(size=args.blocks * 32).astype("<f4")
    values[:32] = 0
    values[32:64] = np.linspace(-8, 7.5, 32, dtype="<f4")
    values[64:96] = np.linspace(8, -7.5, 32, dtype="<f4")
    bf16 = (values.view("<u4") >> 16).astype("<u2")
    values = (bf16.astype("<u4") << 16).view("<f4")

    library = ctypes.CDLL(str(args.library.resolve()))
    reference = library.quantize_row_q4_0_ref
    reference.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    reference.restype = None
    expected = ctypes.create_string_buffer(args.blocks * 18)
    reference(values.ctypes.data, expected, values.size)
    expected_bytes = expected.raw
    actual = quantize_q4_0(bf16.tobytes())
    if actual != expected_bytes:
        first = next(i for i, (left, right) in
                     enumerate(zip(actual, expected_bytes)) if left != right)
        raise SystemExit(
            f"Q4_0 mismatch byte={first} qwen3x={actual[first]} "
            f"llama.cpp={expected_bytes[first]}"
        )
    print(f"q4-llama-reference: ok blocks={args.blocks} bytes={len(actual)} "
          f"sha256={hashlib.sha256(actual).hexdigest()}")


if __name__ == "__main__":
    main()
