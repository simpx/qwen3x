#!/usr/bin/env python3
"""Compare raw F32 trace directories emitted by qwen38 and Transformers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load(path: Path) -> np.ndarray:
    result = np.fromfile(path, dtype="<f4")
    if not result.size:
        raise ValueError(f"empty tensor: {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpp", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--atol", type=float, default=1.0e-3)
    parser.add_argument("--quiet", action="store_true", help="print only failures and the final summary")
    args = parser.parse_args()

    cpp_files = {path.name for path in args.cpp.glob("*.f32")}
    reference_files = {path.name for path in args.reference.glob("*.f32")}
    shared = sorted(cpp_files & reference_files)
    if not shared:
        parser.error("the directories share no .f32 tensors")
    missing_cpp = sorted(reference_files - cpp_files)
    missing_reference = sorted(cpp_files - reference_files)
    if missing_cpp:
        print("not emitted by C++:", ", ".join(missing_cpp))
    if missing_reference:
        print("not emitted by reference:", ", ".join(missing_reference))

    failures = 0
    worst_name = ""
    worst_value = -1.0
    for name in shared:
        cpp = load(args.cpp / name)
        reference = load(args.reference / name)
        if cpp.shape != reference.shape:
            print(f"{name}: shape mismatch {cpp.shape} != {reference.shape}")
            failures += 1
            continue
        diff = np.abs(cpp - reference)
        maximum = float(diff.max())
        rms = float(np.sqrt(np.mean(np.square(diff, dtype=np.float64))))
        location = int(diff.argmax())
        status = "ok" if maximum <= args.atol else "FAIL"
        if maximum > worst_value:
            worst_name = name
            worst_value = maximum
        if not args.quiet or status == "FAIL":
            print(f"{status:4s} {name:42s} max_abs={maximum:.7g} rms={rms:.7g} index={location}")
        failures += maximum > args.atol
    print(f"trace comparison: {len(shared)} shared tensors, worst={worst_name} max_abs={worst_value:.7g}, atol={args.atol:g}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
