#!/usr/bin/env python3
"""可选的重型验证：将本目录 CPU 路径与 Stage 1 官方 full-logits vectors 比较。"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


VOCAB_SIZE = 248320


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--case", help="run only one named vector case")
    parser.add_argument("--max-abs-error", type=float,
                        help="override the versioned CPU tolerance recorded in vectors.json")
    parser.add_argument("--skip-trace", action="store_true", help="do not compare step logits")
    parser.add_argument("--skip-greedy", action="store_true", help="do not compare greedy continuation")
    parser.add_argument("--skip-state", action="store_true", help="do not run prefill/decode equivalence")
    return parser.parse_args()


def ids(tokens: np.ndarray) -> str:
    return ",".join(str(int(token)) for token in tokens)


def trace(engine: Path, weights: Path, input_ids: np.ndarray) -> np.ndarray:
    """Ask one C++ process for a raw `[steps, vocab]` FP32 trace."""

    with tempfile.TemporaryDirectory(prefix="qwen35-cpu-trace-") as directory:
        output = Path(directory) / "logits.f32"
        subprocess.run(
            [str(engine), "--trace-logits", str(weights), ids(input_ids), str(output)], check=True
        )
        actual = np.fromfile(output, dtype=np.float32)
    expected_count = input_ids.size * VOCAB_SIZE
    if actual.size != expected_count:
        raise RuntimeError(f"C++ trace contains {actual.size} values, expected {expected_count}")
    return actual.reshape(input_ids.size, VOCAB_SIZE)


def generate(engine: Path, weights: Path, prefill_ids: np.ndarray, count: int) -> np.ndarray:
    completed = subprocess.run(
        [str(engine), "--generate", str(weights), ids(prefill_ids), str(count)],
        check=True, text=True, capture_output=True,
    )
    line = completed.stdout.strip()
    if not line.startswith("generated:"):
        raise RuntimeError(f"unexpected C++ generate output: {line!r}")
    words = line[len("generated:"):].split()
    return np.asarray([int(word) for word in words], dtype=np.int32)


def state_check(engine: Path, weights: Path, prefill_ids: np.ndarray, decode_ids: np.ndarray) -> None:
    completed = subprocess.run(
        [str(engine), "--state-check", str(weights), ids(prefill_ids), ids(decode_ids)],
        check=True, text=True, capture_output=True,
    )
    if not completed.stdout.startswith("state-check: passed"):
        raise RuntimeError(f"unexpected C++ state-check output: {completed.stdout!r}")


def compare(name: str, official: np.ndarray, actual: np.ndarray, limit: float) -> None:
    error = np.abs(actual.astype(np.float64) - official.astype(np.float64))
    maximum = float(error.max())
    mean = float(error.mean())
    # Reporting the first bad step makes a broken recurrent/cache update diagnosable without
    # pretending one final argmax has enough information.
    by_step = error.max(axis=1)
    first = int(np.argmax(by_step))
    print(f"{name}: max_abs={maximum:.6g}, mean_abs={mean:.6g}, worst_step={first} ({by_step[first]:.6g})")
    if maximum > limit:
        raise RuntimeError(f"{name}: max_abs_error {maximum:.6g} exceeds CPU limit {limit:.6g}")
    official_top = official.argmax(axis=1)
    actual_top = actual.argmax(axis=1)
    if not np.array_equal(official_top, actual_top):
        first = int(np.flatnonzero(official_top != actual_top)[0])
        raise RuntimeError(
            f"{name}: argmax differs at step {first}: C++={actual_top[first]}, official={official_top[first]}"
        )


def main() -> None:
    args = arguments()
    # `Path('./qwen35')` becomes the bare string `qwen35` when passed to subprocess, which
    # searches PATH instead of the current directory.  Resolve once so Make and direct use
    # behave the same way.
    args.engine = args.engine.resolve()
    args.weights = args.weights.resolve()
    args.vectors = args.vectors.resolve()
    if not args.engine.is_file() or not args.weights.is_file():
        raise SystemExit("engine or packed weights are missing")
    metadata = json.loads((args.vectors / "vectors.json").read_text())
    if metadata.get("format") != "qwen3x-hf-vectors" or metadata.get("vocab_size") != VOCAB_SIZE:
        raise SystemExit("unsupported Stage 1 vector bundle")
    if metadata.get("device") != "cpu":
        raise SystemExit("Stage 2 needs the CPU official vector bundle, not a CUDA bundle")
    vector_limit = metadata.get("comparison_tolerances", {}).get("cpu_max_abs_error")
    limit = args.max_abs_error if args.max_abs_error is not None else vector_limit
    if not isinstance(limit, (int, float)) or limit <= 0:
        raise SystemExit("vectors.json has no valid CPU max_abs_error contract")
    cases = metadata.get("cases", [])
    if args.case:
        cases = [case for case in cases if case["name"] == args.case]
        if not cases:
            raise SystemExit(f"no vector case named {args.case!r}")

    with np.load(args.vectors / "vectors.npz", allow_pickle=False) as vectors:
        for index, case in enumerate(cases):
            name = case["name"]
            inputs = vectors[f"{name}.input_ids"]
            official = vectors[f"{name}.step_logits"]
            if not args.skip_trace:
                actual = trace(args.engine, args.weights, inputs)
                compare(name, official, actual, limit)
            if not args.skip_greedy:
                greedy = generate(args.engine, args.weights, vectors[f"{name}.prefill_ids"],
                                  int(case["greedy_tokens"]))
                expected_greedy = vectors[f"{name}.greedy_ids"]
                if not np.array_equal(greedy, expected_greedy):
                    raise RuntimeError(
                        f"{name}: greedy ids differ: C++={greedy.tolist()} official={expected_greedy.tolist()}"
                    )
            # One state check is enough to verify the API itself; all cases already exercise
            # the same forward state through trace().  Avoid doubling every slow scalar CPU run.
            if index == 0 and not args.skip_state:
                state_check(args.engine, args.weights, vectors[f"{name}.prefill_ids"],
                            vectors[f"{name}.decode_ids"])
            print(f"{name}: selected checks passed")
    print("Stage 2 CPU reference: passed")


if __name__ == "__main__":
    main()
