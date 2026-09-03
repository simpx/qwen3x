#!/usr/bin/env python3
"""Smoke-test qwen3x logits against an external llama-debug executable."""

from __future__ import annotations

import argparse
from array import array
import heapq
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

PROMPTS = (
    "Hello",
    "The quick brown fox",
    "The quick brown fox jumps over the lazy dog. 计算 17 + 25。",
)
TOP_K = 10
MAX_ABS_ERROR = 0.50
MEAN_ABS_ERROR = 0.075
MIN_COSINE = 0.9995
LLAMA_TIMEOUT_SECONDS = 600
QWEN_TIMEOUT_SECONDS = 600


def read_array(path: Path, typecode: str) -> array:
    values = array(typecode)
    with path.open("rb") as source:
        values.fromfile(source, path.stat().st_size // values.itemsize)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def top_ids(values, count: int = TOP_K) -> list[int]:
    return heapq.nlargest(count, range(len(values)), key=values.__getitem__)


def compare_logits(actual, expected, case: str) -> dict[str, float | int]:
    if len(actual) != len(expected):
        raise AssertionError(
            f"{case}: qwen3x logits={len(actual)}, llama.cpp logits={len(expected)}"
        )

    maximum = 0.0
    total = 0.0
    actual_square = 0.0
    expected_square = 0.0
    product = 0.0
    for left, right in zip(actual, expected):
        difference = abs(left - right)
        maximum = max(maximum, difference)
        total += difference
        actual_square += left * left
        expected_square += right * right
        product += left * right

    mean = total / len(actual)
    cosine = product / math.sqrt(actual_square * expected_square)
    actual_argmax = max(range(len(actual)), key=actual.__getitem__)
    expected_argmax = max(range(len(expected)), key=expected.__getitem__)
    top_count = min(TOP_K, len(actual))
    actual_top = top_ids(actual, top_count)
    expected_top = top_ids(expected, top_count)
    overlap = len(set(actual_top) & set(expected_top))

    failures = []
    if maximum > MAX_ABS_ERROR:
        failures.append(f"max_abs_error={maximum:.9g} > {MAX_ABS_ERROR}")
    if mean > MEAN_ABS_ERROR:
        failures.append(f"mean_abs_error={mean:.9g} > {MEAN_ABS_ERROR}")
    if cosine < MIN_COSINE:
        failures.append(f"cosine={cosine:.9g} < {MIN_COSINE}")
    if actual_argmax != expected_argmax:
        failures.append(
            f"argmax qwen3x={actual_argmax} llama.cpp={expected_argmax}"
        )
    minimum_overlap = max(1, top_count - 1)
    if overlap < minimum_overlap:
        failures.append(
            f"top-{top_count} overlap={overlap}/{top_count} < {minimum_overlap}"
        )
    if failures:
        raise AssertionError(f"{case}: " + "; ".join(failures))

    return {
        "max_abs_error": maximum,
        "mean_abs_error": mean,
        "cosine": cosine,
        "argmax": actual_argmax,
        "top_overlap": overlap,
    }


def require_matching_tokens(actual, expected, case: str) -> None:
    for index, (left, right) in enumerate(zip(actual, expected)):
        if left != right:
            raise AssertionError(
                f"{case}: token {index} differs: qwen3x={left}, llama.cpp={right}"
            )
    if len(actual) != len(expected):
        raise AssertionError(
            f"{case}: token counts differ: "
            f"qwen3x={len(actual)}, llama.cpp={len(expected)}"
        )


def saved_output(output: Path, tool: str) -> tuple[array, array]:
    token_files = list(output.glob("*-tokens.bin"))
    logit_files = [
        path for path in output.glob("*.bin")
        if not path.name.endswith("-tokens.bin")
    ]
    if len(token_files) != 1 or len(logit_files) != 1:
        raise RuntimeError(
            f"{tool} wrote {len(token_files)} token and "
            f"{len(logit_files)} logit files in {output}"
        )
    if token_files[0].stat().st_size % 4 or logit_files[0].stat().st_size % 4:
        raise RuntimeError(f"{tool} output is not an array of 32-bit values")
    return read_array(token_files[0], "i"), read_array(logit_files[0], "f")


def llama_logits(executable: Path, gguf: Path, prompt: str,
                 output: Path) -> tuple[array, array]:
    command = [
        str(executable), "-m", str(gguf), "-p", prompt,
        "--save-logits", "--logits-output-dir", str(output),
        "-ngl", "99", "-c", "128", "-b", "128", "-ub", "128",
        "-fa", "on", "-ctk", "f32", "-ctv", "f32", "--no-warmup",
    ]
    try:
        subprocess.run(
            command, check=True, timeout=LLAMA_TIMEOUT_SECONDS,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"llama.cpp exceeded {LLAMA_TIMEOUT_SECONDS}s: {executable}"
        ) from error
    except subprocess.CalledProcessError as error:
        detail = (error.stdout or "").strip().splitlines()
        tail = "\n".join(detail[-10:])
        raise RuntimeError(
            f"llama-debug failed with status {error.returncode}"
            + (f"\n{tail}" if tail else "")
        ) from error

    return saved_output(output, "llama-debug")


def qwen_logits(executable: Path, model_bin: Path, render_bin: Path,
                prompt: str, output: Path) -> tuple[array, array]:
    command = [
        str(executable), "-m", str(model_bin), "-r", str(render_bin),
        "-p", prompt, "--save-logits", "--logits-output-dir", str(output),
        "--session-context", "128",
    ]
    try:
        subprocess.run(
            command, check=True, timeout=QWEN_TIMEOUT_SECONDS,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"qwen3x exceeded {QWEN_TIMEOUT_SECONDS}s: {executable}"
        ) from error
    except subprocess.CalledProcessError as error:
        detail = (error.stdout or "").strip().splitlines()
        tail = "\n".join(detail[-10:])
        raise RuntimeError(
            f"qwen3x failed with status {error.returncode}"
            + (f"\n{tail}" if tail else "")
        ) from error

    return saved_output(output, "qwen3x")


def require_file(path: Path, description: str, executable: bool = False) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"{description} not found: {path}")
    if executable and not (path.stat().st_mode & 0o111):
        raise RuntimeError(f"{description} is not executable: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-debug", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    args = parser.parse_args()

    started = time.monotonic()
    try:
        executable = require_file(args.llama_debug, "external llama-debug", True)
        gguf = require_file(args.gguf, "external Qwen3.5-9B Q8_0 GGUF")
        qwen = require_file(args.qwen, "qwen3x CUDA executable", True)
        model_bin = require_file(args.bin, "qwen3x Qwen3.5-9B Q8_0 model bin")
        render_bin = require_file(args.render, "qwen3x render bin")

        with tempfile.TemporaryDirectory(prefix="qwen3x-llama-smoke-") as temporary:
            root = Path(temporary)
            for index, prompt in enumerate(PROMPTS):
                llama_tokens, expected = llama_logits(
                    executable, gguf, prompt, root / f"{index}-llama"
                )
                qwen_tokens, actual = qwen_logits(
                    qwen, model_bin, render_bin, prompt, root / f"{index}-qwen"
                )
                if not llama_tokens:
                    raise RuntimeError(f"case {index}: llama.cpp produced no tokens")
                require_matching_tokens(qwen_tokens, llama_tokens, f"case {index}")
                result = compare_logits(actual, expected, f"case {index}")
                print(
                    f"llama smoke case={index} tokens={len(llama_tokens)} "
                    f"max_abs_error={result['max_abs_error']:.9g} "
                    f"mean_abs_error={result['mean_abs_error']:.9g} "
                    f"cosine={result['cosine']:.9g} "
                    f"argmax={result['argmax']} "
                    f"top10={result['top_overlap']}/10"
                )
    except (AssertionError, OSError, RuntimeError) as error:
        raise SystemExit(f"llama smoke failed: {error}") from error

    print(
        f"llama smoke passed cases={len(PROMPTS)} "
        f"elapsed={time.monotonic() - started:.3f}s"
    )


if __name__ == "__main__":
    main()
