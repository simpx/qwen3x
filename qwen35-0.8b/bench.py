#!/usr/bin/env python3
"""Small, repeatable CPU baseline for the native Qwen engine."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from qwen35 import Engine


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def synthetic_tokens(count: int) -> list[int]:
    # Benchmark compute, not tokenization. These are ordinary, valid token IDs.
    return [100 + i % 1000 for i in range(count)]


def seconds(function) -> float:
    started = time.perf_counter()
    function()
    return time.perf_counter() - started


def print_result(name: str, tokens: int, elapsed: float) -> None:
    print(
        f"{name:<8} {tokens:>4} tokens  "
        f"{elapsed:>8.3f} s  {tokens / elapsed:>7.3f} tok/s"
    )


def decode(session, count: int) -> None:
    for _ in range(count):
        session.eval(100)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the real native Engine without HTTP or tokenization."
    )
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--prefill-tokens", type=positive_int, default=8)
    parser.add_argument("--decode-tokens", type=positive_int, default=4)
    args = parser.parse_args()

    context_size = args.prefill_tokens + args.decode_tokens

    print("Qwen3.5-0.8B CPU baseline")
    print(f"library  {args.library.resolve()}")
    print(f"weights  {args.bin.resolve()}")
    print()

    with Engine(args.library, args.bin) as engine:
        # One full forward faults mmap'd weight pages in before timed measurements.
        warmup = engine.create_session(1)
        warmup_elapsed = seconds(lambda: warmup.sync([100]))
        warmup.close()

        session = engine.create_session(context_size)
        prompt = synthetic_tokens(args.prefill_tokens)
        prefill_elapsed = seconds(
            lambda: session.sync(prompt, checkpoint_at=len(prompt))
        )

        # Each eval appends one token and executes one complete decode forward.
        decode_elapsed = seconds(lambda: decode(session, args.decode_tokens))

    print_result("warmup", 1, warmup_elapsed)
    print_result("prefill", args.prefill_tokens, prefill_elapsed)
    print_result("decode", args.decode_tokens, decode_elapsed)
    print()
    print("Measured path: C++ Session sync/eval; tokenizer, HTTP and sampling excluded.")


if __name__ == "__main__":
    main()
