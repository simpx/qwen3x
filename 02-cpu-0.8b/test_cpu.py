#!/usr/bin/env python3
"""只使用本目录 reference.json 验证 CPU forward、greedy 和 state，不依赖其他 stage。"""

import argparse
import json
import re
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path(__file__).with_name("reference.json"))
    parser.add_argument("--case")
    return parser.parse_args()


def token_text(tokens):
    return ",".join(str(token) for token in tokens)


def run(engine, *arguments):
    completed = subprocess.run(
        [str(engine), *map(str, arguments)], check=True, text=True, capture_output=True
    )
    return completed.stdout.strip()


def parse_generated(text):
    if not text.startswith("generated:"):
        raise RuntimeError(f"unexpected generate output: {text!r}")
    return [int(word) for word in text[len("generated:"):].split()]


def main():
    args = parse_args()
    engine = args.engine.resolve()
    weights = args.weights.resolve()
    if not engine.is_file() or not weights.is_file():
        raise SystemExit("engine or packed weights are missing")

    reference = json.loads(args.reference.read_text())
    if reference.get("format") != "qwen3x-cpu-smoke" or reference.get("version") != 1:
        raise SystemExit("unsupported local CPU reference")
    cases = reference["cases"]
    if args.case:
        cases = [case for case in cases if case["name"] == args.case]
        if not cases:
            raise SystemExit(f"no reference case named {args.case!r}")

    limit = float(reference["max_abs_error"])
    for case in cases:
        prompt = token_text(case["prefill_ids"])
        forward = run(engine, "--forward", weights, prompt)
        match = re.fullmatch(r"next token: (\d+), logit: ([-+0-9.eE]+)", forward)
        if not match:
            raise RuntimeError(f"unexpected forward output: {forward!r}")
        token, logit = int(match.group(1)), float(match.group(2))
        if token != case["next_token"] or abs(logit - case["next_logit"]) > limit:
            raise RuntimeError(
                f"{case['name']}: next result differs: token={token}, logit={logit}"
            )

        expected = case["greedy_ids"]
        generated = parse_generated(run(engine, "--generate", weights, prompt, len(expected)))
        if generated != expected:
            raise RuntimeError(
                f"{case['name']}: greedy differs: C++={generated}, expected={expected}"
            )

        state = run(
            engine, "--state-check", weights, prompt, token_text(case["decode_ids"])
        )
        if not state.startswith("state-check: passed"):
            raise RuntimeError(f"{case['name']}: unexpected state output: {state!r}")
        print(f"{case['name']}: forward, greedy, and state passed")
    print("self-contained CPU regression: passed")


if __name__ == "__main__":
    main()
