#!/usr/bin/env python3
"""A deliberately thin text/chat wrapper around a token-id Qwen3.5 binary.

The C++ teaching cores never parse UTF-8, apply a chat template, or decide a sampling policy.
They receive `--generate <weights> <comma-separated-token-ids> <count>` and return token ids.
Keeping that boundary here makes `forward(Model, State, token, Work)` readable while still
allowing an end-to-end real prompt test with the official tokenizer.

This is a development wrapper, not the eventual no-Python production CLI/server.  It uses the
checkpoint's tokenizer precisely so its input ids are a trustworthy contract for later work.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="local official checkpoint/tokenizer")
    parser.add_argument("--engine", type=Path, required=True, help="Stage 2 or Stage 3 token-id binary")
    parser.add_argument("--weights", type=Path, required=True, help="matching packed .bin weights")
    parser.add_argument("--prompt", required=True, help="one user message")
    parser.add_argument("--system", help="optional system message placed before the user message")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--json", action="store_true", help="print stable machine-readable result for tests")
    return parser.parse_args()


def make_messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    """Make the one intentionally supported chat shape, in obvious conversation order."""

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def encode_messages(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    """Delegate template details to the official tokenizer; C++ only sees the resulting ids."""

    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    return [int(token) for token in ids]


def generate_ids(engine: Path, weights: Path, input_ids: list[int], count: int) -> list[int]:
    """Call the tiny stable C++ token-id CLI and parse only its `generated:` line."""

    if count <= 0:
        raise ValueError("--max-new-tokens must be positive")
    completed = subprocess.run(
        [str(engine.resolve()), "--generate", str(weights.resolve()),
         ",".join(str(token) for token in input_ids), str(count)],
        check=True,
        text=True,
        capture_output=True,
    )
    line = completed.stdout.strip()
    if not line.startswith("generated:"):
        raise RuntimeError(f"unexpected engine output: {line!r}")
    words = line[len("generated:"):].split()
    return [int(word) for word in words]


def main() -> None:
    args = parse_args()
    if not args.model.is_dir() or not args.engine.is_file() or not args.weights.is_file():
        raise SystemExit("--model, --engine, or --weights is missing")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    messages = make_messages(args.prompt, args.system)
    input_ids = encode_messages(tokenizer, messages)
    output_ids = generate_ids(args.engine, args.weights, input_ids, args.max_new_tokens)
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    if args.json:
        print(json.dumps({"messages": messages, "input_ids": input_ids,
                          "generated_ids": output_ids, "text": text}, ensure_ascii=False))
    else:
        print(text)


if __name__ == "__main__":
    main()
