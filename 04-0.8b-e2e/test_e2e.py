#!/usr/bin/env python3
"""Test a real chat prompt through official tokenizer -> C++ token-id engine -> official decoder."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, default=Path(__file__).with_name("qwen35_chat.py"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads((args.vectors / "vectors.json").read_text())
    chat_case = next((case for case in metadata.get("cases", []) if case.get("name") == "official_chat"), None)
    if chat_case is None or not isinstance(chat_case.get("messages"), list):
        raise SystemExit("vectors do not contain an official_chat messages contract; regenerate Stage 1 vectors")
    messages = chat_case["messages"]
    if len(messages) != 1 or not isinstance(messages[0], dict):
        raise SystemExit("this deliberately small wrapper test currently expects exactly one user message")
    message = messages[0]
    if set(message) != {"role", "content"} or message.get("role") != "user" or not isinstance(message.get("content"), str):
        raise SystemExit("this deliberately small wrapper test currently expects exactly one user message")

    with np.load(args.vectors / "vectors.npz", allow_pickle=False) as vectors:
        expected_input = vectors["official_chat.prefill_ids"].astype(np.int32).tolist()
        expected_output = vectors["official_chat.greedy_ids"].astype(np.int32).tolist()

    completed = subprocess.run(
        [sys.executable, str(args.wrapper), "--model", str(args.model), "--engine", str(args.engine),
         "--weights", str(args.weights), "--prompt", message["content"],
         "--max-new-tokens", str(len(expected_output)), "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    result = json.loads(completed.stdout)
    if result.get("input_ids") != expected_input:
        raise SystemExit("chat template ids differ from the Stage 1 official-chat contract")
    if result.get("generated_ids") != expected_output:
        raise SystemExit(
            f"greedy ids differ: wrapper={result.get('generated_ids')} official={expected_output}"
        )
    if not isinstance(result.get("text"), str) or not result["text"].strip():
        raise SystemExit("official tokenizer decoded an empty text response")
    print(f"tokenizer contract: {len(expected_input)} input ids match official_chat")
    print(f"greedy contract: {len(expected_output)} generated ids match official reference")
    print(f"decoded smoke: {result['text']!r}")
    print("Stage 4 end-to-end: passed")


if __name__ == "__main__":
    main()
