#!/usr/bin/env python3
"""用本目录固定 chat contract 验证文字 -> token -> CPU -> token -> 文字。"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path(__file__).with_name("reference.json"))
    parser.add_argument("--wrapper", type=Path, default=Path(__file__).with_name("chat.py"))
    return parser.parse_args()


def main():
    args = parse_args()
    reference = json.loads(args.reference.read_text())
    case = next(case for case in reference["cases"] if case["name"] == "official_chat")
    message = case["messages"][0]
    completed = subprocess.run(
        [sys.executable, str(args.wrapper), "--model", str(args.model),
         "--engine", str(args.engine), "--weights", str(args.weights),
         "--prompt", message["content"], "--max-new-tokens", str(len(case["greedy_ids"])),
         "--json"],
        check=True, text=True, capture_output=True,
    )
    result = json.loads(completed.stdout)
    if result.get("input_ids") != case["prefill_ids"]:
        raise SystemExit("chat template ids differ from the local official contract")
    if result.get("generated_ids") != case["greedy_ids"]:
        raise SystemExit("chat greedy ids differ from the local official contract")
    if not isinstance(result.get("text"), str) or not result["text"].strip():
        raise SystemExit("tokenizer decoded an empty response")
    print(f"chat input: {len(case['prefill_ids'])} ids match")
    print(f"chat output: {len(case['greedy_ids'])} ids match")
    print(f"decoded text: {result['text']!r}")
    print("self-contained text e2e: passed")


if __name__ == "__main__":
    main()
