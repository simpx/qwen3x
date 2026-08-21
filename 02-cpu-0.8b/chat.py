#!/usr/bin/env python3
"""文字/chat 薄入口：官方 tokenizer -> 本目录 C++ engine -> 官方 tokenizer decode。"""

import argparse
import json
import subprocess
from pathlib import Path

from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="官方 checkpoint/tokenizer 目录")
    parser.add_argument("--engine", type=Path, default=HERE / "qwen35")
    parser.add_argument("--weights", type=Path, default=HERE / "build/qwen35-0.8b.bin")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def messages(prompt, system):
    result = []
    if system:
        result.append({"role": "system", "content": system})
    result.append({"role": "user", "content": prompt})
    return result


def generate(engine, weights, input_ids, count):
    if count <= 0 or count > 4096:
        raise ValueError("--max-new-tokens must be in 1..4096")
    completed = subprocess.run(
        [str(engine.resolve()), "--generate", str(weights.resolve()),
         ",".join(str(token) for token in input_ids), str(count)],
        check=True, text=True, capture_output=True,
    )
    line = completed.stdout.strip()
    if not line.startswith("generated:"):
        raise RuntimeError(f"unexpected engine output: {line!r}")
    return [int(word) for word in line[len("generated:"):].split()]


def main():
    args = parse_args()
    if not args.model.is_dir() or not args.engine.is_file() or not args.weights.is_file():
        raise SystemExit("--model, --engine, or --weights is missing")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    conversation = messages(args.prompt, args.system)
    input_ids = tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    input_ids = [int(token) for token in input_ids]
    output_ids = generate(args.engine, args.weights, input_ids, args.max_new_tokens)
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    if args.json:
        print(json.dumps({"messages": conversation, "input_ids": input_ids,
                          "generated_ids": output_ids, "text": text}, ensure_ascii=False))
    else:
        print(text)


if __name__ == "__main__":
    main()
