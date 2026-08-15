#!/usr/bin/env python3
"""Compare qwen38's tokenizer adapter to the official local tokenizer."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

from transformers import AutoTokenizer


CASES = (
    "Hello",
    "你好，Qwen38!",
    "你好，hello <|im_end|>",
    " leading space\nnew line\tand emoji 😀",
)
CHAT_USER_TEXT = "请用一句话解释 DeltaNet。"


def cpp_ids(binary: Path, model: Path, text: str, command: str = "--tokenize") -> list[int]:
    output = subprocess.check_output((str(binary), command, str(model), text), text=True)
    match = re.search(r"^ids:(.*)$", output, re.MULTILINE)
    if not match:
        raise RuntimeError(f"qwen38 tokenizer output had no ID line:\n{output}")
    return [int(value) for value in match.group(1).split()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="CMake qwen38_08b binary with tokenizer support")
    parser.add_argument("model", type=Path, help="local Qwen3.5-0.8B checkpoint")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    for text in CASES:
        expected = tokenizer.encode(text, add_special_tokens=False)
        actual = cpp_ids(args.binary, args.model, text)
        if actual != expected:
            raise SystemExit(f"tokenizer mismatch for {text!r}:\n  C++: {actual}\n  HF:  {expected}")
        print(f"ok tokenizer {text!r}: {len(actual)} token(s)")

    expected_chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": CHAT_USER_TEXT}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(expected_chat, Mapping):
        expected_chat = expected_chat["input_ids"]
    actual_chat = cpp_ids(args.binary, args.model, CHAT_USER_TEXT, "--chat-tokenize")
    if actual_chat != expected_chat:
        raise SystemExit(f"one-turn chat template mismatch:\n  C++: {actual_chat}\n  HF:  {expected_chat}")
    print(f"ok one-turn chat template: {len(actual_chat)} token(s)")


if __name__ == "__main__":
    main()
