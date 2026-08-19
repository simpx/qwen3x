#!/usr/bin/env python3
"""Generate versioned official Qwen3.5-0.8B test vectors without downloading a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reference import VOCAB_SIZE, fingerprint, greedy_from_prefill, load_reference, run_tokens


# These token-id cases avoid coupling the core engine tests to chat-template text.  The final
# case below additionally exercises the official tokenizer so the outer wrapper is covered too.
FIXED_CASES = (
    ("lesson_short", [248044, 198, 198], [198, 198]),
    ("prefix_then_decode", [10, 42, 99, 7], [123, 456]),
)
MESSAGES = [{"role": "user", "content": "用一句话介绍 DeltaNet。"}]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-new-tokens", type=int, default=8)
    return parser.parse_args()


def add_case(arrays: dict[str, np.ndarray], metadata_cases: list[dict[str, object]], name: str,
             prefill: list[int], decode: list[int], reference, max_new_tokens: int,
             messages: list[dict[str, str]] | None = None) -> None:
    inputs = prefill + decode
    rows = run_tokens(reference, inputs)
    greedy = greedy_from_prefill(reference, prefill, max_new_tokens)
    arrays[f"{name}.input_ids"] = np.asarray(inputs, dtype=np.int32)
    arrays[f"{name}.prefill_ids"] = np.asarray(prefill, dtype=np.int32)
    arrays[f"{name}.decode_ids"] = np.asarray(decode, dtype=np.int32)
    arrays[f"{name}.step_logits"] = rows.astype(np.float32, copy=False)
    arrays[f"{name}.greedy_ids"] = np.asarray(greedy, dtype=np.int32)
    metadata_case: dict[str, object] = {
        "name": name,
        "prefill_tokens": len(prefill),
        "decode_tokens": len(decode),
        "greedy_tokens": len(greedy),
    }
    # Store the human-readable provenance alongside the ids.  Stage 4 can prove that its thin
    # tokenizer wrapper really creates this exact official-chat prefill, rather than merely
    # comparing two hand-copied integer lists.
    if messages is not None:
        metadata_case["messages"] = messages
    metadata_cases.append(metadata_case)
    print(f"dumped {name}: {len(inputs)} forward steps, {len(greedy)} greedy tokens")


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    reference = load_reference(args.model, args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    cases: list[dict[str, object]] = []
    for name, prefill, decode in FIXED_CASES:
        add_case(arrays, cases, name, prefill, decode, reference, args.max_new_tokens)

    chat_ids = reference.tokenizer.apply_chat_template(
        MESSAGES, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    add_case(arrays, cases, "official_chat", list(map(int, chat_ids)), [198, 198], reference,
             args.max_new_tokens, MESSAGES)

    np.savez(args.out / "vectors.npz", **arrays)
    metadata = {
        "format": "qwen3x-hf-vectors",
        "version": 1,
        "model": str(args.model),
        "device": str(reference.device),
        "dtype": "float32 official model",
        "logit_contract": "each row is logits after feeding the same-index input token",
        # This is the measured contract for Stage 2: BF16 checkpoint weights expanded to FP32
        # and scalar C++ accumulation versus PyTorch FP32 matmul.  It is deliberately much
        # tighter than a token-level check and is verified by the entire vector suite.
        "comparison_tolerances": {
            "cpu_max_abs_error": 5e-4,
            # Stage 3 keeps activations and cublas accumulation in FP32, just like Stage 2;
            # weights remain the checkpoint's BF16.  CUDA has a different reduction order,
            # so it owns a separate oracle bundle, but the correctness bound stays tight.
            "cuda_max_abs_error": 5e-4,
        },
        "vocab_size": VOCAB_SIZE,
        "fingerprint": fingerprint(args.model, reference.tokenizer),
        "cases": cases,
    }
    (args.out / "vectors.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {args.out / 'vectors.npz'} and vectors.json")


if __name__ == "__main__":
    main()
