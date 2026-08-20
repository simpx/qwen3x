#!/usr/bin/env python3
"""Run the official Qwen3.5 checkpoint directly and save C++ test vectors.

There is deliberately no wrapper around Hugging Face here.  This file shows the
reference contract in its most literal form: load the official model, feed one
token, retain its cache, and save the logits after that token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


VOCAB_SIZE = 248320
STOP_TOKENS = {248044, 248046}

# These token-id cases avoid coupling core-engine tests to chat-template text.  The
# final case below additionally exercises the official tokenizer and chat template.
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


def sha256(path: Path) -> str:
    """Hash small provenance files; never hash/read the large weight shards here."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_case(arrays: dict[str, np.ndarray], metadata_cases: list[dict[str, object]], name: str,
             prefill: list[int], decode: list[int], model, device: torch.device,
             max_new_tokens: int, messages: list[dict[str, str]] | None = None) -> None:
    """Run one compact test case and put its plain arrays into the output bundle."""

    inputs = prefill + decode

    # This is the official incremental forward.  `cache` is exactly the state that
    # C++ calls `State`: attention KV plus DeltaNet recurrent/conv state.
    cache = None
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for token in inputs:
            output = model(
                input_ids=torch.tensor([[token]], device=device, dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            logits = output.logits[0, -1].float().cpu().numpy().copy()
            if logits.shape != (VOCAB_SIZE,):
                raise RuntimeError(f"official model returned logits shape {logits.shape}")
            rows.append(logits)

    # Repeat only the prompt, then let exactly the same official cache survive
    # greedy decode.  `model.generate()` is intentionally not involved.
    cache = None
    with torch.inference_mode():
        for token in prefill:
            output = model(
                input_ids=torch.tensor([[token]], device=device, dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
            )
            cache, logits = output.past_key_values, output.logits[0, -1]
        greedy: list[int] = []
        for _ in range(max_new_tokens):
            next_token = int(logits.argmax())
            if next_token in STOP_TOKENS:
                break
            greedy.append(next_token)
            output = model(
                input_ids=torch.tensor([[next_token]], device=device, dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
            )
            cache, logits = output.past_key_values, output.logits[0, -1]

    arrays[f"{name}.input_ids"] = np.asarray(inputs, dtype=np.int32)
    arrays[f"{name}.prefill_ids"] = np.asarray(prefill, dtype=np.int32)
    arrays[f"{name}.decode_ids"] = np.asarray(decode, dtype=np.int32)
    arrays[f"{name}.step_logits"] = np.asarray(rows, dtype=np.float32)
    arrays[f"{name}.greedy_ids"] = np.asarray(greedy, dtype=np.int32)
    case: dict[str, object] = {
        "name": name,
        "prefill_tokens": len(prefill),
        "decode_tokens": len(decode),
        "greedy_tokens": len(greedy),
    }
    if messages is not None:
        case["messages"] = messages
    metadata_cases.append(case)
    print(f"dumped {name}: {len(inputs)} forward steps, {len(greedy)} greedy tokens")


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if not args.model.is_dir():
        raise SystemExit(f"checkpoint directory does not exist: {args.model}")
    config = json.loads((args.model / "config.json").read_text())
    if config.get("text_config", {}).get("model_type") != "qwen3_5_text":
        raise SystemExit("this script only accepts the official Qwen3.5 text backbone")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but torch.cuda.is_available() is false")
    device = torch.device("cuda:0" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    # The two lines below are intentionally visible rather than hidden in a helper.
    # FP32 is the numerical gold standard for the scalar C++ reference.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval().to(device)

    args.out.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    cases: list[dict[str, object]] = []
    for name, prefill, decode in FIXED_CASES:
        add_case(arrays, cases, name, prefill, decode, model, device, args.max_new_tokens)

    chat_ids = tokenizer.apply_chat_template(MESSAGES, tokenize=True, add_generation_prompt=True, return_dict=False)
    add_case(arrays, cases, "official_chat", list(map(int, chat_ids)), [198, 198], model,
             device, args.max_new_tokens, MESSAGES)

    np.savez(args.out / "vectors.npz", **arrays)
    metadata = {
        "format": "qwen3x-hf-vectors",
        "version": 1,
        "model": str(args.model),
        "device": str(device),
        "dtype": "float32 official model",
        "logit_contract": "each row is logits after feeding the same-index input token",
        "comparison_tolerances": {"cpu_max_abs_error": 5e-4, "cuda_max_abs_error": 5e-4},
        "vocab_size": VOCAB_SIZE,
        "fingerprint": {
            "config_sha256": sha256(args.model / "config.json"),
            "safetensors_index_sha256": sha256(args.model / "model.safetensors.index.json"),
            "tokenizer_sha256": sha256(args.model / "tokenizer.json"),
            "tokenizer_vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
            "torch_version": torch.__version__,
        },
        "cases": cases,
    }
    (args.out / "vectors.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {args.out / 'vectors.npz'} and vectors.json")


if __name__ == "__main__":
    main()
