#!/usr/bin/env python3
"""Validate a vector bundle and visibly rerun one direct official model call."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


VOCAB_SIZE = 248320


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    metadata = json.loads((args.vectors / "vectors.json").read_text())
    if metadata.get("format") != "qwen3x-hf-vectors" or metadata.get("version") != 1:
        raise SystemExit("unsupported or malformed vector metadata")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but torch.cuda.is_available() is false")
    device = torch.device("cuda:0" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    # Again, no reference object: this is the official model call being verified.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval().to(device)
    fingerprint = {
        "config_sha256": sha256(args.model / "config.json"),
        "safetensors_index_sha256": sha256(args.model / "model.safetensors.index.json"),
        "tokenizer_sha256": sha256(args.model / "tokenizer.json"),
        "tokenizer_vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
        "torch_version": torch.__version__,
    }
    if metadata.get("device") != str(device):
        raise SystemExit(
            f"vector device {metadata.get('device')!r} differs from requested official device {str(device)!r}"
        )
    saved_fingerprint = metadata.get("fingerprint", {})
    for name, actual in fingerprint.items():
        if saved_fingerprint.get(name) != actual:
            raise SystemExit(
                f"checkpoint/tokenizer fingerprint differs at {name}: "
                f"saved={saved_fingerprint.get(name)!r}, actual={actual!r}"
            )
    if metadata.get("vocab_size") != VOCAB_SIZE:
        raise SystemExit("vector vocabulary does not match this fixed Qwen3.5 engine")

    with np.load(args.vectors / "vectors.npz", allow_pickle=False) as vectors:
        cases = metadata.get("cases", [])
        if not cases:
            raise SystemExit("vector bundle has no cases")
        for case in cases:
            name = case["name"]
            inputs = vectors[f"{name}.input_ids"]
            logits = vectors[f"{name}.step_logits"]
            greedy = vectors[f"{name}.greedy_ids"]
            if inputs.ndim != 1 or logits.shape != (inputs.size, VOCAB_SIZE):
                raise SystemExit(f"{name}: malformed input/logit shape {inputs.shape} / {logits.shape}")
            if greedy.ndim != 1 or not np.isfinite(logits).all():
                raise SystemExit(f"{name}: malformed greedy ids or non-finite logits")
            print(f"schema {name}: {inputs.size} token steps, {greedy.size} greedy tokens")

        # Rerun the smallest case directly, keeping the official cache after every token.
        cache = None
        rows: list[np.ndarray] = []
        with torch.inference_mode():
            for token in vectors[f"{cases[0]['name']}.input_ids"].tolist():
                output = model(
                    input_ids=torch.tensor([[int(token)]], device=device, dtype=torch.long),
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = output.past_key_values
                rows.append(output.logits[0, -1].float().cpu().numpy().copy())
        actual = np.asarray(rows, dtype=np.float32)
        expected = vectors[f"{cases[0]['name']}.step_logits"]
        maximum = float(np.abs(actual.astype(np.float64) - expected.astype(np.float64)).max())
        print(f"rerun {cases[0]['name']}: max_abs_error={maximum:.9g}")
        if maximum != 0.0:
            raise SystemExit("official FP32 vector is not reproducible on this checkpoint/device")
    print("official reference: passed")


if __name__ == "__main__":
    main()
