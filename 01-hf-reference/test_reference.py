#!/usr/bin/env python3
"""Validate a Stage 1 vector bundle and rerun one official case as a reproducibility check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reference import VOCAB_SIZE, fingerprint, load_reference, run_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads((args.vectors / "vectors.json").read_text())
    if metadata.get("format") != "qwen3x-hf-vectors" or metadata.get("version") != 1:
        raise SystemExit("unsupported or malformed vector metadata")
    reference = load_reference(args.model, args.device)
    if metadata.get("device") != str(reference.device):
        raise SystemExit(
            f"vector device {metadata.get('device')!r} differs from requested official device "
            f"{str(reference.device)!r}; use separate CPU and CUDA vector bundles"
        )
    if metadata.get("fingerprint") != fingerprint(args.model, reference.tokenizer):
        raise SystemExit("checkpoint/tokenizer fingerprint differs from the vector bundle")
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

        # Rerun the smallest case.  Exact equality is expected in eager FP32 for the same local
        # checkpoint/device; a nonzero mismatch is useful evidence, not a threshold to hide.
        name = cases[0]["name"]
        actual = run_tokens(reference, vectors[f"{name}.input_ids"].tolist())
        expected = vectors[f"{name}.step_logits"]
        maximum = float(np.abs(actual.astype(np.float64) - expected.astype(np.float64)).max())
        print(f"rerun {name}: max_abs_error={maximum:.9g}")
        if maximum != 0.0:
            raise SystemExit("official FP32 vector is not reproducible on this checkpoint/device")
    print("Stage 1 official reference: passed")


if __name__ == "__main__":
    main()
