#!/usr/bin/env python3
"""Dump the official Qwen3.5 forward trace for qwen38 development tests.

This script is deliberately development-only.  It loads the pinned Hugging
Face Transformers implementation and writes raw little-endian FP32 tensors
with the same names as ``qwen38_08b --trace``.  The C++ binary never imports
Python or Transformers at runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import Qwen3_5ForConditionalGeneration


def parse_ids(value: str) -> list[int]:
    try:
        result = [int(token) for token in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("token IDs must be comma-separated integers") from exc
    if not result or any(token < 0 for token in result):
        raise argparse.ArgumentTypeError("provide at least one non-negative token ID")
    return result


def write_f32(directory: Path, name: str, value: torch.Tensor) -> None:
    array = value.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False).reshape(-1)
    array.tofile(directory / f"{name}.f32")


def final_token(value: Any) -> torch.Tensor:
    """Extract [hidden] from a module result for our batch=1, token=1 calls."""
    if isinstance(value, tuple):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"hook produced unsupported value {type(value)!r}")
    return value.reshape(-1, value.shape[-1])[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="local Qwen/Qwen3.5-0.8B checkpoint directory")
    parser.add_argument("--ids", required=True, type=parse_ids, help="comma-separated token IDs")
    parser.add_argument("--out", required=True, type=Path, help="new or empty output directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"refusing to overwrite non-empty trace directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but PyTorch cannot see a CUDA device")

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    device = torch.device(args.device)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=dtype, local_files_only=True
    ).to(device).eval()
    text_model = model.model.language_model
    captures: dict[str, torch.Tensor] = {}
    hooks: list[Any] = []

    def capture(name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            captures[name] = final_token(output).detach().float().cpu()

        return hook

    def capture_input(name: str):
        """Capture a module's hidden-state input, after its preceding residual."""

        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            captures[name] = final_token(inputs[0]).detach().float().cpu()

        return hook

    hooks.append(text_model.embed_tokens.register_forward_hook(capture("embedding")))
    hooks.append(text_model.norm.register_forward_hook(capture("final_norm")))
    for index, layer in enumerate(text_model.layers):
        prefix = f"layers.{index}."
        hooks.append(layer.input_layernorm.register_forward_hook(capture(prefix + "input_norm")))
        mixer = layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn
        hooks.append(mixer.register_forward_hook(capture(prefix + "mixer")))
        # This norm consumes hidden + mixer, so its input is precisely the
        # first residual snapshot in the decoder layer.
        hooks.append(
            layer.post_attention_layernorm.register_forward_pre_hook(capture_input(prefix + "after_mixer_residual"))
        )
        hooks.append(layer.post_attention_layernorm.register_forward_hook(capture(prefix + "post_norm")))
        hooks.append(layer.mlp.register_forward_hook(capture(prefix + "mlp")))
        # The decoder layer returns the second residual snapshot.
        hooks.append(layer.register_forward_hook(capture(prefix + "after_mlp_residual")))

    past_key_values = None
    output = None
    try:
        with torch.inference_mode():
            for token in args.ids:
                captures.clear()
                output = model(
                    input_ids=torch.tensor([[token]], dtype=torch.long, device=device),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = output.past_key_values
    finally:
        for hook in hooks:
            hook.remove()

    assert output is not None
    required = {"embedding", "final_norm"}
    for index in range(len(text_model.layers)):
        prefix = f"layers.{index}."
        required.update(
            (
                prefix + "input_norm",
                prefix + "mixer",
                prefix + "after_mixer_residual",
                prefix + "post_norm",
                prefix + "mlp",
                prefix + "after_mlp_residual",
            )
        )
    missing = required.difference(captures)
    if missing:
        raise RuntimeError(f"missing trace hooks: {sorted(missing)}")

    for name in sorted(captures):
        write_f32(args.out, name, captures[name])
    write_f32(args.out, "logits", output.logits[0, -1])
    (args.out / "manifest.txt").write_text(
        "qwen38-reference-trace-v1\n"
        f"model={args.model}\nids={','.join(map(str, args.ids))}\n"
        f"device={args.device}\ndtype={args.dtype}\n",
        encoding="utf-8",
    )
    print(f"trace: {len(args.ids)} token(s), {len(captures) + 1} tensors written to {args.out}")
    print(f"next token: {int(output.logits[0, -1].argmax())}, logit {float(output.logits[0, -1].max()):.6f}")


if __name__ == "__main__":
    main()
