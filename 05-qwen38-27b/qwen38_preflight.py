#!/usr/bin/env python3
"""Check the exact Qwen3.8-27B text contract before writing any 27B inference code.

The 0.8B teaching engine deliberately hard-codes its one model's dimensions so readers can see
the math.  Stage 5 must therefore begin with a factual gate, not pretend those constants will
magically scale.  This script reads only official `config.json` and the safetensors *index*;
it never downloads a weight shard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GIB = 1024 ** 3
EXPECTED_TOTAL_TENSORS = 1199
EXPECTED_TEXT_TENSORS = 851
EXPECTED_SHARDS = 18
# This is the official BF16 checkpoint's safetensors payload at the validated revision.  Keeping
# the byte count makes a silently substituted or partially downloaded checkpoint fail early.
EXPECTED_RAW_WEIGHT_BYTES = 55_562_855_904


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="official Qwen3.8-27B config.json")
    parser.add_argument("--index", type=Path, required=True, help="official model.safetensors.index.json")
    parser.add_argument("--device-gib", type=float, required=True, help="GPU memory available to this run")
    parser.add_argument(
        "--expect-raw-fits", choices=("yes", "no"),
        help="turn a known hardware-capacity answer into an executable test",
    )
    return parser.parse_args()


def need(mapping: dict[str, object], key: str, expected: object, where: str) -> None:
    """Fail loudly at the field that changed; no silent fallback to a nearby Qwen variant."""

    actual = mapping.get(key)
    if actual != expected:
        raise SystemExit(f"{where}.{key}: expected {expected!r}, found {actual!r}")


def gib(byte_count: int | float) -> float:
    return float(byte_count) / GIB


def text_tensor_names() -> set[str]:
    """Return exactly the tensors a text-only 27B packer must read, and nothing vision/MTP owns."""

    names = {
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        # Unlike 0.8B, 27B is explicitly untied: this is a distinct output matrix.
        "lm_head.weight",
    }
    for layer in range(64):
        prefix = f"model.language_model.layers.{layer}."
        names.add(prefix + "input_layernorm.weight")
        names.add(prefix + "post_attention_layernorm.weight")
        names.update(prefix + suffix for suffix in (
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"
        ))
        if layer % 4 != 3:
            names.update(prefix + "linear_attn." + suffix for suffix in (
                "in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight",
                "conv1d.weight", "A_log", "dt_bias", "norm.weight", "out_proj.weight",
            ))
        else:
            names.update(prefix + "self_attn." + suffix for suffix in (
                "q_proj.weight", "k_proj.weight", "v_proj.weight", "q_norm.weight",
                "k_norm.weight", "o_proj.weight",
            ))
    return names


def main() -> None:
    args = parse_args()
    if args.device_gib <= 0:
        raise SystemExit("--device-gib must be positive")
    config = json.loads(args.config.read_text())
    index = json.loads(args.index.read_text())
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise SystemExit("config has no text_config object")

    # These are the model-specific facts the future C++ Qwen3.8 snapshot must encode.  They
    # are intentionally explicit rather than a generic arbitrary-HF-config parser.
    need(config, "model_type", "qwen3_5", "config")
    need(config, "architectures", ["Qwen3_5ForConditionalGeneration"], "config")
    expected_text: dict[str, object] = {
        "model_type": "qwen3_5_text",
        "dtype": "bfloat16",
        "vocab_size": 248320,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4,
        "attn_output_gate": True,
        "output_gate_type": "swish",
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "mamba_ssm_dtype": "float32",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
    }
    for key, expected in expected_text.items():
        need(text, key, expected, "text_config")

    expected_layers = [kind for _ in range(16) for kind in (
        "linear_attention", "linear_attention", "linear_attention", "full_attention"
    )]
    need(text, "layer_types", expected_layers, "text_config")
    rope = text.get("rope_parameters")
    if not isinstance(rope, dict):
        raise SystemExit("text_config.rope_parameters is missing")
    for key, expected in {
        "rope_type": "default", "rope_theta": 10_000_000,
        "partial_rotary_factor": 0.25, "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
    }.items():
        need(rope, key, expected, "rope_parameters")

    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    if not isinstance(metadata, dict) or not isinstance(weight_map, dict):
        raise SystemExit("malformed safetensors index")
    raw_bytes = metadata.get("total_size")
    if not isinstance(raw_bytes, (int, float)) or raw_bytes <= 0:
        raise SystemExit("safetensors index has no positive metadata.total_size")
    raw_bytes = int(raw_bytes)
    shards = set(weight_map.values())
    if (len(weight_map), len(shards), raw_bytes) != (
        EXPECTED_TOTAL_TENSORS, EXPECTED_SHARDS, EXPECTED_RAW_WEIGHT_BYTES
    ):
        raise SystemExit(
            "unexpected Qwen3.8-27B safetensors layout: "
            f"{len(weight_map)} tensors, {len(shards)} shards, {raw_bytes} bytes"
        )
    actual_text_names = {
        name for name in weight_map
        if name == "lm_head.weight" or name.startswith("model.language_model.")
    }
    expected_text_names = text_tensor_names()
    if len(expected_text_names) != EXPECTED_TEXT_TENSORS:
        raise AssertionError("programmer error: Qwen3.8 text schema count changed without updating its contract")
    if actual_text_names != expected_text_names:
        missing = sorted(expected_text_names - actual_text_names)
        extra = sorted(actual_text_names - expected_text_names)
        raise SystemExit(
            "unexpected text-only tensor schema: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    # The terms below are exactly the arrays a direct batch=1 text-only implementation owns.
    # DQKV = small Q + small K + value.  Only the 48 linear-attention layers have S/conv;
    # only the 16 full-attention layers own growing KV cache.
    H, I, AH, KVH, AD = 5120, 17408, 24, 4, 256
    KH, VH, KD, VD, CK = 16, 48, 128, 128, 4
    layers, full_layers = 64, 16
    delta_layers = layers - full_layers
    d_qkv = 2 * KH * KD + VH * VD
    kv_width = KVH * AD
    recurrent_bytes = delta_layers * VH * KD * VD * 4
    conv_bytes = delta_layers * d_qkv * (CK - 1) * 4
    kv_bytes_per_token = full_layers * 2 * kv_width * 4
    kv_bytes_at_4096 = kv_bytes_per_token * 4096
    capacity_bytes = int(args.device_gib * GIB)
    raw_fits = raw_bytes <= capacity_bytes

    print("Qwen3.8-27B text contract: passed")
    print(f"layers: {layers} = {delta_layers} DeltaNet + {full_layers} full attention")
    print(f"widths: H={H}, I={I}, attention={AH}/{KVH} x {AD}, DeltaNet={KH}/{VH} x {KD}/{VD}")
    print(f"state: recurrent={gib(recurrent_bytes):.3f} GiB, conv={gib(conv_bytes):.3f} GiB, "
          f"attention KV={gib(kv_bytes_per_token):.6f} GiB/token ({gib(kv_bytes_at_4096):.3f} GiB at 4096)")
    print(f"official index: {len(weight_map)} all tensors / {len(actual_text_names)} text tensors, "
          f"{len(shards)} shards, raw weight bytes={gib(raw_bytes):.3f} GiB")
    print(f"device preflight: {args.device_gib:.3f} GiB, raw BF16 checkpoint fits={raw_fits}")
    print(f"warning: expanding all checkpoint values to FP32 would require about {gib(raw_bytes * 2):.3f} GiB "
          "before KV/state/work buffers")

    if args.expect_raw_fits is not None:
        expected_fits = args.expect_raw_fits == "yes"
        if raw_fits != expected_fits:
            raise SystemExit(
                f"raw BF16 fit expectation failed: expected {expected_fits}, observed {raw_fits}"
            )


if __name__ == "__main__":
    main()
