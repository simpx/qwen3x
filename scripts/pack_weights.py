#!/usr/bin/env python3
"""Pack an official Qwen3.5-0.8B or Qwen3.5-4B text backbone.

The output is a deliberately small sequential format: a fixed metadata header
followed by tensors in the exact order consumed by engine.cpp. The 4B
checkpoint spans multiple safetensors shards, so the packer keeps shard headers
open and streams tensors without loading the model into RAM.
"""

import argparse
from contextlib import ExitStack
import json
import struct
from pathlib import Path

ALIGNMENT = 64
MAGIC = b"Q35MODL\0"
FORMAT_VERSION = 2
HEADER = struct.Struct("<8sII16I")
MAX_CONTEXT = 262144

SUPPORTED_MODELS = (
    {
        "name": "Qwen3.5-0.8B", "model_id": 800,
        "vocab_size": 248320, "hidden_size": 1024,
        "intermediate_size": 3584, "num_hidden_layers": 24,
        "full_attention_interval": 4, "num_attention_heads": 8,
        "num_key_value_heads": 2, "head_dim": 256, "rotary_dim": 64,
        "linear_num_key_heads": 16, "linear_num_value_heads": 16,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    },
    {
        "name": "Qwen3.5-4B", "model_id": 4000,
        "vocab_size": 248320, "hidden_size": 2560,
        "intermediate_size": 9216, "num_hidden_layers": 32,
        "full_attention_interval": 4, "num_attention_heads": 16,
        "num_key_value_heads": 4, "head_dim": 256, "rotary_dim": 64,
        "linear_num_key_heads": 16, "linear_num_value_heads": 32,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    },
)


def select_model(text_config):
    """Return the one supported shape that exactly matches structural fields."""
    rope = text_config.get("rope_parameters", {})
    for model in SUPPORTED_MODELS:
        contract = {
            "model_type": "qwen3_5_text",
            "vocab_size": model["vocab_size"],
            "hidden_size": model["hidden_size"],
            "intermediate_size": model["intermediate_size"],
            "num_hidden_layers": model["num_hidden_layers"],
            "full_attention_interval": model["full_attention_interval"],
            "num_attention_heads": model["num_attention_heads"],
            "num_key_value_heads": model["num_key_value_heads"],
            "head_dim": model["head_dim"],
            "linear_num_key_heads": model["linear_num_key_heads"],
            "linear_num_value_heads": model["linear_num_value_heads"],
            "linear_key_head_dim": model["linear_key_head_dim"],
            "linear_value_head_dim": model["linear_value_head_dim"],
            "linear_conv_kernel_dim": model["linear_conv_kernel_dim"],
            "max_position_embeddings": MAX_CONTEXT,
            "tie_word_embeddings": True,
            "dtype": "bfloat16",
        }
        if all(text_config.get(key) == value for key, value in contract.items()):
            partial = rope.get("partial_rotary_factor")
            if partial == model["rotary_dim"] / model["head_dim"]:
                return model
    raise ValueError(
        "only official Qwen3.5-0.8B and Qwen3.5-4B text configurations are supported"
    )


def header_values(model):
    """Metadata order shared with model_config.h::ConfigField."""
    return (
        model["model_id"], model["vocab_size"], model["hidden_size"],
        model["intermediate_size"], model["num_hidden_layers"],
        model["full_attention_interval"], model["num_attention_heads"],
        model["num_key_value_heads"], model["head_dim"], model["rotary_dim"],
        model["linear_num_key_heads"], model["linear_num_value_heads"],
        model["linear_key_head_dim"], model["linear_value_head_dim"],
        model["linear_conv_kernel_dim"], MAX_CONTEXT,
    )


def linear_layers(model):
    interval = model["full_attention_interval"]
    for layer in range(model["num_hidden_layers"]):
        yield layer, layer % interval != interval - 1


def expected_tensors(model):
    """Yield the complete sequential tensor schema for one selected model."""
    vocab, hidden = model["vocab_size"], model["hidden_size"]
    intermediate = model["intermediate_size"]
    attention_heads = model["num_attention_heads"]
    kv_heads, attention_dim = model["num_key_value_heads"], model["head_dim"]
    key_heads, value_heads = model["linear_num_key_heads"], model["linear_num_value_heads"]
    key_dim, value_dim = model["linear_key_head_dim"], model["linear_value_head_dim"]
    conv = model["linear_conv_kernel_dim"]
    dqkv = 2 * key_heads * key_dim + value_heads * value_dim

    yield "model.language_model.embed_tokens.weight", "BF16", (vocab, hidden)
    yield "model.language_model.norm.weight", "BF16", (hidden,)
    for layer, is_delta in linear_layers(model):
        layer_prefix = f"model.language_model.layers.{layer}."
        yield layer_prefix + "input_layernorm.weight", "BF16", (hidden,)
        if is_delta:
            prefix = layer_prefix + "linear_attn."
            yield prefix + "in_proj_qkv.weight", "BF16", (dqkv, hidden)
            yield prefix + "in_proj_z.weight", "BF16", (value_heads * value_dim, hidden)
            yield prefix + "in_proj_a.weight", "BF16", (value_heads, hidden)
            yield prefix + "in_proj_b.weight", "BF16", (value_heads, hidden)
            yield prefix + "conv1d.weight", "BF16", (dqkv, 1, conv)
            yield prefix + "A_log", "F32", (value_heads,)
            yield prefix + "dt_bias", "BF16", (value_heads,)
            yield prefix + "norm.weight", "F32", (value_dim,)
            yield prefix + "out_proj.weight", "BF16", (hidden, value_heads * value_dim)
        else:
            prefix = layer_prefix + "self_attn."
            yield prefix + "q_proj.weight", "BF16", (2 * attention_heads * attention_dim, hidden)
            yield prefix + "k_proj.weight", "BF16", (kv_heads * attention_dim, hidden)
            yield prefix + "v_proj.weight", "BF16", (kv_heads * attention_dim, hidden)
            yield prefix + "q_norm.weight", "BF16", (attention_dim,)
            yield prefix + "k_norm.weight", "BF16", (attention_dim,)
            yield prefix + "o_proj.weight", "BF16", (hidden, attention_heads * attention_dim)

        yield layer_prefix + "post_attention_layernorm.weight", "BF16", (hidden,)
        yield layer_prefix + "mlp.gate_proj.weight", "BF16", (intermediate, hidden)
        yield layer_prefix + "mlp.up_proj.weight", "BF16", (intermediate, hidden)
        yield layer_prefix + "mlp.down_proj.weight", "BF16", (hidden, intermediate)


def pad_to_alignment(output):
    remainder = output.tell() % ALIGNMENT
    if remainder:
        output.write(b"\0" * (ALIGNMENT - remainder))


class SafetensorsShard:
    """One open safetensors file with its JSON directory parsed once."""

    def __init__(self, path):
        self.path, self.file, self.header, self.data_start = path, None, None, 0

    def __enter__(self):
        self.file = self.path.open("rb")
        size_bytes = self.file.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"{self.path.name}: missing safetensors header size")
        header_size = struct.unpack("<Q", size_bytes)[0]
        header_bytes = self.file.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError(f"{self.path.name}: truncated safetensors header")
        self.header = json.loads(header_bytes)
        self.data_start = 8 + header_size
        return self

    def __exit__(self, *_):
        self.file.close()

    def copy_tensor(self, name, dtype, shape, output):
        info = self.header.get(name)
        if info is None:
            raise ValueError(f"{name}: absent from {self.path.name}")
        if info["dtype"] != dtype or tuple(info["shape"]) != shape:
            raise ValueError(
                f"{name}: expected {dtype} {shape}, got {info['dtype']} {tuple(info['shape'])}"
            )
        start, end = info["data_offsets"]
        item_size = 2 if dtype == "BF16" else 4
        expected_bytes = item_size
        for dimension in shape:
            expected_bytes *= dimension
        if start < 0 or end < start or end - start != expected_bytes:
            raise ValueError(f"{name}: invalid safetensors data_offsets")

        self.file.seek(self.data_start + start)
        remaining = end - start
        while remaining:
            block = self.file.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError(f"{name}: truncated {self.path.name}")
            output.write(block)
            remaining -= len(block)


def pack(checkpoint_dir, output_path):
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise ValueError(f"missing {index_path}")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    text_config = json.loads((checkpoint_dir / "config.json").read_text())["text_config"]
    model = select_model(text_config)
    expected = list(expected_tensors(model))
    missing = {name for name, _, _ in expected} - weight_map.keys()
    if missing:
        raise ValueError("checkpoint misses text tensors: " + ", ".join(sorted(missing)[:3]))

    shard_names = sorted({weight_map[name] for name, _, _ in expected})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with ExitStack() as stack, temporary.open("wb") as output:
            shards = {
                name: stack.enter_context(SafetensorsShard(checkpoint_dir / name))
                for name in shard_names
            }
            output.write(HEADER.pack(MAGIC, FORMAT_VERSION, 0, *header_values(model)))
            for number, (name, dtype, shape) in enumerate(expected, 1):
                pad_to_alignment(output)
                shards[weight_map[name]].copy_tensor(name, dtype, shape, output)
                print(f"\r[{number:3}/{len(expected)}] {name}", end="", flush=True)
        print()
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        f"wrote {output_path} ({output_path.stat().st_size / 2**30:.2f} GiB, "
        f"{len(expected)} text tensors, {model['name']})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        pack(args.checkpoint_dir, args.output)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
