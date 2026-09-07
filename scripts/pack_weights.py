#!/usr/bin/env python3
"""Pack a supported official Qwen3.5/Qwen3.6/Qwen3.8 text backbone.

The output is a deliberately small sequential format: a fixed metadata header
followed by tensors in the exact order consumed by engine.cpp. The 4B
checkpoint spans multiple safetensors shards, so the packer keeps shard headers
open and streams tensors without loading the model into RAM.
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

ALIGNMENT = 64
MAGIC = b"Q3XMODL\0"
HEADER = struct.Struct("<8s16I")
MAX_CONTEXT = 262144
Q8_BLOCK_SIZE = 32
Q4_BLOCK_SIZE = 32
Q8_DTYPE = np.dtype([
    ("scale", "<f2"),
    ("values", "i1", (Q8_BLOCK_SIZE,)),
], align=False)
assert Q8_DTYPE.itemsize == 34
Q4_DTYPE = np.dtype([
    ("scale", "<f2"),
    ("values", "u1", (Q4_BLOCK_SIZE // 2,)),
], align=False)
assert Q4_DTYPE.itemsize == 18

QWEN36_TOKENIZER_SHA256 = {
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
}

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
        "matrix_type": "BF16", "tie_word_embeddings": True,
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
        "matrix_type": "BF16", "tie_word_embeddings": True,
    },
    {
        "name": "Qwen3.5-9B", "model_id": 9000,
        "vocab_size": 248320, "hidden_size": 4096,
        "intermediate_size": 12288, "num_hidden_layers": 32,
        "full_attention_interval": 4, "num_attention_heads": 16,
        "num_key_value_heads": 4, "head_dim": 256, "rotary_dim": 64,
        "linear_num_key_heads": 16, "linear_num_value_heads": 32,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "matrix_type": "Q8_0", "tie_word_embeddings": False,
    },
    {
        "name": "Qwen3.6-35B-A3B", "model_id": 36035,
        "vocab_size": 248320, "hidden_size": 2048,
        "intermediate_size": 512, "num_hidden_layers": 40,
        "full_attention_interval": 4, "num_attention_heads": 16,
        "num_key_value_heads": 2, "head_dim": 256, "rotary_dim": 64,
        "linear_num_key_heads": 16, "linear_num_value_heads": 32,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 256, "num_experts_per_tok": 8,
        "shared_expert_intermediate_size": 512,
        "matrix_type": "Q4_0", "tie_word_embeddings": False,
        "delta_parameter_dtype": "BF16",
    },
    {
        "name": "Qwen3.8-27B", "model_id": 38027,
        "vocab_size": 248320, "hidden_size": 5120,
        "intermediate_size": 17408, "num_hidden_layers": 64,
        "full_attention_interval": 4, "num_attention_heads": 24,
        "num_key_value_heads": 4, "head_dim": 256, "rotary_dim": 64,
        "linear_num_key_heads": 16, "linear_num_value_heads": 48,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "matrix_type": "Q4_0", "tie_word_embeddings": False,
        "output_gate_type": "swish", "delta_parameter_dtype": "BF16",
    },
)


def select_model(text_config, tie_word_embeddings=None):
    """Return the one supported shape that exactly matches structural fields."""
    rope = text_config.get("rope_parameters", {})
    for model in SUPPORTED_MODELS:
        moe = model.get("num_experts", 0) > 0
        contract = {
            "model_type": "qwen3_5_moe_text" if moe else "qwen3_5_text",
            "vocab_size": model["vocab_size"],
            "hidden_size": model["hidden_size"],
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
            "tie_word_embeddings": model["tie_word_embeddings"],
            "dtype": "bfloat16",
            "attn_output_gate": True,
        }
        if moe:
            contract.update({
                "moe_intermediate_size": model["intermediate_size"],
                "num_experts": model["num_experts"],
                "num_experts_per_tok": model["num_experts_per_tok"],
                "shared_expert_intermediate_size":
                    model["shared_expert_intermediate_size"],
                "hidden_act": "silu",
                "mamba_ssm_dtype": "float32",
                "rms_norm_eps": 1e-6,
                "layer_types": [
                    "full_attention" if layer % model["full_attention_interval"] ==
                    model["full_attention_interval"] - 1 else "linear_attention"
                    for layer in range(model["num_hidden_layers"])
                ],
            })
        else:
            contract["intermediate_size"] = model["intermediate_size"]
        actual_tie = text_config.get("tie_word_embeddings", tie_word_embeddings)
        matches = all(
            (actual_tie if key == "tie_word_embeddings" else text_config.get(key)) == value
            for key, value in contract.items()
        )
        matches = matches and (
            text_config.get("output_gate_type", "silu") ==
            model.get("output_gate_type", "silu")
        )
        if matches:
            partial = rope.get("partial_rotary_factor")
            rope_matches = partial == model["rotary_dim"] / model["head_dim"]
            if moe:
                rope_matches = (rope_matches and rope.get("rope_theta") == 10000000
                                and rope.get("rope_type") == "default")
            if rope_matches:
                return model
    raise ValueError(
        "only official Qwen3.5-0.8B/4B/9B, Qwen3.6-35B-A3B and Qwen3.8-27B "
        "text configurations are supported"
    )


def storage_layout(model, tensors=None):
    """Return exact parameter/storage totals for the fixed sequential schema."""
    cursor = HEADER.size
    parameters = 0
    layer_start = None
    layer_sizes = []
    current_layer = None
    schema = tensors if tensors is not None else expected_tensors(model)
    for name, dtype, shape, matrix in schema:
        layer_text = name.split("model.language_model.layers.", 1)
        layer = int(layer_text[1].split(".", 1)[0]) if len(layer_text) == 2 else None
        if layer != current_layer:
            if current_layer is not None:
                layer_sizes.append(cursor - layer_start)
            current_layer = layer
            layer_start = cursor if layer is not None else None
        cursor += (-cursor) % ALIGNMENT
        count = math.prod(shape)
        parameters += count
        if matrix and model["matrix_type"] == "Q8_0":
            cursor += count // Q8_BLOCK_SIZE * Q8_DTYPE.itemsize
        elif matrix and model["matrix_type"] == "Q4_0":
            cursor += count // Q4_BLOCK_SIZE * Q4_DTYPE.itemsize
        else:
            cursor += count * (2 if dtype == "BF16" else 4)
    if current_layer is not None:
        layer_sizes.append(cursor - layer_start)
    return {"parameters": parameters, "model_bytes": cursor,
            "layer_bytes": tuple(layer_sizes)}


def print_storage_layout(model, tensors=None):
    layout = storage_layout(model, tensors)
    layers = layout["layer_bytes"]
    print(
        f"layout {model['name']}: text_parameters={layout['parameters']} "
        f"model_bytes={layout['model_bytes']} ({layout['model_bytes'] / 2**30:.2f} GiB) "
        f"layers={len(layers)} max_layer_bytes={max(layers, default=0)}"
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

    yield "model.language_model.embed_tokens.weight", "BF16", (vocab, hidden), True
    if not model["tie_word_embeddings"]:
        yield "lm_head.weight", "BF16", (vocab, hidden), True
    yield "model.language_model.norm.weight", "BF16", (hidden,), False
    for layer, is_delta in linear_layers(model):
        layer_prefix = f"model.language_model.layers.{layer}."
        yield layer_prefix + "input_layernorm.weight", "BF16", (hidden,), False
        if is_delta:
            prefix = layer_prefix + "linear_attn."
            yield prefix + "in_proj_qkv.weight", "BF16", (dqkv, hidden), True
            yield prefix + "in_proj_z.weight", "BF16", (value_heads * value_dim, hidden), True
            yield prefix + "in_proj_a.weight", "BF16", (value_heads, hidden), True
            yield prefix + "in_proj_b.weight", "BF16", (value_heads, hidden), True
            yield prefix + "conv1d.weight", "BF16", (dqkv, 1, conv), False
            yield prefix + "A_log", "F32", (value_heads,), False
            yield prefix + "dt_bias", "BF16", (value_heads,), False
            yield prefix + "norm.weight", "F32", (value_dim,), False
            yield prefix + "out_proj.weight", "BF16", (hidden, value_heads * value_dim), True
        else:
            prefix = layer_prefix + "self_attn."
            yield prefix + "q_proj.weight", "BF16", (2 * attention_heads * attention_dim, hidden), True
            yield prefix + "k_proj.weight", "BF16", (kv_heads * attention_dim, hidden), True
            yield prefix + "v_proj.weight", "BF16", (kv_heads * attention_dim, hidden), True
            yield prefix + "q_norm.weight", "BF16", (attention_dim,), False
            yield prefix + "k_norm.weight", "BF16", (attention_dim,), False
            yield prefix + "o_proj.weight", "BF16", (hidden, attention_heads * attention_dim), True

        yield layer_prefix + "post_attention_layernorm.weight", "BF16", (hidden,), False
        if model.get("num_experts", 0):
            experts = model["num_experts"]
            shared = model["shared_expert_intermediate_size"]
            yield layer_prefix + "mlp.gate.weight", "BF16", (experts, hidden), False
            yield layer_prefix + "mlp.experts.gate_up_proj", "BF16", \
                (experts, 2 * intermediate, hidden), True
            yield layer_prefix + "mlp.experts.down_proj", "BF16", \
                (experts, hidden, intermediate), True
            yield layer_prefix + "mlp.shared_expert.gate_proj.weight", "BF16", \
                (shared, hidden), True
            yield layer_prefix + "mlp.shared_expert.up_proj.weight", "BF16", \
                (shared, hidden), True
            yield layer_prefix + "mlp.shared_expert.down_proj.weight", "BF16", \
                (hidden, shared), True
            yield layer_prefix + "mlp.shared_expert_gate.weight", "BF16", \
                (1, hidden), False
        else:
            yield layer_prefix + "mlp.gate_proj.weight", "BF16", (intermediate, hidden), True
            yield layer_prefix + "mlp.up_proj.weight", "BF16", (intermediate, hidden), True
            yield layer_prefix + "mlp.down_proj.weight", "BF16", (hidden, intermediate), True


def checkpoint_dtype(model, name, packed_dtype):
    """Return the official checkpoint dtype for one packed tensor."""
    if packed_dtype == "F32" and ".linear_attn." in name:
        return model.get("delta_parameter_dtype", "F32")
    return packed_dtype


def quantize_q8_0(data):
    """Convert little-endian BF16 rows, already split on 32-value blocks."""
    if len(data) % (Q8_BLOCK_SIZE * 2):
        raise ValueError("Q8_0 input does not contain complete 32-value blocks")
    bits = np.frombuffer(data, dtype="<u2").astype("<u4")
    bits <<= 16
    values = bits.view("<f4").reshape(-1, Q8_BLOCK_SIZE)
    if not np.isfinite(values).all():
        raise ValueError("Q8_0 input contains non-finite weight")
    maximum = np.max(np.abs(values), axis=1)
    scale = maximum / np.float32(127.0)
    inverse = np.zeros_like(scale)
    np.divide(np.float32(1.0), scale, out=inverse, where=scale != 0)
    normalized = values * inverse[:, None]
    rounded = np.copysign(np.floor(np.abs(normalized) + np.float32(0.5)), normalized)
    quants = np.clip(rounded, -127, 127).astype(np.int8)
    blocks = np.empty(scale.size, dtype=Q8_DTYPE)
    blocks["scale"] = scale.astype("<f2")
    blocks["values"] = quants
    return blocks.tobytes()


def quantize_q4_0(data):
    """Convert little-endian BF16 rows to ggml-compatible Q4_0 blocks."""
    if len(data) % (Q4_BLOCK_SIZE * 2):
        raise ValueError("Q4_0 input does not contain complete 32-value blocks")
    bits = np.frombuffer(data, dtype="<u2").astype("<u4")
    bits <<= 16
    values = bits.view("<f4").reshape(-1, Q4_BLOCK_SIZE)
    if not np.isfinite(values).all():
        raise ValueError("Q4_0 input contains non-finite weight")

    largest = np.argmax(np.abs(values), axis=1)
    maximum = values[np.arange(values.shape[0]), largest]
    scale = maximum / np.float32(-8.0)
    inverse = np.zeros_like(scale)
    np.divide(np.float32(1.0), scale, out=inverse, where=scale != 0)
    normalized = values * inverse[:, None]
    quants = np.minimum(15, np.trunc(normalized + np.float32(8.5)))
    quants = quants.astype(np.uint8)
    packed = quants[:, :16] | (quants[:, 16:] << 4)
    blocks = np.empty(scale.size, dtype=Q4_DTYPE)
    blocks["scale"] = scale.astype("<f2")
    blocks["values"] = packed
    return blocks.tobytes()


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

    def copy_tensor(self, name, dtype, shape, quantized, output,
                    source_dtype=None):
        info = self.header.get(name)
        if info is None:
            raise ValueError(f"{name}: absent from {self.path.name}")
        source_dtype = source_dtype or dtype
        if info["dtype"] != source_dtype or tuple(info["shape"]) != shape:
            raise ValueError(
                f"{name}: expected {source_dtype} {shape}, "
                f"got {info['dtype']} {tuple(info['shape'])}"
            )
        start, end = info["data_offsets"]
        item_size = 2 if source_dtype == "BF16" else 4
        expected_bytes = item_size
        for dimension in shape:
            expected_bytes *= dimension
        if start < 0 or end < start or end - start != expected_bytes:
            raise ValueError(f"{name}: invalid safetensors data_offsets")

        self.file.seek(self.data_start + start)
        if quantized:
            if (quantized not in ("Q8_0", "Q4_0") or
                    source_dtype != "BF16" or dtype != "BF16" or
                    len(shape) not in (2, 3) or shape[-1] % Q8_BLOCK_SIZE):
                raise ValueError(f"{name}: invalid {quantized} matrix shape {shape}")
            rows = math.prod(shape[:-1])
            row_bytes = shape[-1] * 2
            rows_per_chunk = max(1, (8 * 1024 * 1024) // row_bytes)
            for row in range(0, rows, rows_per_chunk):
                count = min(rows_per_chunk, rows - row)
                block = self.file.read(count * row_bytes)
                if len(block) != count * row_bytes:
                    raise ValueError(f"{name}: truncated {self.path.name}")
                output.write(
                    quantize_q4_0(block) if quantized == "Q4_0"
                    else quantize_q8_0(block)
                )
            return
        if source_dtype != dtype:
            if source_dtype != "BF16" or dtype != "F32":
                raise ValueError(
                    f"{name}: unsupported {source_dtype} -> {dtype} conversion"
                )
            data = self.file.read(end - start)
            if len(data) != end - start:
                raise ValueError(f"{name}: truncated {self.path.name}")
            bits = np.frombuffer(data, dtype="<u2").astype("<u4")
            bits <<= 16
            output.write(bits.view("<f4").tobytes())
            return
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
    config = json.loads((checkpoint_dir / "config.json").read_text())
    text_config = config["text_config"]
    model = select_model(text_config, config.get("tie_word_embeddings"))
    if model["model_id"] == 36035:
        if config.get("model_type") != "qwen3_5_moe":
            raise ValueError("Qwen3.6 checkpoint must use outer model_type=qwen3_5_moe")
        for filename, expected_hash in QWEN36_TOKENIZER_SHA256.items():
            path = checkpoint_dir / filename
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"Qwen3.6 tokenizer mismatch: {filename}")
    expected = list(expected_tensors(model))
    expected_names = {name for name, _, _, _ in expected}
    missing = expected_names - weight_map.keys()
    if missing:
        raise ValueError("checkpoint misses text tensors: " + ", ".join(sorted(missing)[:3]))
    actual_text_names = {
        name for name in weight_map
        if name == "lm_head.weight" or name.startswith("model.language_model.")
    }
    unexpected = actual_text_names - expected_names
    if unexpected:
        raise ValueError("checkpoint has unexpected text tensors: " +
                         ", ".join(sorted(unexpected)[:3]))
    print_storage_layout(model, expected)

    shard_names = sorted({weight_map[name] for name, _, _, _ in expected})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with ExitStack() as stack, temporary.open("wb") as output:
            shards = {
                name: stack.enter_context(SafetensorsShard(checkpoint_dir / name))
                for name in shard_names
            }
            output.write(HEADER.pack(MAGIC, *header_values(model)))
            for number, (name, dtype, shape, matrix) in enumerate(expected, 1):
                pad_to_alignment(output)
                quantized = (model["matrix_type"]
                             if matrix and model["matrix_type"] != "BF16"
                             else None)
                shards[weight_map[name]].copy_tensor(
                    name, dtype, shape, quantized, output,
                    checkpoint_dtype(model, name, dtype))
                print(f"\r[{number:3}/{len(expected)}] {name}", end="", flush=True)
            expected_size = storage_layout(model, expected)["model_bytes"]
            if output.tell() != expected_size:
                raise ValueError(
                    f"packed size {output.tell()} does not match schema {expected_size}"
                )
        print()
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = hashlib.sha256()
    with output_path.open("rb") as packed:
        for block in iter(lambda: packed.read(8 * 1024 * 1024), b""):
            digest.update(block)
    print(f"wrote {output_path} ({output_path.stat().st_size} bytes, "
          f"{output_path.stat().st_size / 2**30:.2f} GiB, {len(expected)} text tensors, "
          f"{model['name']}, model ID {model['model_id']}, sha256 {digest.hexdigest()})")


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
