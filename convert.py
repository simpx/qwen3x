#!/usr/bin/env python3
"""把固定的 Qwen3.5-0.8B text backbone 转为课程使用的顺序二进制文件。

这不是通用模型转换器。它只做一件事：以 capstone/qwen38.cpp 前向执行所需的
顺序写出原始 BF16/F32 tensor 字节。这样 C++ 代码不必包含 JSON parser、
safetensors schema 或多模型分发器。

用法：
    python3 convert.py models/Qwen3.5-0.8B out/qwen38-0.8b.bin

输入目录需要官方 config.json、model.safetensors.index.json 和对应 shard。
脚本只用 Python 标准库；它不会把 1.6 GiB 权重整体读入内存。
"""

import argparse
import json
import struct
from pathlib import Path

ALIGNMENT = 64
MAGIC = b"Q38COUR\0"
HEADER = struct.Struct("<8sII")

V = 248320
H = 1024
I = 3584
LAYERS = 24
ATN_INTERVAL = 4
AH = 8
KVH = 2
AD = 256
RD = 64
KH = 16
VH = 16
KD = 128
VD = 128
CONV = 4


def linear_layers():
    """Qwen 的层 0,1,2 是 DeltaNet，3 是 full attention，循环重复。"""
    for layer in range(LAYERS):
        yield layer, layer % ATN_INTERVAL != ATN_INTERVAL - 1


def expected_tensors():
    """必须与 C++ Model 构造函数中的 take() 顺序一字不差。"""
    yield "model.language_model.embed_tokens.weight", "BF16", (V, H)
    yield "model.language_model.norm.weight", "BF16", (H,)

    for layer, is_delta in linear_layers():
        prefix = f"model.language_model.layers.{layer}."
        yield prefix + "input_layernorm.weight", "BF16", (H,)
        if is_delta:
            prefix += "linear_attn."
            yield prefix + "in_proj_qkv.weight", "BF16", (2 * KH * KD + VH * VD, H)
            yield prefix + "in_proj_z.weight", "BF16", (VH * VD, H)
            yield prefix + "in_proj_a.weight", "BF16", (VH, H)
            yield prefix + "in_proj_b.weight", "BF16", (VH, H)
            yield prefix + "conv1d.weight", "BF16", (2 * KH * KD + VH * VD, 1, CONV)
            yield prefix + "A_log", "F32", (VH,)
            yield prefix + "dt_bias", "BF16", (VH,)
            yield prefix + "norm.weight", "F32", (VD,)
            yield prefix + "out_proj.weight", "BF16", (H, VH * VD)
        else:
            prefix += "self_attn."
            yield prefix + "q_proj.weight", "BF16", (2 * AH * AD, H)
            yield prefix + "k_proj.weight", "BF16", (KVH * AD, H)
            yield prefix + "v_proj.weight", "BF16", (KVH * AD, H)
            yield prefix + "q_norm.weight", "BF16", (AD,)
            yield prefix + "k_norm.weight", "BF16", (AD,)
            yield prefix + "o_proj.weight", "BF16", (H, AH * AD)

        prefix = f"model.language_model.layers.{layer}."
        yield prefix + "post_attention_layernorm.weight", "BF16", (H,)
        yield prefix + "mlp.gate_proj.weight", "BF16", (I, H)
        yield prefix + "mlp.up_proj.weight", "BF16", (I, H)
        yield prefix + "mlp.down_proj.weight", "BF16", (H, I)


class Shard:
    def __init__(self, path):
        self.path = path
        self.file = path.open("rb")
        header_size = struct.unpack("<Q", self.file.read(8))[0]
        self.data_offset = 8 + header_size
        self.header = json.loads(self.file.read(header_size))

    def copy(self, name, output, expected_dtype, expected_shape):
        info = self.header.get(name)
        if info is None:
            raise ValueError(f"{name}: absent from {self.path.name}")
        if info["dtype"] != expected_dtype or tuple(info["shape"]) != expected_shape:
            raise ValueError(
                f"{name}: expected {expected_dtype} {expected_shape}, "
                f"got {info['dtype']} {tuple(info['shape'])}"
            )
        start, end = info["data_offsets"]
        self.file.seek(self.data_offset + start)
        remaining = end - start
        while remaining:
            block = self.file.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError(f"{name}: truncated shard")
            output.write(block)
            remaining -= len(block)

    def close(self):
        self.file.close()


def pad_to_alignment(output):
    remainder = output.tell() % ALIGNMENT
    if remainder:
        output.write(b"\0" * (ALIGNMENT - remainder))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    index_path = args.checkpoint_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"missing {index_path}")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    config = json.loads((args.checkpoint_dir / "config.json").read_text())["text_config"]
    if config["model_type"] != "qwen3_5_text" or config["hidden_size"] != H or config["num_hidden_layers"] != LAYERS:
        raise SystemExit("this converter only accepts the official Qwen3.5-0.8B text configuration")

    expected = list(expected_tensors())
    expected_names = {name for name, _, _ in expected}
    missing = expected_names - weight_map.keys()
    if missing:
        raise SystemExit("checkpoint misses text tensors: " + ", ".join(sorted(missing)[:3]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    shards = {}
    try:
        with temporary.open("wb") as output:
            output.write(HEADER.pack(MAGIC, 1, 0))
            for number, (name, dtype, shape) in enumerate(expected, 1):
                pad_to_alignment(output)
                filename = weight_map[name]
                if filename not in shards:
                    shards[filename] = Shard(args.checkpoint_dir / filename)
                shard = shards[filename]
                shard.copy(name, output, dtype, shape)
                print(f"\r[{number:3}/{len(expected)}] {name}", end="", flush=True)
        print()
        temporary.replace(args.output)
    finally:
        for shard in shards.values():
            shard.close()
        if temporary.exists():
            temporary.unlink()

    print(f"wrote {args.output} ({args.output.stat().st_size / 2**30:.2f} GiB, {len(expected)} text tensors)")


if __name__ == "__main__":
    main()
