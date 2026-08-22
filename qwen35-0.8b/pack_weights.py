#!/usr/bin/env python3
"""把固定的 Qwen3.5-0.8B text backbone 转为 Engine 使用的顺序二进制文件。

这不是通用模型转换器。它只做一件事：以同目录 engine.cpp 前向执行所需的
顺序写出原始 BF16/F32 tensor 字节。这样 C++ 代码不必包含 JSON parser、
safetensors schema 或多模型分发器。

用法：
    cd qwen35-0.8b
    python3 pack_weights.py ../models/Qwen3.5-0.8B build/qwen35-0.8b.bin

输入目录需要官方 config.json、model.safetensors.index.json 和对应 shard。
脚本只用 Python 标准库；它不会把 1.6 GiB 权重整体读入内存。

阅读提示：expected_tensors() 是这个小格式的完整 schema。每次 yield 的 name、
dtype、shape 和顺序同时是三份契约：官方 checkpoint 的 tensor 名、输出 bin 的
字节布局，以及 engine.cpp 中 Model 构造函数下一次 Reader.take() 应读到的内容。
"""

import argparse
import json
import struct
from pathlib import Path

ALIGNMENT = 64  # 每个 tensor 起点向上对齐，和 C++ Reader.align() 一一对应。
MAGIC = b"Q35COUR\0"  # v1 格式与第 0 章的 00-lessons/09 共享。
FORMAT_VERSION = 1
HEADER = struct.Struct("<8sII")  # magic、格式版本、保留字段，共固定 16 字节。

# 以下常数与同目录 engine.cpp 完全重复，是刻意的“显式固定模型”，不是遗漏。
# 转换器首先用 config.json 检查一部分关键值，避免看似成功地转换了相近但不兼容的模型。
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
    """产出 language-model-only 的固定 tensor stream。

    yield 的 shape 写成 checkpoint 中的原始 shape（所有线性层是 [out,in]）。
    C++ 不保存这些 shape，而是用相同常数重新计算 count；因此此处是转换时的
    防线，能在错误 checkpoint 或官方命名变动时尽早报错。
    """
    # 0.8B 将 lm_head 与 embed_tokens tied，所以只复制 embedding；C++ 最后复用它。
    yield "model.language_model.embed_tokens.weight", "BF16", (V, H)
    yield "model.language_model.norm.weight", "BF16", (H,)

    for layer, is_delta in linear_layers():
        # 每层先是 pre-mixer RMSNorm；之后根据固定的 3:1 pattern 写不同 mixer 权重。
        prefix = f"model.language_model.layers.{layer}."
        yield prefix + "input_layernorm.weight", "BF16", (H,)
        if is_delta:
            # in_proj_qkv 的第一维按 [small Q | small K | value] 拼接，C++ 会按此拆开。
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
            # q_proj 的输出又按 [query | attention gate] 拼接；K/V 的 head 数较少，
            # 这就是 grouped-query attention 的 checkpoint 形状来源。
            prefix += "self_attn."
            yield prefix + "q_proj.weight", "BF16", (2 * AH * AD, H)
            yield prefix + "k_proj.weight", "BF16", (KVH * AD, H)
            yield prefix + "v_proj.weight", "BF16", (KVH * AD, H)
            yield prefix + "q_norm.weight", "BF16", (AD,)
            yield prefix + "k_norm.weight", "BF16", (AD,)
            yield prefix + "o_proj.weight", "BF16", (H, AH * AD)

        # mixer 后每层无条件都有 post norm 和相同形状的 SwiGLU MLP。
        prefix = f"model.language_model.layers.{layer}."
        yield prefix + "post_attention_layernorm.weight", "BF16", (H,)
        yield prefix + "mlp.gate_proj.weight", "BF16", (I, H)
        yield prefix + "mlp.up_proj.weight", "BF16", (I, H)
        yield prefix + "mlp.down_proj.weight", "BF16", (H, I)


def pad_to_alignment(output):
    """在下一个 tensor 之前写 0 padding，使其文件 offset 是 64 的倍数。"""
    remainder = output.tell() % ALIGNMENT
    if remainder:
        output.write(b"\0" * (ALIGNMENT - remainder))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    # index 不是权重本身，而是 tensor name -> shard filename 的目录。
    index_path = args.checkpoint_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"missing {index_path}")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    # 先检查所有会改变 C++ 数学或 shape 的结构字段。
    config = json.loads((args.checkpoint_dir / "config.json").read_text())["text_config"]
    contract = {
        "model_type": "qwen3_5_text",
        "vocab_size": V,
        "hidden_size": H,
        "intermediate_size": I,
        "num_hidden_layers": LAYERS,
        "full_attention_interval": ATN_INTERVAL,
        "num_attention_heads": AH,
        "num_key_value_heads": KVH,
        "head_dim": AD,
        "linear_num_key_heads": KH,
        "linear_num_value_heads": VH,
        "linear_key_head_dim": KD,
        "linear_value_head_dim": VD,
        "linear_conv_kernel_dim": CONV,
        "tie_word_embeddings": True,
        "dtype": "bfloat16",
    }
    wrong = {key: (config.get(key), expected) for key, expected in contract.items()
             if config.get(key) != expected}
    if wrong:
        raise SystemExit("this packer only accepts the official Qwen3.5-0.8B text configuration")

    # 先枚举完整 schema 并检查 names，避免写了一半大文件才发现 checkpoint 缺 tensor。
    expected = list(expected_tensors())
    expected_names = {name for name, _, _ in expected}
    missing = expected_names - weight_map.keys()
    if missing:
        raise SystemExit("checkpoint misses text tensors: " + ", ".join(sorted(missing)[:3]))

    # 官方 0.8B 的全部 text tensor 位于同一个 shard。明确保持这一条直线，避免为了
    # 当前用不到的分发能力重新引入 Shard class 和文件句柄 cache。
    shard_filenames = {weight_map[name] for name in expected_names}
    if len(shard_filenames) != 1:
        raise SystemExit("Qwen3.5-0.8B text tensors must be stored in one safetensors shard")
    shard_path = args.checkpoint_dir / shard_filenames.pop()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 始终先写 sibling .tmp，完整成功后才 replace；中途失败不会留下可被 C++ 误读的 bin。
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        with shard_path.open("rb") as shard, temporary.open("wb") as output:
            header_size_bytes = shard.read(8)
            if len(header_size_bytes) != 8:
                raise ValueError(f"{shard_path.name}: missing safetensors header size")
            header_size = struct.unpack("<Q", header_size_bytes)[0]
            header_bytes = shard.read(header_size)
            if len(header_bytes) != header_size:
                raise ValueError(f"{shard_path.name}: truncated safetensors header")
            shard_header = json.loads(header_bytes)
            data_section_start = 8 + header_size

            output.write(HEADER.pack(MAGIC, FORMAT_VERSION, 0))
            for number, (name, dtype, shape) in enumerate(expected, 1):
                # 每个 take<T>() 都从 64-byte 边界开始，顺序就是 tensor 的唯一索引。
                pad_to_alignment(output)

                info = shard_header.get(name)
                if info is None:
                    raise ValueError(f"{name}: absent from {shard_path.name}")
                if info["dtype"] != dtype or tuple(info["shape"]) != shape:
                    raise ValueError(
                        f"{name}: expected {dtype} {shape}, "
                        f"got {info['dtype']} {tuple(info['shape'])}"
                    )
                start, end = info["data_offsets"]
                item_size = 2 if dtype == "BF16" else 4
                expected_bytes = item_size
                for dimension in shape:
                    expected_bytes *= dimension
                if start < 0 or end < start or end - start != expected_bytes:
                    raise ValueError(f"{name}: invalid safetensors data_offsets")

                shard.seek(data_section_start + start)
                remaining = end - start
                while remaining:
                    block = shard.read(min(8 * 1024 * 1024, remaining))
                    if not block:
                        raise ValueError(f"{name}: truncated shard")
                    output.write(block)
                    remaining -= len(block)
                print(f"\r[{number:3}/{len(expected)}] {name}", end="", flush=True)
        print()
        temporary.replace(args.output)  # 同一文件系统内的原子替换。
    finally:
        if temporary.exists():
            # exception path：清理不完整的输出；成功 replace 后临时路径已经不存在。
            temporary.unlink()

    print(f"wrote {args.output} ({args.output.stat().st_size / 2**30:.2f} GiB, {len(expected)} text tensors)")


if __name__ == "__main__":
    main()
