#!/usr/bin/env python3
"""把固定的 Qwen3.5-0.8B text backbone 转为课程使用的顺序二进制文件。

这不是通用模型转换器。它只做一件事：以同目录 09_qwen35_0_8b.cpp 前向执行所需的
顺序写出原始 BF16/F32 tensor 字节。这样 C++ 代码不必包含 JSON parser、
safetensors schema 或多模型分发器。

输入 checkpoint 目录里的文件格式：

1. config.json：模型结构说明。本脚本只读其中的 text_config，确认
   model_type、hidden_size 和 num_hidden_layers 确实是本课固定的 0.8B。

2. model.safetensors.index.json：外部目录，只记录 tensor 在哪个 shard 文件：

       {
         "weight_map": {
           "tensor.name": "model.safetensors-00001-of-00001.safetensors"
         }
       }

   它不记录 tensor 在 shard 内的字节位置。

3. 每个 .safetensors shard：一个二进制文件，布局是：

       [8-byte 小端 u64: JSON header 长度]
       [JSON header]
       [所有 tensor 的原始连续字节，也就是 data section]

   JSON header 按 tensor name 记录类型、shape 和位置：

       "tensor.name": {
         "dtype": "BF16",
         "shape": [1024, 3584],
         "data_offsets": [start, end]
       }

   data_offsets 是相对 data section 开头的半开区间 [start,end)，不是整个
   shard 的绝对 offset。所以 tensor 的文件绝对起点是：

       8 + header_size + start

一个 tensor 的两层查找过程：

    tensor name
      -> index.json 的 weight_map 找到 shard filename
      -> shard 内部 JSON header 找到 dtype / shape / data_offsets
      -> seek(8 + header_size + start)
      -> 原样复制 end-start 个字节

输出 qwen35-0.8b.bin 的格式：

    [16-byte: magic + format version + reserved]
    [0 padding，使下一个 tensor 从 64-byte 边界开始]
    [expected_tensors() 第 1 个 tensor 的原始字节]
    [0 padding 到下一个 64-byte 边界]
    [expected_tensors() 第 2 个 tensor 的原始字节]
    ...

输出 bin 不再保存 tensor name、dtype 或 shape；它只保存对齐后的原始字节。
09_qwen35_0_8b.cpp 必须用完全相同的顺序和 shape 调用 Reader.take()。

用法：
    cd from-scratch
    python3 09_pack_weights.py ../models/Qwen3.5-0.8B ../models/qwen35-0.8b.bin

输入目录需要官方 config.json、model.safetensors.index.json 和对应 shard。
本课的固定 0.8B checkpoint 只有一个 shard；脚本会明确检查所有需要的 text
tensor 都指向这同一个文件，再在 main() 中按顺序直接复制。
脚本只用 Python 标准库；它不会把 1.6 GiB 权重整体读入内存。

阅读提示：expected_tensors() 是这个小格式的完整 schema。每次 yield 的 name、
dtype、shape 和顺序同时是三份契约：官方 checkpoint 的 tensor 名、输出 bin 的
字节布局，以及 09_qwen35_0_8b.cpp 中 Model 构造函数下一次 Reader.take() 应读到的内容。
"""

import argparse
import json
import struct
from pathlib import Path

ALIGNMENT = 64  # 每个 tensor 起点向上对齐，和 C++ Reader.align() 一一对应。
MAGIC = b"Q35COUR\0"  # 防止把任意文件误当作本课程 model.bin。
HEADER = struct.Struct("<8sII")  # magic、格式版本、保留字段，共固定 16 字节。

# 以下常数与同目录 09_qwen35_0_8b.cpp 完全重复，是刻意的“显式固定模型”，不是遗漏。
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
    # 先检查模型家族、hidden width 和层数。更细的矩阵 shape 在下面复制前逐个检查。
    config = json.loads((args.checkpoint_dir / "config.json").read_text())["text_config"]
    if config["model_type"] != "qwen3_5_text" or config["hidden_size"] != H or config["num_hidden_layers"] != LAYERS:
        raise SystemExit("this packer only accepts the official Qwen3.5-0.8B text configuration")

    # 先枚举完整 schema 并检查 names，避免写了一半大文件才发现 checkpoint 缺 tensor。
    expected = list(expected_tensors())
    expected_names = {name for name, _, _ in expected}
    missing = expected_names - weight_map.keys()
    if missing:
        raise SystemExit("checkpoint misses text tensors: " + ", ".join(sorted(missing)[:3]))

    # 本课固定的官方 0.8B 是 00001-of-00001：所有需要的 text tensor 都在
    # 同一个 safetensors 文件。明确拒绝多 shard，就能把下面的读取路径保持为
    # 一条直线，而不需要 Shard class、文件句柄 cache 或分发逻辑。
    shard_filenames = {weight_map[name] for name in expected_names}
    if len(shard_filenames) != 1:
        raise SystemExit("this lesson expects all Qwen3.5-0.8B text tensors in one shard")
    shard_path = args.checkpoint_dir / shard_filenames.pop()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 始终先写 sibling .tmp，完整成功后才 replace；中途失败不会留下可被 C++ 误读的 bin。
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        # safetensors = [8-byte header_size][JSON header][raw tensor data section]。
        # 本课只打开这一个 shard，并在这里直接读出内部目录。
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

            output.write(HEADER.pack(MAGIC, 1, 0))
            for number, (name, dtype, shape) in enumerate(expected, 1):
                # 每个 take<T>() 都从 64-byte 边界开始，顺序就是 tensor 的唯一索引。
                pad_to_alignment(output)

                # 名字 -> 内部 header 条目 -> dtype/shape/相对字节区间。
                info = shard_header.get(name)
                if info is None:
                    raise ValueError(f"{name}: absent from {shard_path.name}")
                if info["dtype"] != dtype or tuple(info["shape"]) != shape:
                    raise ValueError(
                        f"{name}: expected {dtype} {shape}, "
                        f"got {info['dtype']} {tuple(info['shape'])}"
                    )

                # data_offsets=[start,end) 相对 data section；加上 data_section_start
                # 才是 shard 内的绝对文件位置。
                start, end = info["data_offsets"]
                shard.seek(data_section_start + start)
                remaining = end - start
                while remaining:
                    # 每次最多复制 8 MiB，不把整个大 tensor 一次性放进内存。
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
