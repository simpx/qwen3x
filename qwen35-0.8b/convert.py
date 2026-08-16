#!/usr/bin/env python3
"""把固定的 Qwen3.5-0.8B text backbone 转为课程使用的顺序二进制文件。

这不是通用模型转换器。它只做一件事：以同目录 qwen35.cpp 前向执行所需的
顺序写出原始 BF16/F32 tensor 字节。这样 C++ 代码不必包含 JSON parser、
safetensors schema 或多模型分发器。

用法：
    cd qwen35-0.8b
    python3 convert.py ../models/Qwen3.5-0.8B out/qwen35-0.8b.bin

输入目录需要官方 config.json、model.safetensors.index.json 和对应 shard。
脚本只用 Python 标准库；它不会把 1.6 GiB 权重整体读入内存。

阅读提示：expected_tensors() 是这个小格式的完整 schema。每次 yield 的 name、
dtype、shape 和顺序同时是三份契约：官方 checkpoint 的 tensor 名、输出 bin 的
字节布局，以及 qwen35.cpp 中 Model 构造函数下一次 Reader.take() 应读到的内容。
"""

import argparse
import json
import struct
from pathlib import Path

ALIGNMENT = 64  # 每个 tensor 起点向上对齐，和 C++ Reader.align() 一一对应。
MAGIC = b"Q35COUR\0"  # 防止把任意文件误当作本课程 model.bin。
HEADER = struct.Struct("<8sII")  # magic、格式版本、保留字段，共固定 16 字节。

# 以下常数与同目录 qwen35.cpp 完全重复，是刻意的“显式固定模型”，不是遗漏。
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


class Shard:
    """一个 safetensors shard 的轻量只读视图。

    safetensors 的前 8 字节是小端 u64 header 长度；随后是 JSON header，tensor
    data_offsets 相对 data section。我们只解析这点格式信息，按块拷贝所需 tensor，
    不需要 numpy、torch 或 safetensors runtime。
    """
    def __init__(self, path):
        self.path = path
        self.file = path.open("rb")
        header_size = struct.unpack("<Q", self.file.read(8))[0]
        self.data_offset = 8 + header_size
        self.header = json.loads(self.file.read(header_size))

    def copy(self, name, output, expected_dtype, expected_shape):
        """验证一个 tensor 后，将它的原始字节流复制到输出。

        BF16 与 F32 都无需 Python 解码/重编码：C++ 将在使用 BF16 时做 bit-level
        转换，而 F32 norm/A_log 保持原字节。因此内存峰值只是一块 8 MiB copy buffer。
        """
        info = self.header.get(name)
        if info is None:
            raise ValueError(f"{name}: absent from {self.path.name}")
        if info["dtype"] != expected_dtype or tuple(info["shape"]) != expected_shape:
            raise ValueError(
                f"{name}: expected {expected_dtype} {expected_shape}, "
                f"got {info['dtype']} {tuple(info['shape'])}"
            )
        # data_offsets 是相对 data_offset 的半开区间 [start,end)，不是文件绝对位置。
        start, end = info["data_offsets"]
        self.file.seek(self.data_offset + start)
        remaining = end - start
        while remaining:
            # 分块读写让 1.4+ GiB 转换不会创建巨大的 Python bytes 对象。
            block = self.file.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError(f"{name}: truncated shard")
            output.write(block)
            remaining -= len(block)

    def close(self):
        # Shard 可被多个 tensor 复用；main 的 finally 统一关闭所有已经打开的文件。
        self.file.close()


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
    # 先检查模型家族、hidden width 和层数。更细的矩阵 shape 由 Shard.copy 再检查。
    config = json.loads((args.checkpoint_dir / "config.json").read_text())["text_config"]
    if config["model_type"] != "qwen3_5_text" or config["hidden_size"] != H or config["num_hidden_layers"] != LAYERS:
        raise SystemExit("this converter only accepts the official Qwen3.5-0.8B text configuration")

    # 先枚举完整 schema 并检查 names，避免写了一半大文件才发现 checkpoint 缺 tensor。
    expected = list(expected_tensors())
    expected_names = {name for name, _, _ in expected}
    missing = expected_names - weight_map.keys()
    if missing:
        raise SystemExit("checkpoint misses text tensors: " + ", ".join(sorted(missing)[:3]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 始终先写 sibling .tmp，完整成功后才 replace；中途失败不会留下可被 C++ 误读的 bin。
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    shards = {}
    try:
        with temporary.open("wb") as output:
            output.write(HEADER.pack(MAGIC, 1, 0))
            for number, (name, dtype, shape) in enumerate(expected, 1):
                # 每个 take<T>() 都从 64-byte 边界开始，顺序就是 tensor 的唯一索引。
                pad_to_alignment(output)
                filename = weight_map[name]
                if filename not in shards:
                    # 一个 safetensors shard 通常装多个 tensor，因此缓存打开的文件句柄。
                    shards[filename] = Shard(args.checkpoint_dir / filename)
                shard = shards[filename]
                shard.copy(name, output, dtype, shape)
                print(f"\r[{number:3}/{len(expected)}] {name}", end="", flush=True)
        print()
        temporary.replace(args.output)  # 同一文件系统内的原子替换。
    finally:
        for shard in shards.values():
            shard.close()
        if temporary.exists():
            # exception path：清理不完整的输出；成功 replace 后临时路径已经不存在。
            temporary.unlink()

    print(f"wrote {args.output} ({args.output.stat().st_size / 2**30:.2f} GiB, {len(expected)} text tensors)")


if __name__ == "__main__":
    main()
