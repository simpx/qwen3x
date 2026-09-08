#!/usr/bin/env python3
"""Pack the fixed MLX checkpoint without re-quantizing or loading tensor arrays.

The .bin is a single safetensors container with explicit qwen3x MLX metadata.
All text tensor bytes and dtypes are preserved. No MLX/Python runtime dependency
is introduced into the executable by this offline tool.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

REVISION = "38740b847e4cb78f352aba30aa41c76e08e6eb46"


def pack(directory, output):
    config = json.loads((directory / "config.json").read_text())
    if config["quantization"]["mode"] != "affine" or config["quantization"]["bits"] != 4:
        raise ValueError("expected the fixed affine4 checkpoint")
    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    tensors = []
    for shard in sorted(set(index.values())):
        if Path(shard).name != shard:
            raise ValueError("invalid shard path")
        path = directory / shard
        with path.open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            if not 0 < length <= 1024 * 1024:
                raise ValueError("invalid header length")
            header = json.loads(f.read(length))
        for name, item in header.items():
            if not name.startswith("language_model.") or ".mtp." in name:
                continue
            if index.get(name) != shard:
                raise ValueError(f"index mismatch: {name}")
            start, end = item["data_offsets"]
            item_bytes = {"U32": 4, "BF16": 2, "F32": 4}[item["dtype"]]
            for dim in item["shape"]:
                if not isinstance(dim, int) or dim <= 0:
                    raise ValueError(f"invalid shape: {name}")
                item_bytes *= dim
            if end - start != item_bytes or start < 0 or 8 + length + end > path.stat().st_size:
                raise ValueError(f"invalid offsets: {name}")
            tensors.append((name, item, path, 8 + length + start, item_bytes))
    expected = {k for k in index if k.startswith("language_model.") and ".mtp." not in k}
    if {t[0] for t in tensors} != expected or len(tensors) != len(expected):
        raise ValueError("missing or duplicated text tensors")
    header = {"__metadata__": {"qwen3x_format": "mlx-affine-v1", "model_id": "36035",
              "source_revision": REVISION, "config": json.dumps({
                  k: config[k] for k in ["model_type", "text_config", "quantization"]},
                  separators=(",", ":"))}}
    size = 0
    for name, item, _, _, count in tensors:
        header[name] = {"dtype": item["dtype"], "shape": item["shape"], "data_offsets": [size, size + count]}
        size += count
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as out:
        prefix = struct.pack("<Q", len(encoded)) + encoded
        out.write(prefix); digest.update(prefix)
        for name, _, path, offset, count in tensors:
            with path.open("rb") as src:
                src.seek(offset)
                left = count
                while left:
                    block = src.read(min(left, 8 * 1024 * 1024))
                    if not block:
                        raise ValueError(f"truncated tensor: {name}")
                    out.write(block); digest.update(block); left -= len(block)
    if temporary.stat().st_size != 8 + len(encoded) + size:
        raise ValueError("packed size mismatch")
    temporary.replace(output)
    report = {"format": "mlx-affine-v1", "source_revision": REVISION,
              "tensors": len(tensors), "bytes": output.stat().st_size, "sha256": digest.hexdigest(),
              "output": str(output), "requantized": False}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    pack(args.checkpoint, args.output)
