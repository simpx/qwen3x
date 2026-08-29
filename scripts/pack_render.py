#!/usr/bin/env python3
"""Pack Qwen3.5 render tables into render.cpp's dependency-free format.

The runtime format contains raw byte tokens, merge pairs and the small slice of
Unicode data needed by Qwen's NFC normalizer and pre-tokenizer.  Converting the
ByteLevel alphabet here keeps render.cpp about model behavior rather than JSON.
"""

import argparse
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path

from tokenizers import __version__ as tokenizers_version


MAGIC = b"Q35RND1\0"
VERSION = 1
MODEL_VOCAB_SIZE = 248320
HEADER = struct.Struct("<8sIIQ9I")

# Little-endian file order:
#   fixed header (magic, version, header/file sizes, nine table counts)
#   source metadata, byte IDs, base vocab, merges, added tokens
#   Unicode category ranges, combining classes, decompositions, compositions
# Variable strings are u32 byte length followed by bytes. There are no offsets:
# every section is consumed exactly once in this fixed order.

def byte_alphabet():
    """Return GPT-2 ByteLevel's byte -> Unicode codepoint bijection."""
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    bytes_ = list(visible)
    codepoints = list(visible)
    extra = 0
    for byte in range(256):
        if byte not in visible:
            bytes_.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return {byte: chr(codepoint) for byte, codepoint in zip(bytes_, codepoints)}


def raw_token(text, inverse):
    try:
        return bytes(inverse[character] for character in text)
    except KeyError as error:
        raise ValueError(f"base token contains a non-ByteLevel character: {error.args[0]!r}")


def added_tokens(tokenizer, config):
    by_id = {}
    for token in tokenizer["added_tokens"]:
        by_id[int(token["id"])] = dict(token)
    for id_text, token in config.get("added_tokens_decoder", {}).items():
        token = dict(token)
        token["id"] = int(id_text)
        previous = by_id.get(token["id"])
        if previous is not None and previous["content"] != token["content"]:
            raise ValueError(f"added token {token['id']} differs between tokenizer files")
        by_id[token["id"]] = token

    result = []
    for token_id, token in sorted(by_id.items()):
        unsupported = {
            name: token.get(name) for name in
            ("lstrip", "rstrip", "single_word", "normalized")
            if token.get(name, False)
        }
        if unsupported:
            raise ValueError(f"added token {token_id} has unsupported flags: {unsupported}")
        result.append((token_id, token["content"].encode(), bool(token.get("special"))))
    return result


def write_string(output, value):
    data = value.encode()
    output.write(struct.pack("<I", len(data)))
    output.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    tokenizer_path = args.checkpoint_dir / "tokenizer.json"
    config_path = args.checkpoint_dir / "tokenizer_config.json"
    tokenizer_bytes = tokenizer_path.read_bytes()
    config_bytes = config_path.read_bytes()
    tokenizer = json.loads(tokenizer_bytes)
    config = json.loads(config_bytes)

    expected_regex = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|"
        r"\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )
    contract = {
        "normalizer": tokenizer.get("normalizer"),
        "pre_tokenizer": tokenizer.get("pre_tokenizer"),
        "decoder": tokenizer.get("decoder"),
        "model_type": tokenizer.get("model", {}).get("type"),
        "dropout": tokenizer.get("model", {}).get("dropout"),
        "byte_fallback": tokenizer.get("model", {}).get("byte_fallback"),
        "ignore_merges": tokenizer.get("model", {}).get("ignore_merges"),
    }
    pretokenizers = contract["pre_tokenizer"].get("pretokenizers", [])
    valid = (
        contract["normalizer"] == {"type": "NFC"}
        and len(pretokenizers) == 2
        and pretokenizers[0].get("type") == "Split"
        and pretokenizers[0].get("pattern", {}).get("Regex") == expected_regex
        and pretokenizers[1] == {
            "type": "ByteLevel", "add_prefix_space": False,
            "trim_offsets": False, "use_regex": False,
        }
        and contract["decoder"] == {
            "type": "ByteLevel", "add_prefix_space": False,
            "trim_offsets": False, "use_regex": False,
        }
        and contract["model_type"] == "BPE"
        and contract["dropout"] is None
        and contract["byte_fallback"] is False
        and contract["ignore_merges"] is False
    )
    if not valid:
        raise SystemExit("tokenizer is not the fixed Qwen3.5 ByteLevel-BPE contract")

    vocab = tokenizer["model"]["vocab"]
    base_count = len(vocab)
    if base_count != 248044 or set(vocab.values()) != set(range(base_count)):
        raise SystemExit("Qwen3.5 base vocabulary must contain dense IDs 0..248043")

    byte_to_character = byte_alphabet()
    inverse = {character: byte for byte, character in byte_to_character.items()}
    by_id = [None] * base_count
    for text, token_id in vocab.items():
        by_id[token_id] = raw_token(text, inverse)
    byte_ids = [vocab[byte_to_character[byte]] for byte in range(256)]

    merges = []
    for rank, merge in enumerate(tokenizer["model"]["merges"]):
        pieces = merge.split(" ")
        if len(pieces) != 2:
            raise ValueError(f"merge {rank} is not a pair: {merge!r}")
        left, right = pieces
        try:
            merges.append((vocab[left], vocab[right], vocab[left + right]))
        except KeyError as error:
            raise ValueError(f"merge {rank} references an unknown token: {error.args[0]!r}")

    added = added_tokens(tokenizer, config)
    decodable_count = max(base_count, max(token_id for token_id, _, _ in added) + 1)
    # Unicode oracle sections are added in the next incremental step. Keeping
    # their counts in the v1 header fixes the final format without mixing the
    # core tokenizer tables and Unicode extraction in one review.
    ranges = []
    combining = []
    decompositions = []
    compositions = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=args.output.parent,
            prefix=args.output.name + ".", suffix=".tmp", delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(HEADER.pack(
                MAGIC, VERSION, HEADER.size, 0,
                MODEL_VOCAB_SIZE, base_count, decodable_count,
                len(merges), len(added), len(ranges), len(combining),
                len(decompositions), len(compositions),
            ))
            write_string(output, f"tokenizers-{tokenizers_version}")
            write_string(output, hashlib.sha256(tokenizer_bytes).hexdigest())
            write_string(output, hashlib.sha256(config_bytes).hexdigest())

            output.write(struct.pack("<256I", *byte_ids))
            for token in by_id:
                output.write(struct.pack("<I", len(token)))
                output.write(token)
            for left, right, result in merges:
                output.write(struct.pack("<III", left, right, result))
            for token_id, content, special in added:
                output.write(struct.pack("<IBI", token_id, special, len(content)))
                output.write(content)
            for first, last, flags in ranges:
                output.write(struct.pack("<IIB", first, last, flags))
            for codepoint, ccc in combining:
                output.write(struct.pack("<IB", codepoint, ccc))
            for codepoint, decomposition in decompositions:
                output.write(struct.pack("<IB", codepoint, len(decomposition)))
                output.write(struct.pack(f"<{len(decomposition)}I", *decomposition))
            for first, second, result in compositions:
                output.write(struct.pack("<III", first, second, result))
            file_size = output.tell()
            output.seek(0)
            output.write(HEADER.pack(
                MAGIC, VERSION, HEADER.size, file_size,
                MODEL_VOCAB_SIZE, base_count, decodable_count,
                len(merges), len(added), len(ranges), len(combining),
                len(decompositions), len(compositions),
            ))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, args.output)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

    print(
        f"wrote {args.output} ({args.output.stat().st_size / 2**20:.2f} MiB, "
        f"{base_count} base tokens, {len(merges)} merges, {len(added)} added tokens, "
        "Unicode sections pending)"
    )


if __name__ == "__main__":
    main()
