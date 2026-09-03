#!/usr/bin/env python3
"""Portable CPU -> backend correctness vectors, using the exact same model.bin.

No Transformers, tokenizer or sampling: these are engine implementation checks,
not official model-quality scores. Generate on WSL; check on an Apple GPU.
"""

import argparse
from array import array
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.qwen35 import Engine

CASES = [
    {"name": "short", "tokens": [248044, 198, 198, 198, 198], "checkpoint": 3},
    {"name": "chunk_boundary", "tokens": [10, 42, 99, 7, 123, 456, 10, 42, 99, 7], "checkpoint": 8},
    {"name": "chat", "tokens": [248045, 846, 198, 95826, 110827, 97431, 24167, 6738,
                                 1710, 248046, 198, 248045, 74455, 198, 248068, 271,
                                 248069, 271, 198, 198], "checkpoint": 9},
]
ATOL = 5e-4
PATH_ATOL = 5e-5


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(actual, expected, label, tolerance=ATOL):
    if len(actual) != len(expected) or not actual:
        raise AssertionError(f"{label}: logits shape mismatch")
    maximum = 0.0
    actual_top = expected_top = 0
    for i, (a, b) in enumerate(zip(actual, expected)):
        if not math.isfinite(a) or not math.isfinite(b):
            raise AssertionError(f"{label}: non-finite logit at {i}")
        error = abs(a - b)
        if error > tolerance:
            raise AssertionError(f"{label}: token={i} actual={a} CPU={b} error={error} > {tolerance}")
        maximum = max(maximum, error)
        if a > actual[actual_top]:
            actual_top = i
        if b > expected[expected_top]:
            expected_top = i
    if actual_top != expected_top:
        raise AssertionError(f"{label}: argmax {actual_top} != CPU {expected_top}")
    return maximum


def dump(library, model, directory):
    directory.mkdir(parents=True, exist_ok=True)
    model_hash = sha256(model)
    with Engine(library, model) as engine, (directory / "logits.f32").open("wb") as file:
        for case in CASES:
            with closing(engine.create_session(128)) as session:
                for token in case["tokens"]:
                    session.eval(token)
                    row = array("f", session.copy_logits())
                    compare(row, row, case["name"])  # reject a non-finite baseline
                    row.tofile(file)
            print(f"CPU vectors: {case['name']} steps={len(case['tokens'])}", flush=True)
        vocab = engine.vocab_size
    metadata = {"format": "qwen3x-backend-smoke-v1", "oracle": "qwen3x CPU, not official HF",
                "model_sha256": model_hash, "model_bytes": model.stat().st_size,
                "source_library_sha256": sha256(library), "source_platform": platform.platform(),
                "vocab": vocab, "cases": CASES, "atol": ATOL,
                "logits_sha256": sha256(directory / "logits.f32")}
    (directory / "vectors.json").write_text(json.dumps(metadata, indent=2) + "\n")


def check_case(engine, case, rows):
    tokens, checkpoint = case["tokens"], case["checkpoint"]
    worst = 0.0
    with closing(engine.create_session(128)) as session:
        for i, token in enumerate(tokens):
            session.eval(token)
            worst = max(worst, compare(session.copy_logits(), rows[i], f"{case['name']} step[{i}]"))
        fresh = session.copy_logits()
        session.reset()
        if session.sync(tokens, checkpoint_at=checkpoint) != 0:
            raise AssertionError("fresh sync unexpectedly reused tokens")
        compare(session.copy_logits(), fresh, "prefill/decode", PATH_ATOL)
        if session.sync(tokens[:checkpoint], checkpoint_at=checkpoint) != checkpoint:
            raise AssertionError("checkpoint was not restored")
        worst = max(worst, compare(session.copy_logits(), rows[checkpoint - 1], "checkpoint logits"))
        if session.sync(tokens, checkpoint_at=checkpoint) != checkpoint:
            raise AssertionError("checkpoint suffix was not reused")
        compare(session.copy_logits(), fresh, "restored continuation", PATH_ATOL)
        if session.sync(tokens, checkpoint_at=checkpoint) != len(tokens):
            raise AssertionError("live cache did not hit")
        compare(session.copy_logits(), fresh, "cache-only", PATH_ATOL)
        # Another Session must not overwrite this Session's scratch/state/logits.
        with closing(engine.create_session(128)) as other:
            other.eval(tokens[0])
            compare(other.copy_logits(), rows[0], "second Session")
            compare(session.copy_logits(), fresh, "Session isolation", PATH_ATOL)
        session.reset()
        session.sync(tokens, checkpoint_at=-1)
        compare(session.copy_logits(), fresh, "reset/rebuild", PATH_ATOL)
    print(f"backend smoke: {case['name']} passed max_abs_error={worst:.9g}", flush=True)
    return worst


def check(library, model, directory):
    # A failed/interrupted rerun must not leave a previous "passed" report current.
    (directory / "check.json").write_text(json.dumps({"result": "not_completed",
                                                     "candidate_library": str(library.resolve())}) + "\n")
    metadata = json.loads((directory / "vectors.json").read_text())
    if metadata.get("format") != "qwen3x-backend-smoke-v1" or metadata.get("cases") != CASES:
        raise ValueError("unexpected backend smoke format or cases")
    if metadata.get("atol") != ATOL:
        raise ValueError("unexpected smoke tolerance")
    if model.stat().st_size != metadata["model_bytes"] or sha256(model) != metadata["model_sha256"]:
        raise ValueError("model.bin differs from the CPU reference; do not compare different packs")
    path = directory / "logits.f32"
    vocab = metadata["vocab"]
    if vocab <= 0 or path.stat().st_size != sum(len(c["tokens"]) for c in CASES) * vocab * 4:
        raise ValueError("truncated or extra logits data")
    if sha256(path) != metadata["logits_sha256"]:
        raise ValueError("logits checksum mismatch")
    worst = 0.0
    with Engine(library, model) as engine, path.open("rb") as file:
        if engine.vocab_size != vocab:
            raise ValueError("vocabulary mismatch")
        for case in CASES:
            rows = []
            for _ in case["tokens"]:
                row = array("f"); row.fromfile(file, vocab); rows.append(row)
            worst = max(worst, check_case(engine, case, rows))
    report = {"result": "passed", "candidate_library": str(library.resolve()),
              "candidate_library_sha256": sha256(library), "candidate_platform": platform.platform(),
              "model_sha256": metadata["model_sha256"], "vectors_sha256": metadata["logits_sha256"],
              "max_abs_error": worst, "atol": ATOL, "path_atol": PATH_ATOL,
              "cases": len(CASES), "steps": sum(len(c["tokens"]) for c in CASES)}
    (directory / "check.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dump", "check"))
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    args = parser.parse_args()
    if sys.byteorder != "little" or array("f").itemsize != 4:
        parser.error("vectors require little-endian IEEE FP32")
    (dump if args.mode == "dump" else check)(args.library, args.model, args.vectors)


if __name__ == "__main__":
    main()
