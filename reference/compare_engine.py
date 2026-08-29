#!/usr/bin/env python3
"""Compare every SIMD Engine step with official Transformers logits vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from qwen35 import Engine


TOP_K = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hugging_face_revision(model: Path) -> str | None:
    metadata = model / ".cache/huggingface/download/config.json.metadata"
    if not metadata.exists():
        return None
    revision = metadata.read_text(encoding="utf-8").splitlines()[0].strip()
    return revision or None


def command_output(*command: str) -> str:
    return subprocess.check_output(command, text=True).strip()


def top_ids(logits: np.ndarray, count: int = TOP_K) -> list[int]:
    candidates = np.argpartition(logits, -count)[-count:]
    ordered = candidates[np.argsort(logits[candidates], kind="stable")][::-1]
    return [int(token) for token in ordered]


def compare_logits(actual_values, expected: np.ndarray, tolerance: float,
                   case: str, step: str) -> float:
    actual = np.asarray(actual_values, dtype=np.float32)
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{case} {step}: logits shape {actual.shape} != {expected.shape}"
        )
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    maximum = float(difference.max())
    if maximum > tolerance:
        token = int(difference.argmax())
        raise AssertionError(
            f"{case} {step}: max_abs_error={maximum:.9g} > {tolerance:.9g} "
            f"at token={token}, C++={actual[token]:.9g}, HF={expected[token]:.9g}"
        )
    actual_argmax = int(actual.argmax())
    expected_argmax = int(expected.argmax())
    if actual_argmax != expected_argmax:
        raise AssertionError(
            f"{case} {step}: argmax C++={actual_argmax}, HF={expected_argmax}"
        )
    actual_top = top_ids(actual)
    expected_top = top_ids(expected)
    if actual_top != expected_top:
        raise AssertionError(
            f"{case} {step}: top-{TOP_K} C++={actual_top}, HF={expected_top}"
        )
    return maximum


def check_case(engine: Engine, vectors, case: dict, tolerance: float) -> float:
    name = case["name"]
    inputs = vectors[f"{name}.input_ids"]
    prefill = vectors[f"{name}.prefill_ids"]
    decode = vectors[f"{name}.decode_ids"]
    expected = vectors[f"{name}.step_logits"]
    greedy = vectors[f"{name}.greedy_ids"]
    if expected.shape != (inputs.size, engine.vocab_size):
        raise AssertionError(
            f"{name}: expected logits shape {expected.shape}, "
            f"wanted {(inputs.size, engine.vocab_size)}"
        )

    worst = 0.0
    session = engine.create_session(max(128, int(inputs.size + greedy.size + 1)))
    try:
        # Fresh token-by-token path: compare all V logits after every token.
        for index, token in enumerate(inputs.tolist()):
            session.eval(int(token))
            worst = max(worst, compare_logits(
                session.copy_logits(), expected[index], tolerance,
                name, f"fresh[{index}]",
            ))

        # Build the prompt through sync so that its boundary becomes a saved
        # checkpoint, then advance the live State through the decode suffix.
        session.reset()
        cached = session.sync(prefill.tolist(), checkpoint_at=int(prefill.size))
        if cached != 0:
            raise AssertionError(f"{name}: initial checkpoint sync reused {cached}")
        worst = max(worst, compare_logits(
            session.copy_logits(), expected[prefill.size - 1], tolerance,
            name, "checkpoint_save",
        ))
        for offset, token in enumerate(decode.tolist()):
            session.eval(int(token))
            worst = max(worst, compare_logits(
                session.copy_logits(), expected[prefill.size + offset], tolerance,
                name, f"decode[{offset}]",
            ))

        # Restore the prompt checkpoint after decode has advanced the live State.
        cached = session.sync(prefill.tolist(), checkpoint_at=int(prefill.size))
        if cached != prefill.size:
            raise AssertionError(
                f"{name}: checkpoint restore reused {cached}, wanted {prefill.size}"
            )
        worst = max(worst, compare_logits(
            session.copy_logits(), expected[prefill.size - 1], tolerance,
            name, "checkpoint_restore",
        ))

        # Append the known decode suffix from that restored checkpoint.
        cached = session.sync(inputs.tolist(), checkpoint_at=int(prefill.size))
        if cached != prefill.size:
            raise AssertionError(
                f"{name}: append sync reused {cached}, wanted {prefill.size}"
            )
        worst = max(worst, compare_logits(
            session.copy_logits(), expected[-1], tolerance, name, "append_sync"
        ))

        # A reset forces a complete rebuild; it must reach identical logits.
        session.reset()
        cached = session.sync(inputs.tolist(), checkpoint_at=int(prefill.size))
        if cached != 0:
            raise AssertionError(f"{name}: fresh rebuild unexpectedly reused {cached}")
        worst = max(worst, compare_logits(
            session.copy_logits(), expected[-1], tolerance, name, "rebuild"
        ))

        # Official and C++ greedy continuation must choose the same tokens.
        session.reset()
        session.sync(prefill.tolist(), checkpoint_at=int(prefill.size))
        for index, expected_token in enumerate(greedy.tolist()):
            actual_token = session.argmax()
            if actual_token != expected_token:
                raise AssertionError(
                    f"{name} greedy[{index}]: C++={actual_token}, HF={expected_token}"
                )
            session.eval(actual_token)
    finally:
        # Closing each large Session immediately also keeps this diagnostic's
        # memory use independent of the number of reference cases.
        session.close()

    print(
        f"reference {name}: steps={inputs.size} greedy={greedy.size} "
        f"max_abs_error={worst:.9g}"
    )
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True,
                        help="directory containing vectors.json and vectors.npz")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--chat-template", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cxxflags", default="")
    args = parser.parse_args()

    metadata = json.loads((args.vectors / "vectors.json").read_text())
    if metadata.get("format") != "qwen3x-hf-vectors":
        raise SystemExit("unsupported Transformers reference format")
    tolerance = float(metadata["comparison_tolerances"]["cpu_max_abs_error"])
    fingerprint = metadata["fingerprint"]
    expected_hashes = {
        "config_sha256": sha256(args.model / "config.json"),
        "safetensors_index_sha256": sha256(args.model / "model.safetensors.index.json"),
        "tokenizer_sha256": sha256(args.model / "tokenizer.json"),
        "chat_template_sha256": sha256(args.chat_template),
    }
    for name, actual in expected_hashes.items():
        if fingerprint.get(name) != actual:
            raise AssertionError(
                f"reference {name}={fingerprint.get(name)!r}, local={actual!r}; "
                "run make -C reference dump"
            )

    worst = 0.0
    with np.load(args.vectors / "vectors.npz", allow_pickle=False) as vectors:
        with Engine(args.library, args.bin) as engine:
            for case in metadata["cases"]:
                worst = max(worst, check_case(engine, vectors, case, tolerance))
    print(
        f"SIMD Engine reference: passed cases={len(metadata['cases'])} "
        f"max_abs_error={worst:.9g} tolerance={tolerance:.9g}"
    )
    model_shards = {
        shard.name: sha256(shard)
        for shard in sorted(args.model.glob("*.safetensors"))
    }
    if fingerprint.get("safetensors_sha256") not in (None, model_shards):
        raise AssertionError(
            "reference safetensors hashes differ; run make -C reference dump"
        )
    revision = hugging_face_revision(args.model)
    if fingerprint.get("hugging_face_revision") not in (None, revision):
        raise AssertionError(
            "reference model revision differs; run make -C reference dump"
        )

    cpu_flags = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("flags"):
                cpu_flags = line.partition(":")[2].strip()
                break
    report = {
        "result": "passed",
        "cases": len(metadata["cases"]),
        "max_abs_error": worst,
        "tolerance": tolerance,
        "reference": metadata,
        "local_inputs": {
            "hugging_face_revision": revision,
            **expected_hashes,
            "safetensors_sha256": model_shards,
            "packed_bin_sha256": sha256(args.bin),
        },
        "engine_build": {
            "commit": command_output(
                "git", "-C", str(Path(__file__).resolve().parents[1]),
                "rev-parse", "HEAD",
            ),
            "dirty": bool(command_output(
                "git", "-C", str(Path(__file__).resolve().parents[1]),
                "status", "--short",
            )),
            "compiler": command_output("c++", "--version").splitlines()[0],
            "cxxflags": args.cxxflags,
            "machine": platform.machine(),
            "cpu_flags": cpu_flags,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"reference provenance: {args.report}")


if __name__ == "__main__":
    main()
