#!/usr/bin/env python3
"""Full-model Qwen3.6 CPU oracle and CUDA/Metal comparison smoke."""

import argparse
from array import array
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.qwen3x import Engine


FORMAT = "qwen3x-qwen36-35b-smoke-v1"
PROMPT = "Hello"
PROMPT_TOKENS = [9419]
STEPS = 4
ATOL = 5e-4
PATH_ATOL = 5e-5
CUDA_STATE_BYTES_8192 = 469_547_332
CUDA_WORK_BYTES = 1_249_600


class GpuMemorySampler:
    def __init__(self):
        wsl = Path("/usr/lib/wsl/lib/nvidia-smi")
        self.command = str(wsl) if wsl.exists() else "nvidia-smi"
        self.values = []
        self.stop_event = threading.Event()
        self.thread = None

    def sample(self):
        try:
            output = subprocess.check_output([
                self.command, "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ], text=True, stderr=subprocess.DEVNULL, timeout=2)
            self.values.append(max(int(line.strip()) for line in output.splitlines()))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass

    def __enter__(self):
        self.sample()
        def poll():
            while not self.stop_event.wait(0.2):
                self.sample()
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()
        self.sample()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(actual, expected, label, tolerance):
    if len(actual) != len(expected) or not actual:
        raise AssertionError(f"{label}: logits shape mismatch")
    worst = 0.0
    for token, (left, right) in enumerate(zip(actual, expected)):
        if not math.isfinite(left) or not math.isfinite(right):
            raise AssertionError(f"{label}: non-finite logit token={token}")
        error = abs(left - right)
        if error > tolerance:
            raise AssertionError(
                f"{label}: token={token} actual={left} oracle={right} "
                f"error={error} > {tolerance}"
            )
        worst = max(worst, error)
    actual_argmax = max(range(len(actual)), key=actual.__getitem__)
    expected_argmax = max(range(len(expected)), key=expected.__getitem__)
    if actual_argmax != expected_argmax:
        raise AssertionError(
            f"{label}: argmax actual={actual_argmax} oracle={expected_argmax}"
        )
    return worst


def exercise(engine, oracle=None, context=128):
    rows = []
    greedy = []
    worst = 0.0
    started = time.perf_counter()
    with closing(engine.create_session(context)) as session:
        prefill_started = time.perf_counter()
        if session.sync(PROMPT_TOKENS, checkpoint_at=1) != 0:
            raise AssertionError("fresh prompt unexpectedly reused cache")
        prefill_seconds = time.perf_counter() - prefill_started
        rows.append(array("f", session.copy_logits()))
        decode_seconds = []
        for step in range(STEPS):
            token = session.argmax()
            greedy.append(token)
            decode_started = time.perf_counter()
            session.eval(token)
            decode_seconds.append(time.perf_counter() - decode_started)
            rows.append(array("f", session.copy_logits()))
            print(f"full-model step {step + 1}/{STEPS} token={token}", flush=True)

        if session.sync(PROMPT_TOKENS, checkpoint_at=1) != 1:
            raise AssertionError("prompt checkpoint was not restored")
        worst = max(worst, compare(session.copy_logits(), rows[0],
                                   "checkpoint logits", PATH_ATOL))
        for index, token in enumerate(greedy, 1):
            session.eval(token)
            worst = max(worst, compare(session.copy_logits(), rows[index],
                                       f"restored decode[{index}]", PATH_ATOL))

        full_tokens = PROMPT_TOKENS + greedy
        if session.sync(full_tokens, checkpoint_at=1) != len(full_tokens):
            raise AssertionError("live cache did not cover the full sequence")
        worst = max(worst, compare(session.copy_logits(), rows[-1],
                                   "live cache logits", PATH_ATOL))
        session.reset()
        if session.sync(full_tokens, checkpoint_at=1) != 0:
            raise AssertionError("reset rebuild unexpectedly reused cache")
        worst = max(worst, compare(session.copy_logits(), rows[-1],
                                   "reset rebuild logits", PATH_ATOL))

    if oracle is not None:
        if greedy != oracle["greedy_tokens"]:
            raise AssertionError(
                f"greedy tokens {greedy} != CPU {oracle['greedy_tokens']}"
            )
        for index, row in enumerate(rows):
            worst = max(worst, compare(row, oracle["rows"][index],
                                       f"CPU/CUDA position[{index}]", ATOL))
    timings = {
        "elapsed_seconds": time.perf_counter() - started,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
    }
    return rows, greedy, worst, timings


def write_vectors(directory, metadata, rows):
    directory.mkdir(parents=True, exist_ok=True)
    logits = directory / "logits.f32"
    with logits.open("wb") as file:
        for row in rows:
            row.tofile(file)
    metadata = dict(metadata)
    metadata["logits_sha256"] = sha256(logits)
    (directory / "vectors.json").write_text(json.dumps(metadata, indent=2) + "\n")


def dump(library, model, directory, handoff):
    model_hash = sha256(model)
    with Engine(library, model) as engine:
        rows, greedy, worst, timings = exercise(engine)
        vocab = engine.vocab_size
    metadata = {
        "format": FORMAT,
        "model_sha256": model_hash,
        "model_bytes": model.stat().st_size,
        "library_sha256": sha256(library),
        "platform": platform.platform(),
        "prompt": PROMPT,
        "prompt_tokens": PROMPT_TOKENS,
        "greedy_tokens": greedy,
        "argmax_tokens": [max(range(len(row)), key=row.__getitem__) for row in rows],
        "positions": len(rows),
        "vocab": vocab,
        "context": 128,
        "atol": ATOL,
        "path_atol": PATH_ATOL,
        "generation_command": "make cpu-35b-smoke",
        **timings,
        "internal_path_max_abs_error": worst,
    }
    write_vectors(directory, metadata, rows)
    if handoff:
        handoff.mkdir(parents=True, exist_ok=True)
        shutil.copy2(directory / "vectors.json", handoff / "vectors.json")
        shutil.copy2(directory / "logits.f32", handoff / "logits.f32")
    print(json.dumps(metadata, indent=2))


def load_oracle(directory):
    metadata = json.loads((directory / "vectors.json").read_text())
    if metadata.get("format") != FORMAT:
        raise ValueError("unexpected Qwen3.6 smoke vector format")
    path = directory / "logits.f32"
    if sha256(path) != metadata["logits_sha256"]:
        raise ValueError("CPU logits checksum mismatch")
    expected_bytes = metadata["positions"] * metadata["vocab"] * 4
    if path.stat().st_size != expected_bytes:
        raise ValueError("CPU logits size mismatch")
    rows = []
    with path.open("rb") as file:
        for _ in range(metadata["positions"]):
            row = array("f")
            row.fromfile(file, metadata["vocab"])
            rows.append(row)
    metadata["rows"] = rows
    return metadata


def check(library, model, directory, output, context):
    oracle = load_oracle(directory)
    if model.stat().st_size != oracle["model_bytes"] or sha256(model) != oracle["model_sha256"]:
        raise ValueError("candidate model.bin differs from CPU oracle")
    with GpuMemorySampler() as gpu:
        with Engine(library, model) as engine:
            rows, greedy, worst, timings = exercise(engine, oracle, context)
    report = {
        "result": "passed",
        "format": FORMAT,
        "candidate_library": str(library.resolve()),
        "candidate_library_sha256": sha256(library),
        "platform": platform.platform(),
        "model_sha256": oracle["model_sha256"],
        "greedy_tokens": greedy,
        "argmax_tokens": [max(range(len(row)), key=row.__getitem__) for row in rows],
        "positions": len(rows),
        "context": context,
        "max_abs_error": worst,
        "atol": ATOL,
        "generation_command": "make cuda-35b-smoke",
        **timings,
        "cuda_model_bytes": model.stat().st_size,
        "cuda_state_bytes": CUDA_STATE_BYTES_8192 if context == 8192 else None,
        "cuda_work_bytes": CUDA_WORK_BYTES,
    }
    if gpu.values:
        report["gpu_memory_baseline_mib"] = gpu.values[0]
        report["gpu_memory_peak_mib"] = max(gpu.values)
        report["gpu_memory_peak_delta_mib"] = max(gpu.values) - gpu.values[0]
    output.mkdir(parents=True, exist_ok=True)
    logits = output / "logits.f32"
    with logits.open("wb") as file:
        for row in rows:
            row.tofile(file)
    report["logits_sha256"] = sha256(logits)
    (output / "check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dump", "check"))
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--handoff", type=Path)
    parser.add_argument("--context", type=int, default=128)
    args = parser.parse_args()
    if args.mode == "dump":
        dump(args.library, args.model, args.vectors, args.handoff)
    else:
        if not args.output:
            parser.error("check requires --output")
        check(args.library, args.model, args.vectors, args.output, args.context)


if __name__ == "__main__":
    main()
