#!/usr/bin/env python3
"""Warm/repeated C ABI runtime benchmark; the shared library is a test adapter.

Example (run alone, with no inference server using the GPU):
  caffeinate -i python3 scripts/bench_session.py --library build/metal/libqwen3x-metal.dylib \
    --model build/qwen35-9b-q8_0-model.bin --output build/bench-9b.json
"""

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference"))
from qwen3x import Engine, set_log_callback  # noqa: E402


def mac_memory():
    """Read Darwin rusage_info_v4 (sys/resource.h), without a helper process."""
    class Usage(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16), ("values", ctypes.c_uint64 * 35)]
    usage = Usage()
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    library.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    library.proc_pid_rusage.restype = ctypes.c_int
    if library.proc_pid_rusage(os.getpid(), 4, ctypes.byref(usage)):
        raise OSError(ctypes.get_errno(), "proc_pid_rusage")
    return {"phys_footprint_bytes": usage.values[7],
            "peak_phys_footprint_bytes": usage.values[28]}


def system_snapshot():
    if sys.platform != "darwin":
        return {}
    raw = subprocess.check_output(["sysctl", "vm.swapusage", "kern.memorystatus_vm_pressure_level"], text=True)
    return {"swap_used_bytes": int(float(re.search(r"used = ([0-9.]+)M", raw)[1]) * 1024**2),
            "pressure_level": int(re.search(r"vm_pressure_level: (\d+)", raw)[1]), "raw": raw.strip()}


def metal_load_fields(message):
    match = re.fullmatch(
        r"Metal ready device=(.*?) weights=(\d+) allocated=(\d+) "
        r"max_buffer=(\d+) recommended_working_set=(\d+)",
        message,
    )
    if not match:
        return {}
    return {"metal_device": match[1], "model_weight_bytes": int(match[2]),
            "metal_allocated_after_load_bytes": int(match[3]),
            "metal_max_buffer_bytes": int(match[4]),
            "metal_recommended_working_set_bytes": int(match[5])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", type=int, nargs="+", default=[128, 512, 4096])
    parser.add_argument("--decode", type=int, default=128)
    parser.add_argument("--context", type=int, default=40960)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.decode < 1 or min(args.prompts) < 1:
        parser.error("counts must be positive")
    if max(args.prompts) + args.decode + 1 > args.context:
        parser.error("prompt + first decode + measured decode exceeds context")
    if args.output.exists():
        parser.error("output already exists; choose a new evidence file")
    logs = []
    set_log_callback(args.library, lambda level, file, line, message: logs.append(message))
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
              "library": str(args.library.resolve()), "model": str(args.model.resolve()),
              "context": args.context, "session_slots": 1, "repeats": args.repeats,
              "warmup_per_prompt": 1, "decode_tokens": args.decode,
              "ttft_definition": "sync to full logits; no sampling, HTTP or checkpoint",
              "decode_definition": "eval calls after one excluded first eval",
              "system_before": system_snapshot(), "cases": [], "status": "running"}
    report["source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in ("runtime.cpp", "engine.cpp", "arch/cpu.cpp", "arch/cpu.h",
                                            "arch/arm/kernels.h", "arch/x86/kernels.h",
                                            "arch/apple/parallel.h", "arch/metal/engine.mm",
                                            "arch/metal/kernels.metal", "scripts/bench_session.py")}
    pressure = {report["system_before"].get("pressure_level")}
    stopped = threading.Event()

    def monitor():
        while not stopped.wait(1):
            try:
                pressure.add(system_snapshot().get("pressure_level"))
            except (OSError, subprocess.SubprocessError):
                pressure.add("read_failed")

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    engine = session = None
    try:
        started = time.perf_counter()
        engine = Engine(args.library, args.model)
        report["load_seconds"] = time.perf_counter() - started
        report["native_load_logs"] = logs[:]
        for message in logs:
            report.update(metal_load_fields(message))
        load_log_count = len(logs)
        started = time.perf_counter()
        session = engine.create_session(args.context)
        report["session_create_seconds"] = time.perf_counter() - started
        report["native_state_logs"] = logs[load_log_count:]
        for message in report["native_state_logs"]:
            match = re.search(
                r"Metal state model=(.*?) context=(\d+) bytes=(\d+) allocated=(\d+)", message,
            )
            if match:
                report["state_model"] = match[1]
                report["state_context"] = int(match[2])
                report["state_bytes"] = int(match[3])
                report["metal_allocated_after_session_bytes"] = int(match[4])
        save()
        for count in args.prompts:
            tokens = [100 + i % 1000 for i in range(count)]
            case = {"prompt_tokens": count, "runs": []}
            report["cases"].append(case)
            for index in range(args.repeats + 1):
                session.reset()
                started = time.perf_counter()
                cached = session.sync(tokens, checkpoint_at=-1)
                ready = time.perf_counter()
                assert cached == 0, "warm benchmark must not reuse the prefix cache"
                session.eval(100)  # First decode is excluded from sustained throughput.
                decode_start = time.perf_counter()
                for _ in range(args.decode):
                    session.eval(100)
                finished = time.perf_counter()
                run = {"index": index, "warmup": index == 0, "cached_tokens": cached,
                       "prefill_seconds": ready - started, "ttft_seconds": ready - started,
                       "prefill_tps": count / (ready - started),
                       "first_decode_seconds": decode_start - ready,
                       "decode_seconds": finished - decode_start,
                       "decode_tps": args.decode / (finished - decode_start)}
                if sys.platform == "darwin":
                    run.update(mac_memory())
                case["runs"].append(run)
                print(json.dumps({"prompt": count, **run}), flush=True)
                save()
            case["median"] = {key: statistics.median(run[key] for run in case["runs"][1:])
                              for key in ("prefill_seconds", "prefill_tps", "ttft_seconds",
                                          "decode_seconds", "decode_tps")}
            save()
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = str(error)
        raise
    finally:
        stopped.set()
        monitor_thread.join()
        report["system_after"] = system_snapshot()
        if sys.platform == "darwin":
            report.update(mac_memory())
            report["swap_delta_bytes"] = (report["system_after"]["swap_used_bytes"]
                                          - report["system_before"]["swap_used_bytes"])
        report["pressure_levels_observed"] = sorted(pressure - {None}, key=str)
        report["peak_rss_native_units"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save()
        if session is not None:
            session.close()
        if engine is not None:
            engine.close()
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "medians": [case["median"] for case in report["cases"]]}), flush=True)


if __name__ == "__main__":
    main()
