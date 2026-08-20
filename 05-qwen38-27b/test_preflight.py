#!/usr/bin/env python3
"""Exercise the Stage 5 contract checker with official metadata and three deliberate corruptions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--device-gib", type=float, required=True)
    parser.add_argument("--expect-raw-fits", choices=("yes", "no"), required=True)
    return parser.parse_args()


def invoke(args: argparse.Namespace, config: Path, index: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(args.checker), "--config", str(config), "--index", str(index),
         "--device-gib", str(args.device_gib), "--expect-raw-fits", args.expect_raw_fits],
        text=True,
        capture_output=True,
    )


def require_failure(result: subprocess.CompletedProcess[str], expected_text: str) -> None:
    if result.returncode == 0 or expected_text not in result.stderr:
        raise SystemExit(
            f"corrupt metadata did not fail as expected ({expected_text!r}): "
            f"returncode={result.returncode}, stderr={result.stderr!r}"
        )


def main() -> None:
    args = parse_args()
    args.checker = args.checker.resolve()
    baseline = invoke(args, args.config, args.index)
    if baseline.returncode:
        raise SystemExit("official metadata unexpectedly failed:\n" + baseline.stderr)
    print("official metadata: passed")

    config = json.loads(args.config.read_text())
    index = json.loads(args.index.read_text())
    with tempfile.TemporaryDirectory(prefix="qwen38-preflight-") as directory:
        root = Path(directory)

        bad_config = json.loads(json.dumps(config))
        bad_config["text_config"]["hidden_size"] = 123
        config_path = root / "bad-config.json"
        config_path.write_text(json.dumps(bad_config))
        require_failure(invoke(args, config_path, args.index), "text_config.hidden_size")
        print("wrong hidden_size: rejected")

        bad_schema = json.loads(json.dumps(index))
        shard = bad_schema["weight_map"].pop("lm_head.weight")
        bad_schema["weight_map"]["model.language_model.fake.weight"] = shard
        index_path = root / "bad-schema.json"
        index_path.write_text(json.dumps(bad_schema))
        require_failure(invoke(args, args.config, index_path), "unexpected text-only tensor schema")
        print("wrong text tensor name: rejected")

        bad_size = json.loads(json.dumps(index))
        bad_size["metadata"]["total_size"] -= 1
        index_path = root / "bad-size.json"
        index_path.write_text(json.dumps(bad_size))
        require_failure(invoke(args, args.config, index_path), "unexpected Qwen3.8-27B safetensors layout")
        print("wrong raw byte size: rejected")
    print("Stage 5 preflight tests: passed")


if __name__ == "__main__":
    main()
