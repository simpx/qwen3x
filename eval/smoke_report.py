#!/usr/bin/env python3
"""Report the latest completed 4B and 9B official-recipe smoke runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = ("mmlu_pro", "ceval", "ifeval")
OFFICIAL_SCORES = {
    "4b": {"mmlu_pro": 0.791, "ceval": 0.851, "ifeval": 0.898},
    "9b": {"mmlu_pro": 0.825, "ceval": 0.882, "ifeval": 0.915},
}
OFFICIAL_MODEL_CARD = "https://huggingface.co/Qwen/Qwen3.5-4B#benchmark-results"
SMOKE_SUBSETS = {
    "mmlu_pro": ["computer science", "math"],
    "ceval": ["computer_network", "advanced_mathematics"],
    "ifeval": ["default"],
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_runs(root: Path, size: str) -> dict[str, tuple[Path, dict]]:
    runs = {}
    for path in root.glob(f"{size}/*/qwen35-manifest.json"):
        manifest = read_json(path)
        if manifest.get("result") != "completed":
            continue
        contract = manifest.get("run_contract") or {}
        dataset = contract.get("dataset")
        if dataset not in DATASETS or contract.get("model") != f"qwen3.5-{size}":
            continue
        if contract.get("dataset_args") != {
                dataset: {"subset_list": SMOKE_SUBSETS[dataset]}}:
            continue
        if (manifest.get("prediction_summary") or {}).get("samples") != 2:
            continue
        current = runs.get(dataset)
        if current is None or path.stat().st_mtime > current[0].stat().st_mtime:
            runs[dataset] = (path.parent, manifest)
    return runs


def report_score(run: Path, dataset: str) -> float:
    paths = list(run.glob(f"reports/*/{dataset}.json"))
    if len(paths) != 1:
        raise SystemExit(f"expected one {dataset} report below {run}")
    return float(read_json(paths[0])["score"])


def validate(manifest: dict) -> None:
    contract = manifest["run_contract"]
    generation = contract["generation_config"]
    expected = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "seed": 42,
        "max_tokens": 32768,
    }
    for name, value in expected.items():
        if generation.get(name) != value:
            raise SystemExit(f"non-canonical smoke parameter {name}={generation.get(name)!r}")
    thinking = ((generation.get("extra_body") or {})
                .get("chat_template_kwargs") or {})
    if thinking.get("enable_thinking") is not True:
        raise SystemExit("canonical smoke must enable thinking")
    context = int((contract.get("server") or {}).get("context_size") or 0)
    if context < 65536:
        raise SystemExit(f"canonical smoke context is too small: {context}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parent / "results/smoke")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    found = {size: latest_runs(args.root, size) for size in ("4b", "9b")}
    result = {
        "official_model_card": OFFICIAL_MODEL_CARD,
        "note": (
            "Local scores use six fixed samples; official scores use full datasets. "
            "Qwen does not publish its complete benchmark harness, so this uses "
            "its public generation recommendations with EvalScope prompts. The "
            "delta is a smoke signal, not a statistically comparable estimate."
        ),
        "models": {},
    }
    for size, runs in found.items():
        if len(runs) != len(DATASETS):
            continue
        model = {"datasets": {}}
        for dataset in DATASETS:
            if dataset not in runs:
                continue
            run, manifest = runs[dataset]
            validate(manifest)
            local = report_score(run, dataset)
            official = OFFICIAL_SCORES[size][dataset]
            summary = manifest.get("prediction_summary") or {}
            context = manifest["run_contract"]["server"]["context_size"]
            max_tokens = manifest["run_contract"]["generation_config"]["max_tokens"]
            model["datasets"][dataset] = {
                "samples": summary.get("samples"),
                "score": local,
                "official_full_score": official,
                "delta": local - official,
                "length_limited_samples": summary.get("length_limited_samples"),
                "max_input_tokens": summary.get("max_input_tokens"),
                "max_tokens": max_tokens,
                "session_context": context,
                "context_headroom": (
                    context - summary["max_input_tokens"] - max_tokens
                    if summary.get("max_input_tokens") is not None else None
                ),
                "prediction_input_sha256": manifest.get("prediction_input_sha256"),
                "run": str(run),
            }
        result["models"][size] = model

    if not result["models"]:
        raise SystemExit(f"no completed canonical smoke runs below {args.root}")

    for dataset in DATASETS:
        left = ((result["models"].get("4b") or {}).get("datasets") or {}).get(dataset)
        right = ((result["models"].get("9b") or {}).get("datasets") or {}).get(dataset)
        if left and right and left["prediction_input_sha256"] != right["prediction_input_sha256"]:
            raise SystemExit(f"4B and 9B used different {dataset} samples")

    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    output = args.output or args.root / "latest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")

    print("dataset   local 4B  official 4B  delta     local 9B  official 9B  delta")
    for dataset in DATASETS:
        cells = [f"{dataset:<10}"]
        for size in ("4b", "9b"):
            row = ((result["models"].get(size) or {}).get("datasets") or {}).get(dataset)
            if row:
                cells.append(
                    f"{row['score'] * 100:7.1f}%  "
                    f"{row['official_full_score'] * 100:8.1f}%  "
                    f"{row['delta'] * 100:+6.1f}pp"
                )
            else:
                cells.append("      -           -        -")
        print("  ".join(cells))
    print(f"details: {output}")


if __name__ == "__main__":
    main()
