#!/usr/bin/env python3
"""Compare one qwen35 EvalScope run with a Transformers reference run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


OFFICIAL_MODEL_CARD = "https://huggingface.co/Qwen/Qwen3.5-0.8B#benchmark-results"
OFFICIAL_SCORES = {
    False: {"mmlu_pro": 0.297, "ceval": 0.464, "ifeval": 0.521},
    True: {"mmlu_pro": 0.423, "ceval": 0.505, "ifeval": 0.440},
}
PRIMARY_SAMPLE_METRIC = {
    "mmlu_pro": "acc",
    "ceval": "acc",
    "ifeval": "prompt_level_strict",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def predictions(run: Path) -> dict[tuple[str, int], dict]:
    result = {}
    for path in run.glob("predictions/*/*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                item = json.loads(line)
                result[(path.name, int(item["index"]))] = item
    return result


def report(run: Path, dataset: str) -> dict:
    matches = list(run.glob(f"reports/*/{dataset}.json"))
    if len(matches) != 1:
        raise SystemExit(f"expected one {dataset} report below {run}, found {len(matches)}")
    return read_json(matches[0])


def reviews(run: Path) -> dict[tuple[str, int], dict]:
    result = {}
    for path in run.glob("reviews/*/*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                item = json.loads(line)
                result[(path.name, int(item["index"]))] = (
                    item["sample_score"]["score"]["value"]
                )
    return result


def paired_primary(engine: Path, reference: Path, dataset: str) -> dict:
    metric = PRIMARY_SAMPLE_METRIC[dataset]
    engine_rows = reviews(engine)
    reference_rows = reviews(reference)
    if engine_rows.keys() != reference_rows.keys():
        raise SystemExit("runs contain different reviewed sample ids")
    pairs = []
    for key in engine_rows:
        left = engine_rows[key].get(metric)
        right = reference_rows[key].get(metric)
        if left not in (0, 1) or right not in (0, 1):
            raise SystemExit(f"{metric} is not binary at {key[0]} index={key[1]}")
        pairs.append((int(left), int(right)))
    if not pairs:
        raise SystemExit("runs contain no paired reviews")

    both_correct = sum(left and right for left, right in pairs)
    engine_only = sum(left and not right for left, right in pairs)
    reference_only = sum(not left and right for left, right in pairs)
    both_wrong = len(pairs) - both_correct - engine_only - reference_only
    discordant = engine_only + reference_only
    delta = (engine_only - reference_only) / len(pairs)
    variance = max(0.0, discordant / len(pairs) - delta * delta) / len(pairs)
    # Exact two-sided sign test on discordant pairs (McNemar exact test).
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(engine_only, reference_only) + 1)
    )
    exact_p = min(1.0, 2.0 * tail / (2 ** discordant)) if discordant else 1.0
    return {
        "metric": metric,
        "samples": len(pairs),
        "both_correct": both_correct,
        "engine_only_correct": engine_only,
        "reference_only_correct": reference_only,
        "both_wrong": both_wrong,
        "score_delta": delta,
        "paired_standard_error": math.sqrt(variance),
        "mcnemar_exact_p": exact_p,
    }


def content(item: dict) -> str:
    return item["model_output"]["choices"][0]["message"]["content"]


def input_contract(item: dict) -> dict:
    messages = item.get("messages", [])
    if (messages and messages[-1].get("role") == "assistant"
            and messages[-1].get("content") == content(item)):
        messages = messages[:-1]
    return {
        "messages": [
            {"role": message.get("role"), "content": message.get("content")}
            for message in messages
        ],
        "metadata": item.get("metadata"),
    }


def input_sha256(items: dict[tuple[str, int], dict]) -> str:
    digest = hashlib.sha256()
    for key in sorted(items):
        sample = {
            "prediction_file": key[0],
            "index": key[1],
            **input_contract(items[key]),
        }
        digest.update(json.dumps(
            sample, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-exact", action="store_true",
                        help="fail unless every generated response is identical")
    args = parser.parse_args()

    engine_manifest = read_json(args.engine / "qwen35-manifest.json")
    reference_manifest = read_json(args.reference / "qwen35-manifest.json")
    for name, manifest in (("engine", engine_manifest), ("reference", reference_manifest)):
        if manifest.get("result") != "completed":
            raise SystemExit(f"{name} run is not completed")
    engine_contract = engine_manifest["run_contract"]
    reference_contract = reference_manifest["run_contract"]
    dataset = engine_contract["dataset"]
    for field in ("dataset", "dataset_source", "dataset_args", "tokenizer",
                  "generation_config", "transport", "chat_template_sha256"):
        if engine_contract.get(field) != reference_contract.get(field):
            raise SystemExit(f"run contracts differ at {field}")

    engine_predictions = predictions(args.engine)
    reference_predictions = predictions(args.reference)
    engine_keys = set(engine_predictions)
    reference_keys = set(reference_predictions)
    if not engine_keys or not reference_keys:
        raise SystemExit("runs contain no matching sample ids")
    if engine_keys != reference_keys:
        raise SystemExit(
            "runs contain different sample ids: "
            f"engine_only={len(engine_keys - reference_keys)} "
            f"reference_only={len(reference_keys - engine_keys)}"
        )
    shared = sorted(engine_keys)
    input_mismatches = [
        key for key in shared
        if input_contract(engine_predictions[key]) != input_contract(reference_predictions[key])
    ]
    if input_mismatches:
        raise SystemExit(
            "runs contain different prompt/scoring input at "
            f"{input_mismatches[0][0]} index={input_mismatches[0][1]}"
        )
    exact_input_sha256 = input_sha256(engine_predictions)
    exact = sum(
        content(engine_predictions[key]) == content(reference_predictions[key])
        for key in shared
    )
    different = [
        {"prediction_file": key[0], "index": key[1]}
        for key in shared
        if content(engine_predictions[key]) != content(reference_predictions[key])
    ]
    engine_report = report(args.engine, dataset)
    reference_report = report(args.reference, dataset)
    generation = engine_contract["generation_config"]
    template = (generation.get("chat_template_kwargs") or
                (generation.get("extra_body") or {}).get(
                    "chat_template_kwargs"
                ) or {})
    thinking = bool(template.get("enable_thinking"))
    official_score = OFFICIAL_SCORES[thinking][dataset]
    engine_score = engine_report["score"]
    reference_score = reference_report["score"]
    result = {
        "dataset": dataset,
        "matched_samples": len(shared),
        "prediction_input_sha256": exact_input_sha256,
        "exact_outputs": exact,
        "exact_output_rate": exact / len(shared),
        "different_output_samples": different[:20],
        "different_output_samples_truncated": len(different) > 20,
        "engine": {
            "backend": engine_contract["backend"],
            "server": engine_contract.get("server"),
            "score": engine_score,
            "metrics": {item["name"]: item["score"]
                        for item in engine_report["metrics"]},
            "prediction_summary": engine_manifest.get("prediction_summary"),
            "review_summary": engine_manifest.get("review_summary"),
        },
        "reference": {
            "backend": reference_contract["backend"],
            "server": reference_contract.get("server"),
            "score": reference_score,
            "metrics": {item["name"]: item["score"]
                        for item in reference_report["metrics"]},
            "prediction_summary": reference_manifest.get("prediction_summary"),
            "review_summary": reference_manifest.get("review_summary"),
        },
        "score_delta": engine_score - reference_score,
        "paired_primary": paired_primary(args.engine, args.reference, dataset),
        "official_model_card": {
            "source": OFFICIAL_MODEL_CARD,
            "mode": "thinking" if thinking else "non-thinking",
            "score": official_score,
            "engine_delta": engine_score - official_score,
            "reference_delta": reference_score - official_score,
            "note": "Model-card delta may include eval recipe and sampling variance.",
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if args.require_exact and exact != len(shared):
        raise SystemExit("generated outputs differ")


if __name__ == "__main__":
    main()
