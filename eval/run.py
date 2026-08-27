#!/usr/bin/env python3
"""Run reproducible EvalScope benchmarks against the local OpenAI server."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
import urllib.request
from importlib.metadata import version
from pathlib import Path

from evalscope import TaskConfig, run_task


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent / "qwen35-0.8b"
SUPPORTED_DATASETS = ("mmlu_pro", "ceval", "ifeval")
DATASET_SOURCES = {
    "mmlu_pro": {"hub": "modelscope", "id": "TIGER-Lab/MMLU-Pro", "revision": "master"},
    "ceval": {"hub": "modelscope", "id": "evalscope/ceval", "revision": "master"},
    "ifeval": {"hub": "modelscope", "id": "opencompass/ifeval", "revision": "master"},
}
DATASET_SUBSETS = {"mmlu_pro": 14, "ceval": 52, "ifeval": 1}
DATASET_TOTAL_SAMPLES = {"mmlu_pro": 12032, "ceval": 1346, "ifeval": 541}
DATASET_MIN_SUBSET_SAMPLES = {"mmlu_pro": 381, "ceval": 12, "ifeval": 541}
SMOKE_SUBSET_SAMPLES = {"mmlu_pro": 410, "ceval": 19, "ifeval": 541}
LENGTH_FINISH_REASONS = {"length", "max_length", "max_tokens", "model_length"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=PROJECT, text=True
    ).strip()


def hugging_face_revision(model: Path) -> str | None:
    metadata = model / ".cache/huggingface/download/config.json.metadata"
    if not metadata.exists():
        return None
    revision = metadata.read_text(encoding="utf-8").splitlines()[0].strip()
    return revision or None


def tokenizer_fingerprint(model: Path) -> dict:
    files = ("config.json", "tokenizer.json", "tokenizer_config.json")
    return {
        "hugging_face_revision": hugging_face_revision(model),
        "sha256": {
            name: sha256(model / name)
            for name in files
            if (model / name).is_file()
        },
    }


def semantic_server_contract(backend: str, server_info: dict) -> dict:
    """Keep result-affecting server properties; omit scheduling-only knobs."""

    keys = {
        "qwen35": ("compute", "context_size"),
        "transformers": ("device", "dtype", "max_context_tokens"),
    }[backend]
    return {key: server_info[key] for key in keys if key in server_info}


def previous_server_contract(previous: dict, backend: str, fallback: dict) -> dict:
    """Recover semantic provenance from manifests created before it was contractual."""

    merged = {}
    for attempt in previous.get("attempts", []):
        candidate = semantic_server_contract(
            backend, attempt.get("server_info") or {}
        )
        for key, value in candidate.items():
            if key in merged and merged[key] != value:
                raise SystemExit(
                    "old resume attempts used different semantic server "
                    f"configurations at {key}"
                )
            merged[key] = value
    return merged or fallback


def sampling_parameters(thinking: bool, smoke: bool) -> dict:
    """Return the sampling recipe used by Qwen's benchmark table."""

    if smoke:
        return {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "presence_penalty": 0.0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
        }
    # The Language benchmark table has one explicit experimental-settings
    # footnote for both thinking and non-thinking rows.  Do not replace this
    # with the mode-specific Best Practices meant for general API generation.
    return {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }


def run_directory_name(
    stamp: str, dataset: str, thinking: bool, smoke: bool, seed: int
) -> str:
    mode = "thinking" if thinking else "non-thinking"
    scale = "smoke" if smoke else "official"
    return f"{stamp}-{dataset}-{mode}-{scale}-seed{seed}"


def expected_sample_count(
    dataset: str, smoke: bool, limit_per_subset: int | None
) -> int | None:
    """Return an exact expected count when the selected split makes it known."""
    if smoke:
        available = SMOKE_SUBSET_SAMPLES[dataset]
        return available if limit_per_subset is None else min(available, limit_per_subset)
    if limit_per_subset is None:
        return DATASET_TOTAL_SAMPLES[dataset]
    if limit_per_subset <= DATASET_MIN_SUBSET_SAMPLES[dataset]:
        return limit_per_subset * DATASET_SUBSETS[dataset]
    # A larger stratified limit may exhaust only some subsets. Avoid claiming
    # an exact count without storing all per-subset sizes.
    return None


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def mark_manifest_running(manifest: dict) -> None:
    """Start a new attempt without exposing stale completion metadata."""
    manifest["result"] = "running"
    for field in (
        "error",
        "completed_at",
        "report_sha256",
        "prediction_summary",
        "prediction_input_sha256",
        "review_summary",
    ):
        manifest.pop(field, None)


def summarize_predictions(run_dir: Path) -> dict:
    summary = {
        "samples": 0,
        "failed_samples": 0,
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
        "max_output_tokens": 0,
        "length_limited_samples": 0,
        "finish_reasons": {},
        "latency_seconds": 0.0,
    }
    for path in run_dir.glob("predictions/*/*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            output = json.loads(line)["model_output"]
            summary["samples"] += 1
            summary["failed_samples"] += int(output.get("error") is not None)
            usage = output.get("usage") or {}
            output_tokens = int(usage.get("output_tokens") or 0)
            summary["input_tokens"] += int(usage.get("input_tokens") or 0)
            summary["cached_tokens"] += int(usage.get("input_tokens_cache_read") or 0)
            summary["output_tokens"] += output_tokens
            summary["max_output_tokens"] = max(
                summary["max_output_tokens"], output_tokens
            )
            choices = output.get("choices") or []
            reason = choices[0].get("stop_reason") if choices else None
            reason = reason or "unknown"
            summary["finish_reasons"][reason] = (
                summary["finish_reasons"].get(reason, 0) + 1
            )
            summary["length_limited_samples"] += int(
                reason in LENGTH_FINISH_REASONS
            )
            summary["latency_seconds"] += float(output.get("time") or 0)
    summary["cache_hit_rate"] = (
        summary["cached_tokens"] / summary["input_tokens"]
        if summary["input_tokens"] else 0.0
    )
    return summary


def prediction_input_sha256(run_dir: Path) -> str:
    """Hash the exact evaluated prompts, sample ids, and scoring metadata."""
    samples = {}
    for path in run_dir.glob("predictions/*/*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            item = json.loads(line)
            messages = item.get("messages", [])
            choices = (item.get("model_output", {}).get("choices") or [])
            generated = None
            if choices:
                generated = (choices[0].get("message") or {}).get("content")
            if (messages and messages[-1].get("role") == "assistant"
                    and messages[-1].get("content") == generated):
                messages = messages[:-1]
            stable_messages = [
                {"role": message.get("role"), "content": message.get("content")}
                for message in messages
            ]
            sample = {
                "prediction_file": path.name,
                "index": item.get("index"),
                "messages": stable_messages,
                "metadata": item.get("metadata"),
            }
            key = (path.name, int(item["index"]))
            if key in samples:
                raise RuntimeError(
                    f"duplicate prediction sample {path.name} index={item['index']}"
                )
            samples[key] = sample

    # Concurrent requests finish out of order.  The fingerprint describes the
    # sample set, so normalize by stable sample id rather than JSONL line order.
    digest = hashlib.sha256()
    for key in sorted(samples):
        sample = samples[key]
        digest.update(json.dumps(
            sample, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def summarize_reviews(run_dir: Path) -> dict:
    samples = 0
    unscored = 0
    for path in run_dir.glob("reviews/*/*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            item = json.loads(line)
            samples += 1
            score = (item.get("sample_score") or {}).get("score") or {}
            value = score.get("value")
            if value is None or value == {}:
                unscored += 1
    return {
        "samples": samples,
        "scored_samples": samples - unscored,
        "unscored_samples": unscored,
    }


def completion_errors(
    predictions: dict,
    reviews: dict,
    report_count: int = 1,
    expected_samples: int | None = None,
) -> list[str]:
    """Return reasons why an EvalScope run is not a complete scored result."""
    errors = []
    if report_count == 0:
        errors.append("no JSON report was saved")
    if predictions["samples"] == 0:
        errors.append("no predictions were saved")
    if (expected_samples is not None
            and predictions["samples"] != expected_samples):
        errors.append(
            "unexpected prediction count: "
            f"expected={expected_samples} actual={predictions['samples']}"
        )
    if predictions["failed_samples"]:
        errors.append(f"failed predictions={predictions['failed_samples']}")
    if reviews["samples"] != predictions["samples"]:
        errors.append(
            "prediction/review count mismatch: "
            f"predictions={predictions['samples']} reviews={reviews['samples']}"
        )
    if reviews["unscored_samples"]:
        errors.append(f"unscored reviews={reviews['unscored_samples']}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.5-0.8b")
    parser.add_argument("--tokenizer", type=Path,
                        default=PROJECT.parent / "models/Qwen3.5-0.8B")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS,
                        default="mmlu_pro")
    parser.add_argument("--samples", type=int,
                        help="approximate total samples, stratified across subsets")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="use deterministic greedy sampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--request-timeout", type=float, default=7200,
                        help="HTTP timeout in seconds for one generated response")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="number of simultaneous OpenAI requests")
    parser.add_argument("--backend", choices=("qwen35", "transformers"), default="qwen35")
    parser.add_argument("--artifact", type=Path,
                        default=PROJECT / "build/qwen35-0.8b.bin")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--resume", type=Path,
                        help="resume an existing run directory using EvalScope cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples is not None and args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")
    if args.request_timeout <= 0:
        raise SystemExit("--request-timeout must be positive")

    # Some IFEval rules call nltk.word_tokenize() directly.  EvalScope only
    # downloads punkt_tab lazily from another rule, so the first such sample
    # can otherwise be left unscored.  Make the evaluator dependency ready
    # before any prediction is reviewed.
    if args.dataset == "ifeval":
        from evalscope.utils.resource_utils import check_nltk_data

        check_nltk_data("punkt_tab")

    api_url = args.server.rstrip("/") + "/v1"
    try:
        with urllib.request.urlopen(
            args.server.rstrip("/") + "/readyz", timeout=5
        ) as response:
            server_info = json.load(response)
            if server_info.get("status") != "ready":
                raise RuntimeError("server is not ready")
    except Exception as error:
        raise SystemExit(f"cannot use Qwen server at {args.server}: {error}") from error

    generation = {
        **sampling_parameters(args.thinking, args.smoke),
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "timeout": args.request_timeout,
        # A failed long request must stay visible.  Silent retries can repeat
        # tens of minutes of generation and obscure which attempt produced the
        # saved prediction; resume is the explicit retry mechanism.
        "retries": 1,
        "retry_interval": 0,
        "chat_template_kwargs": {
            "enable_thinking": args.thinking,
            "preserve_thinking": True,
        },
    }
    # EvalScope applies --limit independently to every subset. Restrict smoke
    # mode to one representative subset so LIMIT=20 really means 20 requests
    # per benchmark, rather than hundreds across MMLU-Pro/C-Eval categories.
    dataset_args = {}
    if args.smoke:
        smoke_subsets = {
            "mmlu_pro": {"subset_list": ["computer science"]},
            "ceval": {"subset_list": ["computer_network"]},
            "ifeval": {"subset_list": ["default"]},
        }
        dataset_args = {args.dataset: smoke_subsets[args.dataset]}

    # EvalScope's integer limit applies independently to every subset. Convert
    # the user-facing approximate total into a per-subset count, giving a small
    # but genuinely stratified sample instead of accidentally running N*subsets.
    evalscope_limit = None
    if args.samples is not None:
        subsets = 1 if args.smoke else DATASET_SUBSETS[args.dataset]
        evalscope_limit = math.ceil(args.samples / subsets)
    expected_samples = expected_sample_count(
        args.dataset, args.smoke, evalscope_limit
    )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = (args.resume if args.resume is not None else
               args.output / run_directory_name(
                   stamp, args.dataset, args.thinking, args.smoke, args.seed
               ))
    if args.resume is None:
        run_dir.mkdir(parents=True, exist_ok=False)
    elif not run_dir.is_dir():
        raise SystemExit(f"resume directory does not exist: {run_dir}")
    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--short"))
    manifest_path = run_dir / "qwen35-manifest.json"
    previous = None
    if args.resume is not None:
        if not manifest_path.exists():
            raise SystemExit(f"resume manifest does not exist: {manifest_path}")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))

    run_contract = {
        "dataset": args.dataset,
        "dataset_source": DATASET_SOURCES[args.dataset],
        "evalscope_version": version("evalscope"),
        "dataset_args": dataset_args,
        "model": args.model,
        "backend": args.backend,
        "server": semantic_server_contract(args.backend, server_info),
        "generation_config": generation,
        "transport": {"openai_max_retries": 0},
        "tokenizer": tokenizer_fingerprint(args.tokenizer),
        "backend_artifact": args.artifact.name,
        "backend_artifact_sha256": sha256(args.artifact),
        "chat_template_sha256": sha256(PROJECT / "chat_template.jinja"),
    }
    if previous is not None:
        old_contract = previous.get("run_contract")
        # Older manifests treated request concurrency as semantic. It only
        # controls execution and does not change prompts or generation config.
        if isinstance(old_contract, dict):
            old_contract = dict(old_contract)
            old_contract.pop("eval_batch_size", None)
            old_generation = dict(old_contract.get("generation_config") or {})
            old_generation.setdefault("timeout", args.request_timeout)
            old_generation.setdefault("retries", 1)
            old_generation.setdefault("retry_interval", 0)
            old_contract["generation_config"] = old_generation
            old_contract.setdefault("transport", {"openai_max_retries": 0})
            # Before server provenance became contractual it was still kept in
            # every attempt. Migrate an old run once from this resume's server;
            # all newly created runs compare this field strictly.
            old_contract.setdefault(
                "server",
                previous_server_contract(
                    previous, args.backend, run_contract["server"]
                ),
            )
            # Runs created before tokenizer provenance was added already store
            # their exact prompts.  Migrate them once using the tokenizer given
            # to the resume command; all new runs record it from the start.
            old_contract.setdefault("tokenizer", run_contract["tokenizer"])
        if old_contract != run_contract:
            raise SystemExit(
                "resume configuration differs from qwen35-manifest.json; "
                "use the same server/model/sampling/template/bin settings"
            )

    manifest = previous or {
        "result": "running",
        "engine_commit": commit,
        "engine_dirty": dirty,
        "run_contract": run_contract,
        "attempts": [],
    }
    manifest["run_contract"] = run_contract
    attempt = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requested_samples": args.samples,
        "evalscope_limit_per_subset": evalscope_limit,
        "expected_samples": expected_samples,
        "resumed": args.resume is not None,
        "engine_commit": commit,
        "engine_dirty": dirty,
        "api_url": api_url,
        "server_info": server_info,
        "concurrency": args.concurrency,
    }
    # A resumed attempt supersedes both a previous failure and a previous
    # completion. Keeping stale errors/reports beside result=running makes a
    # live manifest misleading, especially when expanding a cached 112-sample
    # run into a 504-sample run.
    mark_manifest_running(manifest)
    manifest["attempts"].append(attempt)
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)

    config = TaskConfig(
        model=args.model,
        model_id=args.model,
        model_args={"max_retries": 0},
        api_url=api_url,
        api_key="EMPTY",
        eval_type="openai_api",
        datasets=[args.dataset],
        dataset_args=dataset_args,
        limit=evalscope_limit,
        eval_batch_size=args.concurrency,
        generation_config=generation,
        work_dir=str(run_dir),
        use_cache=str(run_dir) if args.resume is not None else None,
        # Predictions are expensive; reviews are cheap.  Recompute reviews on
        # resume so evaluator fixes/resources (for example IFEval punkt_tab)
        # cannot leave a stale empty score in an otherwise valid run.
        rerun_review=args.resume is not None,
        no_timestamp=True,
        enable_progress_tracker=True,
    )
    try:
        run_task(task_cfg=config)
    except BaseException as error:
        manifest["result"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        attempt["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        attempt["result"] = "failed"
        write_manifest(manifest_path, manifest)
        raise
    manifest["report_sha256"] = {
        str(report.relative_to(run_dir)): sha256(report)
        for report in sorted(run_dir.rglob("reports/**/*.json"))
    }
    manifest["prediction_summary"] = summarize_predictions(run_dir)
    # ModelScope exposes these datasets through the mutable name "master".
    # Hashing the exact prompts and scoring metadata makes the concrete sample
    # set immutable and comparable even if that branch later moves.
    manifest["prediction_input_sha256"] = prediction_input_sha256(run_dir)
    manifest["review_summary"] = summarize_reviews(run_dir)
    errors = completion_errors(
        manifest["prediction_summary"],
        manifest["review_summary"],
        len(manifest["report_sha256"]),
        expected_samples,
    )
    if errors:
        message = "; ".join(errors)
        manifest["result"] = "failed"
        manifest["error"] = f"incomplete EvalScope result: {message}"
        attempt["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        attempt["result"] = "failed"
        write_manifest(manifest_path, manifest)
        raise RuntimeError(manifest["error"])

    manifest["result"] = "completed"
    manifest.pop("error", None)
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    attempt["finished_at"] = manifest["completed_at"]
    attempt["result"] = "completed"
    write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
