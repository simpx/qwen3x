#!/usr/bin/env python3
"""Run stable inspect/review/bugfix fixtures through scripts.agent."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import agent  # noqa: E402


SCENES = ("inspect", "review", "bugfix")
FIXTURES = PROJECT / "eval" / "agent" / "fixtures"


def command_result(command, cwd):
    started = time.monotonic()
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False
    )
    return {
        "command": command,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration_ms": (time.monotonic() - started) * 1000,
    }


def prepare_fixture(scene, destination):
    fixture = FIXTURES / scene
    shutil.copytree(fixture / "repo", destination, dirs_exist_ok=True)
    commands = [
        ["git", "init", "-q"],
        ["git", "config", "user.name", "Agent Eval"],
        ["git", "config", "user.email", "agent-eval@example.invalid"],
        ["git", "add", "."],
        ["git", "commit", "-qm", "fixture baseline"],
    ]
    setup = [command_result(command, destination) for command in commands]
    if any(item["exit_code"] != 0 for item in setup):
        raise RuntimeError("could not initialize fixture git repository")
    if scene == "review":
        applied = command_result(
            ["git", "apply", str(fixture / "change.patch")], destination
        )
        setup.append(applied)
        if applied["exit_code"] != 0:
            raise RuntimeError("could not apply review fixture patch")
    return setup


def git_snapshot(cwd):
    diff = command_result(["git", "diff", "--no-ext-diff"], cwd)
    status = command_result(["git", "status", "--porcelain"], cwd)
    changed_files = []
    for line in status["stdout"].splitlines():
        if len(line) >= 4:
            path = line[3:]
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            changed_files.append(path)
    return {
        "diff": diff["stdout"],
        "status": status["stdout"],
        "changed_files": sorted(changed_files),
    }


def inspect_verdict(answer, trace, rubric, final_snapshot):
    ranges = []
    pattern = re.compile(r"pipeline\.py: lines (\d+)-(\d+) of (\d+)")
    for tool in trace["tools"]:
        if tool["name"] != "read_file" or not isinstance(tool["result"], str):
            continue
        match = pattern.match(tool["result"])
        if match:
            ranges.append(tuple(int(value) for value in match.groups()))
    lower_answer = answer.lower()
    aliases = rubric.get("aliases", {})
    missing_terms = [
        term for term in rubric["required_terms"]
        if not any(
            candidate.lower() in lower_answer
            for candidate in [term, *aliases.get(term, [])]
        )
    ]
    missing_lines = []
    for symbol, line in rubric.get("expected_lines", {}).items():
        symbol_patterns = [
            re.escape(candidate.lower())
            for candidate in [symbol, *aliases.get(symbol, [])]
        ]
        line_pattern = rf"(?:lines?\s+|:){line}\b"
        if not any(
            re.search(
                rf"{symbol_pattern}.{{0,100}}{line_pattern}", lower_answer,
                re.DOTALL,
            ) or re.search(
                rf"{line_pattern}.{{0,100}}{symbol_pattern}", lower_answer,
                re.DOTALL,
            )
            for symbol_pattern in symbol_patterns
        ):
            missing_lines.append({"symbol": symbol, "line": line})
    read_to_end = any(end == total for _, end, total in ranges)
    checks = {
        "minimum_successful_reads": (
            len(ranges) >= rubric["minimum_successful_reads"]
        ),
        "read_to_end": read_to_end,
        "required_terms": not missing_terms,
        "exact_locations": not missing_lines,
        "working_tree_unchanged": not final_snapshot["changed_files"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "successful_read_ranges": ranges,
        "missing_terms": missing_terms,
        "missing_locations": missing_lines,
    }


def review_verdict(answer, rubric, initial_snapshot, final_snapshot):
    lower_answer = answer.lower()
    found = []
    missing = []
    for issue in rubric["issues"]:
        all_present = all(
            term.lower() in lower_answer for term in issue["all_terms"]
        )
        any_present = any(
            term.lower() in lower_answer for term in issue["any_terms"]
        )
        (found if all_present and any_present else missing).append(issue["id"])
    checks = {
        "required_location": rubric["required_location"] in answer,
        "all_issues_found": not missing,
        "diff_not_modified": final_snapshot["diff"] == initial_snapshot["diff"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "found_issues": found,
        "missing_issues": missing,
    }


def bugfix_verdict(trace, rubric, initial_test, final_test, final_snapshot):
    tool_names = {item["name"] for item in trace["tools"]}
    checks = {
        "test_failed_before_agent": initial_test["exit_code"] != 0,
        "test_passed_after_agent": final_test["exit_code"] == 0,
        "changed_files_allowed": (
            final_snapshot["changed_files"] == rubric["allowed_changed_files"]
        ),
        "required_tools_used": all(
            name in tool_names for name in rubric["required_tool_names"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "tool_names": sorted(tool_names),
    }


def aggregate_metrics(trace, duration_ms):
    prompt_tokens = 0
    cached_tokens = 0
    completion_tokens = 0
    ttft_ms = []
    backend_metrics = []
    for turn in trace["turns"]:
        usage = turn["response"].get("usage") or {}
        prompt_tokens += usage.get("prompt_tokens", 0)
        completion_tokens += usage.get("completion_tokens", 0)
        details = usage.get("prompt_tokens_details") or {}
        cached_tokens += details.get("cached_tokens", 0)
        if turn["timing"].get("ttft_ms") is not None:
            ttft_ms.append(turn["timing"]["ttft_ms"])
        response_body = turn["response"].get("body") or {}
        if response_body.get("x_qwen3x_eval"):
            backend_metrics.append(response_body["x_qwen3x_eval"])
    tool_errors = [
        item for item in trace["tools"]
        if isinstance(item["result"], str) and item["result"].startswith("ERROR:")
    ]
    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "turn_ttft_ms": ttft_ms,
        "total_duration_ms": duration_ms,
        "tool_error_count": len(tool_errors),
        "backend_turns": backend_metrics,
        "peak_cuda_allocated_bytes": max(
            (item.get("peak_cuda_allocated_bytes", 0)
             for item in backend_metrics),
            default=0,
        ),
        "peak_cuda_reserved_bytes": max(
            (item.get("peak_cuda_reserved_bytes", 0)
             for item in backend_metrics),
            default=0,
        ),
    }


def agent_arguments(options, cwd):
    values = [
        "--cwd", str(cwd), "--yes", "--url", options.url,
        "--model", options.model, "--max-tokens", str(options.max_tokens),
        "--max-turns", str(options.max_turns),
        "--command-timeout", str(options.command_timeout),
        "--request-timeout", str(options.request_timeout),
        "--max-read-lines", str(options.max_read_lines),
    ]
    if options.api_key:
        values.extend(["--api-key", options.api_key])
    return agent.argument_parser().parse_args(values)


def run_scene(scene, options, output):
    fixture = FIXTURES / scene
    prompt = (fixture / "task.txt").read_text(encoding="utf-8").strip()
    rubric = json.loads((fixture / "rubric.json").read_text(encoding="utf-8"))
    scene_output = output / scene
    scene_output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"qwen35-{scene}-") as directory:
        cwd = Path(directory)
        setup = prepare_fixture(scene, cwd)
        initial_snapshot = git_snapshot(cwd)
        initial_test = None
        if scene == "bugfix":
            initial_test = command_result(rubric["test_command"], cwd)

        args = agent_arguments(options, cwd)
        trace = agent.TraceRecorder(args, prompt)
        trace.value["config"].update({
            "backend": options.backend,
            "artifact": options.artifact,
            "source": options.source,
            "revision": options.revision,
        })
        messages = [{"role": "system", "content": agent.SYSTEM_PROMPT}]
        answer = ""
        error = None
        started = time.monotonic()
        try:
            answer = agent.run_task(args, messages, prompt, trace=trace)
            trace.finish("completed", answer=answer)
        except agent.AgentError as caught:
            error = str(caught)
            trace.finish(
                "error", error=error,
                error_class=agent.classify_error(caught),
            )
        duration_ms = (time.monotonic() - started) * 1000

        final_snapshot = git_snapshot(cwd)
        final_test = None
        if scene == "bugfix":
            final_test = command_result(rubric["test_command"], cwd)

        if scene == "inspect":
            verdict = inspect_verdict(
                answer, trace.value, rubric, final_snapshot
            )
        elif scene == "review":
            verdict = review_verdict(
                answer, rubric, initial_snapshot, final_snapshot
            )
        else:
            verdict = bugfix_verdict(
                trace.value, rubric, initial_test, final_test, final_snapshot
            )
        if error is not None:
            verdict["passed"] = False
            verdict["agent_error"] = error

        trace.value["artifacts"] = {
            "final_diff": final_snapshot["diff"],
            "final_status": final_snapshot["status"],
            "initial_test": initial_test,
            "final_test": final_test,
            "verdict": verdict,
        }
        trace.write(scene_output / "trace.json")
        result = {
            "scene": scene,
            "endpoint": options.url,
            "model": options.model,
            "backend": options.backend,
            "artifact": options.artifact,
            "source": options.source,
            "revision": options.revision,
            "answer": answer,
            "error": error,
            "metrics": aggregate_metrics(trace.value, duration_ms),
            "finish_reasons": [
                turn["response"]["finish_reason"]
                for turn in trace.value["turns"]
            ],
            "verdict": verdict,
            "artifacts": {
                "trace": str((scene_output / "trace.json").resolve()),
                "final_diff": final_snapshot["diff"],
                "final_status": final_snapshot["status"],
                "initial_test": initial_test,
                "final_test": final_test,
                "setup": setup,
            },
        }
        (scene_output / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", choices=(*SCENES, "all"))
    parser.add_argument("--url", default=agent.DEFAULT_URL)
    parser.add_argument("--model", default=agent.DEFAULT_MODEL)
    parser.add_argument("--backend", default="qwen35")
    parser.add_argument("--artifact")
    parser.add_argument("--source")
    parser.add_argument("--revision")
    parser.add_argument("--api-key")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--max-read-lines", type=int, default=200)
    parser.add_argument("--command-timeout", type=float, default=30.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--output")
    return parser


def main(argv=None):
    options = argument_parser().parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (Path(options.output) if options.output else
              PROJECT / "eval" / "results" / "agent" / stamp)
    output.mkdir(parents=True, exist_ok=True)
    scenes = SCENES if options.scene == "all" else (options.scene,)
    results = []
    for scene in scenes:
        print(f"agent-eval: running {scene}", file=sys.stderr)
        results.append(run_scene(scene, options, output))
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": options.url,
        "model": options.model,
        "backend": options.backend,
        "artifact": options.artifact,
        "source": options.source,
        "revision": options.revision,
        "results": [
            {
                "scene": item["scene"],
                "passed": item["verdict"]["passed"],
                "error": item["error"],
                "metrics": item["metrics"],
            }
            for item in results
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output.resolve())
    return 0 if all(item["verdict"]["passed"] for item in results) else 1


if __name__ == "__main__":
    sys.exit(main())
