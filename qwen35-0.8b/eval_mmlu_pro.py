#!/usr/bin/env python3
"""在 qwen35_cuda 上运行 MMLU-Pro 官方 runner 风格的 CoT prompt，并给出抽样 score。

Qwen/Qwen3.5-0.8B 的模型卡报告了 MMLU-Pro non-thinking 29.7。这里逐项复用
TIGER-AI-Lab/MMLU-Pro 的官方 local runner：同一个 5-shot CoT 模板、同一答案抽取正则、
temperature=0（本项目的 greedy decode）。它不是完整官方复现，原因也写在输出中：

* 官方 runner 的 context 是 4096、max_new_tokens=2048；这里同样支持 4096，默认仍以
  512 为保险上限，但 GPU 会在官方 `Question:` stop token 序列处立即停下；
* 同一 category 的 five-shot prefix 会在 GPU prefill 一次，并用 device-to-device state
  clone 供该 category 的全部 test questions 复用；
* 因而默认从固定 seed 抽 20 题；prompt 过长时会少于五个 in-context example。

分数是“official-runner-style sample estimate”，只用来检验是否与 29.7 大致相符，不能替代
官方全量成绩。外围开发依赖：`pip install duckdb huggingface_hub transformers`。
"""

import argparse
import concurrent.futures
import os
import random
import re
import subprocess
import time
from pathlib import Path

import duckdb
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


DATASET = "TIGER-Lab/MMLU-Pro"
REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
LETTERS = "ABCDEFGHIJ"
MAX_CONTEXT = 4096  # 与官方 MMLU-Pro runner 的 context window 相同。


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--engine", default=str(Path(__file__).with_name("qwen35_cuda")))
    parser.add_argument("--limit", type=int, default=20, help="fixed-seed random sample size; 0 means all 12,032")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--category", help="optional MMLU-Pro category; useful for measuring one cached prefix")
    parser.add_argument("--workers", type=int, default=1,
                        help="independent CUDA workers on one GPU; 1 is safest, 5 fit this 16 GiB RTX 4080")
    parser.add_argument("--fast-answer-only", action="store_true",
                        help="short letter-only prompt + one generated token; fast regression, not official MMLU-Pro")
    parser.add_argument("--shortest-per-category", action="store_true",
                        help="with --fast-answer-only, take the shortest inputs at the requested ratio in every category")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="must fit after the prompt inside the CUDA engine's 4096-token limit")
    parser.add_argument("--time-limit-seconds", type=float,
                        help="stop after completing the current question past this wall-time budget")
    # These two flags make the script testable offline and record the exact downloaded parquet revision.
    parser.add_argument("--test-parquet", type=Path)
    parser.add_argument("--validation-parquet", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_dir() or not args.weights.is_file():
        parser.error("checkpoint directory and packed weights file are required")
    if (args.limit < 0 or args.workers <= 0 or args.max_new_tokens <= 0 or args.max_new_tokens >= MAX_CONTEXT or
            args.time_limit_seconds is not None and args.time_limit_seconds <= 0):
        parser.error("invalid --limit or --max-new-tokens")
    if bool(args.test_parquet) != bool(args.validation_parquet):
        parser.error("pass both --test-parquet and --validation-parquet, or neither")
    if args.show and args.workers != 1:
        parser.error("--show requires --workers 1 so responses remain ordered")
    if args.shortest_per_category and not args.fast_answer_only:
        parser.error("--shortest-per-category is only an end-to-end coverage option for --fast-answer-only")
    return args


def parquet_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.test_parquet:
        return args.test_parquet, args.validation_parquet
    common = {"repo_id": DATASET, "repo_type": "dataset", "revision": REVISION}
    return (
        Path(hf_hub_download(filename="data/test-00000-of-00001.parquet", **common)),
        Path(hf_hub_download(filename="data/validation-00000-of-00001.parquet", **common)),
    )


def read_rows(path: Path) -> list[dict]:
    # DuckDB reads the official parquet directly and keeps this course repo independent of a
    # particular datasets/pyarrow pairing. Nothing is copied into the repository.
    columns = "question, options, answer, answer_index, cot_content, category"
    cursor = duckdb.connect().execute(f"SELECT {columns} FROM read_parquet(?)", [str(path)])
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def format_example(row: dict, include_answer: bool) -> str:
    text = "Question:\n" + row["question"] + "\nOptions:\n"
    text += "".join(f"{LETTERS[i]}. {option}\n" for i, option in enumerate(row["options"]))
    if include_answer:
        # Exact replacement used in TIGER-AI-Lab/MMLU-Pro/evaluate_from_local.py.
        cot = row["cot_content"].replace("A: Let's think step by step.",
                                         "Answer: Let's think step by step.")
        return text + cot + "\n\n"
    return text + "Answer: Let's think step by step."


def make_prompt_parts(row: dict, validation: list[dict], tokenizer,
                      max_new_tokens: int) -> tuple[list[int], list[int], int]:
    same_subject = [example for example in validation if example["category"] == row["category"]]
    # The official runner starts with k=5 and lowers it if its 4096-token window would overflow.
    # We repeat that algorithm with the same 4096-token window and return the actual k for logs.
    for k in range(min(5, len(same_subject)), -1, -1):
        prefix = (
            "The following are multiple choice questions (with answers) about " + row["category"] +
            '. Think step by step and then finish your answer with "the answer is (X)" '
            "where X is the correct letter choice.\n"
        )
        prefix += "".join(format_example(example, True) for example in same_subject[:k])
        suffix = format_example(row, False)
        # Tokenize the concatenation first: that is the exact official prompt. Then prove the prefix
        # has a clean token boundary before putting its recurrent/KV state into the GPU cache.
        ids = tokenizer(prefix + suffix)["input_ids"]
        prefix_ids = tokenizer(prefix)["input_ids"]
        if ids[:len(prefix_ids)] != prefix_ids:
            raise RuntimeError("MMLU-Pro prefix ends inside a tokenizer merge; cannot safely cache it")
        if len(ids) + max_new_tokens <= MAX_CONTEXT:
            return prefix_ids, ids[len(prefix_ids):], k
    raise RuntimeError("even the zero-shot MMLU-Pro prompt exceeds 4096 tokens")


def make_fast_prompt_parts(row: dict, tokenizer) -> tuple[list[int], list[int], int]:
    """A deliberately separate, short quality-regression prompt for large local samples.

    This is Qwen's text chat framing with ``enable_thinking=False``.  Without the empty
    ``<think>`` block Qwen3.5 starts a reasoning turn, so the one-token completion would only
    decode ``<think>`` instead of a choice letter.  It belongs in this Python evaluator, not
    in the small C++ inference lesson.
    """
    prefix = ("<|im_start|>user\n"
              "Answer the multiple-choice question with only its correct letter.\n\n")
    suffix = "Question:\n" + row["question"] + "\nOptions:\n"
    suffix += "".join(f"{LETTERS[index]}. {option}\n" for index, option in enumerate(row["options"]))
    suffix += ("Answer:<|im_end|>\n<|im_start|>assistant\n"
               "<think>\n\n</think>\n\n")
    ids = tokenizer(prefix + suffix)["input_ids"]
    prefix_ids = tokenizer(prefix)["input_ids"]
    if ids[:len(prefix_ids)] != prefix_ids:
        raise RuntimeError("fast MMLU-Pro prefix ends inside a tokenizer merge")
    return prefix_ids, ids[len(prefix_ids):], 0


def shortest_per_category_sample(rows: list[dict], count: int, tokenizer) -> list[dict]:
    """Return exactly ``count`` short prompts while retaining each MMLU-Pro category.

    This is intentionally a throughput/coverage suite rather than a representative benchmark:
    selecting shorter questions means 30% of all cases can finish on a small local GPU in about
    an hour.  The category quotas follow the original data proportions, and the output log says
    plainly that its accuracy is not a published MMLU-Pro number.
    """
    if count >= len(rows):
        return rows
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["category"], []).append(row)
    ratio = count / len(rows)
    quotas = {category: int(len(items) * ratio) for category, items in grouped.items()}
    remainder = count - sum(quotas.values())
    # Largest fractional remainders give an exact total and keep the selection proportional.
    order = sorted(grouped, key=lambda category: (len(grouped[category]) * ratio - quotas[category], category),
                   reverse=True)
    for category in order[:remainder]:
        quotas[category] += 1
    chosen = []
    for category, items in grouped.items():
        def length(row: dict) -> int:
            prefix_ids, suffix_ids, _ = make_fast_prompt_parts(row, tokenizer)
            return len(prefix_ids) + len(suffix_ids)
        chosen.extend(sorted(items, key=length)[:quotas[category]])
    assert len(chosen) == count
    return chosen


def extract_answer(text: str) -> str | None:
    # These are the three official MMLU-Pro extraction levels, intentionally in the same order.
    match = re.search(r"answer is \(?([A-J])\)?", text)
    if match:
        return match.group(1)
    match = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if match:
        return match.group(1)
    found = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return found.group(0) if found else None


class CudaWorker:
    """让模型/权重在 GPU 上跨题常驻；stdin 边界仍只有 token ids。"""

    def __init__(self, engine: str, weights: Path):
        env = os.environ.copy()
        wsl_driver_dir = "/usr/lib/wsl/lib"
        if Path(wsl_driver_dir, "libcuda.so.1").exists():
            env["LD_LIBRARY_PATH"] = wsl_driver_dir + ":" + env.get("LD_LIBRARY_PATH", "")
        self.process = subprocess.Popen(
            [engine, "--serve", str(weights)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, env=env,
        )

    def generate(self, ids: list[int], count: int) -> list[int]:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(f"{count}\t{','.join(map(str, ids))}\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline().strip()
        if not line.startswith("generated:"):
            raise RuntimeError(f"CUDA worker failed: {line!r}")
        return [int(word) for word in line[len("generated:"):].split()]

    def cache(self, prefix_ids: list[int]) -> None:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(f"cache\t{','.join(map(str, prefix_ids))}\n")
        self.process.stdin.flush()
        if self.process.stdout.readline().strip() != "ready":
            raise RuntimeError("CUDA worker could not cache MMLU-Pro prefix")

    def generate_from_cache(self, suffix_ids: list[int], count: int,
                            stop_sequences: list[list[int]]) -> list[int]:
        assert self.process.stdin is not None and self.process.stdout is not None
        stops = ";".join(",".join(map(str, sequence)) for sequence in stop_sequences) or "-"
        self.process.stdin.write(
            f"generate\t{count}\t{stops}\t{','.join(map(str, suffix_ids))}\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline().strip()
        if not line.startswith("generated:"):
            raise RuntimeError(f"CUDA cached worker failed: {line!r}")
        return [int(word) for word in line[len("generated:"):].split()]

    def close(self) -> None:
        if self.process.stdin:
            self.process.stdin.write("quit\n")
            self.process.stdin.flush()
            self.process.stdin.close()
        if self.process.wait(timeout=10):
            raise RuntimeError("CUDA worker exited with an error")


def generate(tokenizer, worker: CudaWorker, ids: list[int], count: int) -> str:
    output_ids = worker.generate(ids, count)
    # Official vLLM uses stop=["Question:"]. qwen35 itself does not know text tokens, so cut the
    # decoded continuation at the same literal stop string before applying the official extractor.
    return tokenizer.decode(output_ids, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False).split("Question:", 1)[0]


def question_stop_sequences(tokenizer) -> list[list[int]]:
    # BPE may encode Question: differently after one or two newlines. Sending all three exact
    # possibilities gives the GPU a tokenizer-aware equivalent of vLLM's stop=["Question:"].
    found = []
    for text in ("Question:", "\nQuestion:", "\n\nQuestion:"):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids and ids not in found:
            found.append(ids)
    return found


def run_partition(engine: str, weights: Path, groups, stops: list[list[int]], count: int,
                  deadline: float | None) -> list[tuple[int, dict, int, list[int]]]:
    """Run several prefix groups in one CUDA process; one process owns one GPU context."""
    worker = CudaWorker(engine, weights)
    output = []
    try:
        for prefix_ids, items in groups:
            if deadline is not None and time.monotonic() >= deadline:
                break
            worker.cache(list(prefix_ids))
            for suffix_ids, shots, row, original_number in items:
                if deadline is not None and output and time.monotonic() >= deadline:
                    return output
                output.append((original_number, row, shots,
                               worker.generate_from_cache(suffix_ids, count, stops)))
    finally:
        worker.close()
    return output


def main() -> None:
    args = arguments()
    test_path, validation_path = parquet_paths(args)
    test, validation = read_rows(test_path), read_rows(validation_path)
    if args.category:
        test = [row for row in test if row["category"] == args.category]
        if not test:
            raise RuntimeError(f"MMLU-Pro has no test rows for category {args.category!r}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if args.limit == 0:
        selected = test
    elif args.shortest_per_category:
        selected = shortest_per_category_sample(test, min(args.limit, len(test)), tokenizer)
    else:
        selected = random.Random(args.seed).sample(test, min(args.limit, len(test)))
    prepared = []
    # The non-thinking chat frame makes Qwen emit the choice letter as its *first* completion
    # token.  Decode exactly that one token: continuing for three more does not affect scoring,
    # but it costs three complete model forwards per question.
    generation_count = 1 if args.fast_answer_only else args.max_new_tokens
    for original_number, row in enumerate(selected, 1):
        if args.fast_answer_only:
            prefix_ids, suffix_ids, shots = make_fast_prompt_parts(row, tokenizer)
        else:
            prefix_ids, suffix_ids, shots = make_prompt_parts(row, validation, tokenizer, generation_count)
        prepared.append((tuple(prefix_ids), suffix_ids, shots, row, original_number))
    prepared.sort(key=lambda item: item[0])

    # Build groups that each need exactly one cache prefill, then balance whole groups across CUDA
    # processes. Keeping a category inside one worker preserves cache reuse and avoids cache IPC.
    groups = []
    for prefix_ids, suffix_ids, shots, row, original_number in prepared:
        if not groups or groups[-1][0] != prefix_ids:
            groups.append((prefix_ids, []))
        groups[-1][1].append((suffix_ids, shots, row, original_number))
    # The fast mode has one tiny shared prefix. Duplicate that cache into every worker so all
    # independent CUDA contexts can share the 3,610-question sample instead of leaving workers
    # idle behind one giant group.
    if args.fast_answer_only and args.workers > 1 and len(groups) == 1:
        prefix_ids, items = groups[0]
        groups = [(prefix_ids, items[index::args.workers]) for index in range(args.workers)
                  if items[index::args.workers]]
    worker_count = min(args.workers, len(groups))
    partitions = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    for group in sorted(groups, key=lambda group: len(group[1]), reverse=True):
        index = min(range(worker_count), key=lambda item: loads[item])
        partitions[index].append(group)
        loads[index] += len(group[1])

    stops = [] if args.fast_answer_only else question_stop_sequences(tokenizer)
    started = time.monotonic()
    deadline = None if args.time_limit_seconds is None else started + args.time_limit_seconds
    mode = "fast-answer-only (not official)" if args.fast_answer_only else "official-runner-style"
    selection = "shortest-input, category-stratified coverage" if args.shortest_per_category else "fixed-seed random"
    print(f"MMLU-Pro {mode} sample: {len(selected)}/{len(test)}, "
          f"selection={selection}, cached CUDA workers={worker_count}, prefix groups={len(groups)}")
    if worker_count == 1:
        records = run_partition(args.engine, args.weights, partitions[0], stops,
                                generation_count, deadline)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(run_partition, args.engine, args.weights, partition, stops,
                                       generation_count, deadline) for partition in partitions]
            records = [record for future in futures for record in future.result()]
    records.sort(key=lambda item: item[0])

    correct = invalid = 0
    shot_counts = []
    for completed, (original_number, row, shots, output_ids) in enumerate(records, 1):
        response = tokenizer.decode(output_ids, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False).split("Question:", 1)[0]
        prediction = extract_answer(response)
        # The official code randomly chooses a label on extraction failure; retain that rule and
        # count it separately so a score is never mistaken for perfect format compliance.
        if prediction is None:
            invalid += 1
            prediction = random.Random(args.seed + original_number).choice(LETTERS[:len(row["options"])])
        correct += prediction == row["answer"]
        shot_counts.append(shots)
        if args.show:
            print(f"{completed:3}: {row['category']}: predicted={prediction} expected={row['answer']} "
                  f"shots={shots} response={response!r}")
        elif completed % 10 == 0 or completed == len(records):
            print(f"{completed:3}/{len(selected)}: running accuracy = {correct / completed:.1%}")
    elapsed = time.monotonic() - started
    print()
    if not records:
        print("no question completed before the time limit")
        return
    print(f"MMLU-Pro {mode} accuracy: {correct}/{len(records)} = {correct / len(records):.1%}")
    if args.fast_answer_only:
        print("fast-answer-only uses the real questions but a different prompt; do not compare it to 29.7%.")
        if args.shortest_per_category:
            print("the short-input selection is for >=30% end-to-end coverage within one hour, not quality ranking.")
    else:
        print("official model-card score (full 12,032 questions): 29.7%")
    print(f"invalid responses: {invalid}; average in-context examples: {sum(shot_counts) / len(records):.2f}; "
          f"elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
