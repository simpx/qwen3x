#!/usr/bin/env python3
"""在 qwen35_cuda 上运行 MMLU-Pro 官方 runner 风格的 CoT prompt，并给出抽样 score。

Qwen/Qwen3.5-0.8B 的模型卡报告了 MMLU-Pro non-thinking 29.7。这里逐项复用
TIGER-AI-Lab/MMLU-Pro 的官方 local runner：同一个 5-shot CoT 模板、同一答案抽取正则、
temperature=0（本项目的 greedy decode）。它不是完整官方复现，原因也写在输出中：

* 官方 runner 的 context 是 4096、max_new_tokens=2048；教学程序同样支持 4096 context，
  但没有 tokenizer-aware stop，因此默认只生成 512 token（足以容纳大多数 CoT）；
* 官方测试集共有 12,032 题；本目录的 cuBLAS/persistent-worker CUDA 已避免重复上传，
  但它仍是 batch=1，完整测试不是短小 regression；
* 因而默认从固定 seed 抽 20 题；prompt 过长时会少于五个 in-context example。

分数是“official-runner-style sample estimate”，只用来检验是否与 29.7 大致相符，不能替代
官方全量成绩。外围开发依赖：`pip install duckdb huggingface_hub transformers`。
"""

import argparse
import os
import random
import re
import subprocess
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
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="must fit after the prompt inside the CUDA engine's 4096-token limit")
    # These two flags make the script testable offline and record the exact downloaded parquet revision.
    parser.add_argument("--test-parquet", type=Path)
    parser.add_argument("--validation-parquet", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_dir() or not args.weights.is_file():
        parser.error("checkpoint directory and packed weights file are required")
    if args.limit < 0 or args.max_new_tokens <= 0 or args.max_new_tokens >= MAX_CONTEXT:
        parser.error("invalid --limit or --max-new-tokens")
    if bool(args.test_parquet) != bool(args.validation_parquet):
        parser.error("pass both --test-parquet and --validation-parquet, or neither")
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


def make_prompt(row: dict, validation: list[dict], tokenizer, max_new_tokens: int) -> tuple[list[int], int]:
    same_subject = [example for example in validation if example["category"] == row["category"]]
    # The official runner starts with k=5 and lowers it if its 4096-token window would overflow.
    # We repeat that algorithm with the same 4096-token window and return the actual k for logs.
    for k in range(min(5, len(same_subject)), -1, -1):
        prompt = (
            "The following are multiple choice questions (with answers) about " + row["category"] +
            '. Think step by step and then finish your answer with "the answer is (X)" '
            "where X is the correct letter choice.\n"
        )
        prompt += "".join(format_example(example, True) for example in same_subject[:k])
        prompt += format_example(row, False)
        ids = tokenizer(prompt)["input_ids"]  # Official runner sends this raw CoT prompt, not chat template.
        if len(ids) + max_new_tokens <= MAX_CONTEXT:
            return ids, k
    raise RuntimeError("even the zero-shot MMLU-Pro prompt exceeds 4096 tokens")


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


def main() -> None:
    args = arguments()
    test_path, validation_path = parquet_paths(args)
    test, validation = read_rows(test_path), read_rows(validation_path)
    selected = test if args.limit == 0 else random.Random(args.seed).sample(test, min(args.limit, len(test)))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    correct = invalid = 0
    shot_counts = []
    print(f"MMLU-Pro official-runner-style sample: {len(selected)}/{len(test)}, persistent CUDA worker={args.engine}")
    worker = CudaWorker(args.engine, args.weights)
    try:
        for number, row in enumerate(selected, 1):
            ids, shots = make_prompt(row, validation, tokenizer, args.max_new_tokens)
            response = generate(tokenizer, worker, ids, args.max_new_tokens)
            prediction = extract_answer(response)
            # The official code randomly chooses a label on extraction failure; retain that rule and
            # count it separately so a score is never mistaken for perfect format compliance.
            if prediction is None:
                invalid += 1
                prediction = random.Random(args.seed + number).choice(LETTERS[:len(row["options"])])
            correct += prediction == row["answer"]
            shot_counts.append(shots)
            if args.show:
                print(f"{number:3}: {row['category']}: predicted={prediction} expected={row['answer']} "
                      f"shots={shots} response={response!r}")
            elif number % 10 == 0 or number == len(selected):
                print(f"{number:3}/{len(selected)}: running accuracy = {correct / number:.1%}")
    finally:
        worker.close()
    print()
    print(f"official-runner-style MMLU-Pro sample accuracy: {correct}/{len(selected)} = {correct / len(selected):.1%}")
    print(f"official model-card score (full 12,032 questions): 29.7%")
    print(f"invalid responses: {invalid}; average in-context examples: {sum(shot_counts) / len(shot_counts):.2f}")


if __name__ == "__main__":
    main()
