#!/usr/bin/env python3
"""用公开 MMLU 选择题算 qwen35 的生成式准确率。

这是最终回答质量的评测，而不是 test_official.py 那种数值 oracle：每道题都经官方
tokenizer/chat template 变成 prompt，C++ executable 贪婪生成一个 A/B/C/D，脚本与标准
答案比较，最后报告 accuracy。它使用本目录的 qwen35_cuda（CUDA）性能 binary。

为保持这个教学项目的边界清楚，这不是 lm-eval-harness 的标准 MMLU 分数。标准协议为
每个候选答案计算 log-likelihood；本程序故意用用户实际会看到的“生成一个选项字母”方式，
所以名称是 *zero-shot generative MMLU accuracy*，不能与排行榜上的 likelihood MMLU
直接比较。

需要开发期外围依赖：
    pip install transformers datasets

例子（默认 CUDA，跑一个完整 100 题 subject）：
    python3 eval_mmlu.py ../models/Qwen3.5-0.8B out/qwen35-0.8b.bin \
        --subject abstract_algebra

评测会以 `qwen35_cuda --serve` 启动一个常驻 worker：权重只上传一次，而不是每题都重新
启动进程和上传 1.5 GiB checkpoint。
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


LABELS = "ABCD"
ANSWER = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="official Qwen3.5-0.8B directory (tokenizer only)")
    parser.add_argument("weights", type=Path, help="packed qwen35-0.8b.bin")
    parser.add_argument("--subject", default="abstract_algebra",
                        help="MMLU subject/config; default is the complete 100-question abstract_algebra test")
    parser.add_argument("--limit", type=int,
                        help="evaluate only the first N test examples (useful for slow CPU smoke tests)")
    parser.add_argument("--engine", default=str(Path(__file__).with_name("qwen35_cuda")),
                        help="persistent-worker CUDA executable (qwen35_cuda by default)")
    parser.add_argument("--max-new-tokens", type=int, default=4,
                        help="answer budget; four tokens is enough for `A` / `The answer is A`")
    parser.add_argument("--show", action="store_true", help="print every question's prediction")
    args = parser.parse_args()
    if not args.checkpoint.is_dir():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if not args.weights.is_file():
        parser.error(f"packed weights not found: {args.weights}")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    return args


def question_text(example: dict) -> str:
    """The sole task prompt. Answer labels, not answer contents, make parsing unambiguous."""
    choices = "\n".join(f"{label}. {choice}" for label, choice in zip(LABELS, example["choices"]))
    return (
        "Answer this multiple-choice question. Reply with only one letter: A, B, C, or D.\n\n"
        f"{example['question']}\n{choices}"
    )


class CudaWorker:
    """评测进程唯一的性能外壳；C++ 协议仍只传递 token ids。"""

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


def generate(tokenizer, worker: CudaWorker, text: str, max_new_tokens: int) -> str:
    # MMLU is text-only; its official Qwen chat prompt deliberately remains outside the C++ model.
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True,
        add_generation_prompt=True, return_dict=False
    )
    if len(ids) + max_new_tokens > 4096:
        raise RuntimeError("MMLU prompt exceeds the CUDA engine's 4096-token limit")
    output_ids = worker.generate(ids, max_new_tokens)
    return tokenizer.decode(output_ids, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False).strip()


def parse_choice(text: str) -> str | None:
    """Invalid or rambling generations are counted as wrong instead of silently guessed."""
    match = ANSWER.search(text)
    return match.group(1).upper() if match else None


def main() -> None:
    args = arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    dataset = load_dataset("cais/mmlu", args.subject, split="test")
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    correct = 0
    invalid = 0
    total = len(dataset)
    print(f"MMLU {args.subject}: {total} questions, persistent CUDA worker={args.engine}")
    worker = CudaWorker(args.engine, args.weights)
    try:
        for index, example in enumerate(dataset, 1):
            output = generate(tokenizer, worker, question_text(example), args.max_new_tokens)
            prediction = parse_choice(output)
            expected = LABELS[int(example["answer"])]
            if prediction is None:
                invalid += 1
            elif prediction == expected:
                correct += 1
            if args.show:
                shown = prediction or "<invalid>"
                print(f"{index:3}/{total}: predicted={shown} expected={expected} text={output!r}")
            elif index % 10 == 0 or index == total:
                print(f"{index:3}/{total}: running accuracy = {correct / index:.1%}")
    finally:
        worker.close()

    accuracy = correct / total if total else 0.0
    print()
    print(f"zero-shot generative MMLU accuracy ({args.subject}): {correct}/{total} = {accuracy:.1%}")
    print(f"invalid generations: {invalid}/{total}")


if __name__ == "__main__":
    main()
