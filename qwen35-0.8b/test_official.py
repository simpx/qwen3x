#!/usr/bin/env python3
"""将 CPU/CUDA 教学实现与官方 Qwen3.5 FP32 forward 做端到端数值对照。

这不是聊天质量评测，也不是“两个有相同 bug 的 executable 互相比较”。它的黄金
reference 是官方 checkpoint 加上 Hugging Face Transformers 的 Qwen3_5ForCausalLM：

    官方 tokenizer/chat template -> token ids -> 官方 FP32 model
                                      |              |
                                      +-> qwen35     +-> full logits / greedy ids
                                      +-> qwen35_cuda

脚本刻意使用 FP32 官方 forward。BF16 official inference 的每层 linear 都会再次舍入，
会让最终 logit 与本项目“BF16 weights + FP32 activation”的教学语义相差约 0.15；这不
应被误报为 C++ 误差。官方模型也逐 token、携带 cache 运行，因此 prefill 与 decode 的
执行顺序和 qwen35.cpp 完全一致。

需要开发期依赖（绝非 C++ runtime dependency）：
    pip install 'transformers>=5.0' torch

用法：
    make official-oracle MODEL=../models/Qwen3.5-0.8B
    # 或复用已经 pack 的权重，跳过一次 1.4 GiB 临时转换：
    make official-oracle MODEL=../models/Qwen3.5-0.8B WEIGHTS=out/qwen35-0.8b.bin
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


VOCAB_SIZE = 248320
STOP_TOKENS = {248044, 248046}  # 与 qwen35.cpp::generate() 保持同一个 text/chat 停止规则。
# 选择一条固定的普通 text chat，而非手写 token ids：这也覆盖官方 tokenizer/template。
MESSAGES = [{"role": "user", "content": "用一句话介绍 DeltaNet。"}]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="official Qwen3.5-0.8B checkpoint directory")
    parser.add_argument("--weights", type=Path,
                        help="optional qwen35 packed weights; omit to create a temporary one")
    parser.add_argument("--cpu-engine", default="./qwen35")
    parser.add_argument("--cuda-engine", default="./qwen35_cuda")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    # The threshold is intentionally much tighter than a BF16 comparison. It applies to every
    # one of 248,320 logits, so an argmax-only regression cannot hide a large distribution drift.
    parser.add_argument("--max-abs-error", type=float, default=1e-4)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    return args


def command_logits(engine: str, weights: Path, prompt_ids: list[int]) -> np.ndarray:
    """Ask one C++ executable for its complete next-token distribution after prefill."""
    ids = ",".join(map(str, prompt_ids))
    completed = subprocess.run([engine, "--logits", str(weights), ids], check=True,
                               text=True, capture_output=True)
    # qwen35 emits one %.9g FP32 value per line. fromstring avoids creating 248k Python float
    # objects and detects accidental human-readable banners through its expected-size check.
    logits = np.fromstring(completed.stdout, sep="\n", dtype=np.float32)
    if logits.size != VOCAB_SIZE:
        raise RuntimeError(f"{engine}: expected {VOCAB_SIZE} logits, received {logits.size}")
    return logits


def command_generate(engine: str, weights: Path, prompt_ids: list[int], count: int) -> list[int]:
    ids = ",".join(map(str, prompt_ids))
    completed = subprocess.run([engine, "--generate", str(weights), ids, str(count)],
                               check=True, text=True, capture_output=True)
    line = completed.stdout.strip()
    if not line.startswith("generated:"):
        raise RuntimeError(f"{engine}: unexpected generate output {line!r}")
    return [int(word) for word in line[len("generated:"):].split()]


def official_prefill(model, prompt_ids: list[int]):
    """Run official inference one token at a time, exactly as the C++ prefill loop does."""
    cache = None
    logits = None
    with torch.inference_mode():
        for token in prompt_ids:
            output = model(input_ids=torch.tensor([[token]], device="cuda"),
                           past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            logits = output.logits[0, -1].float()
    return logits, cache


def official_greedy(model, logits, cache, count: int) -> list[int]:
    """Continue the same official cache, exercising recurrent state and attention KV cache."""
    result = []
    with torch.inference_mode():
        for _ in range(count):
            token = int(logits.argmax())
            if token in STOP_TOKENS:
                break
            result.append(token)
            output = model(input_ids=torch.tensor([[token]], device="cuda"),
                           past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            logits = output.logits[0, -1].float()
    return result


def compare_logits(name: str, official: np.ndarray, actual: np.ndarray, limit: float) -> bool:
    error = np.abs(actual.astype(np.float64) - official.astype(np.float64))
    maximum = float(error.max())
    mean = float(error.mean())
    official_top = int(official.argmax())
    actual_top = int(actual.argmax())
    print(f"{name:5} logits: max_abs={maximum:.6g}, mean_abs={mean:.6g}, "
          f"top={actual_top} (official {official_top})")
    if actual_top != official_top:
        print(f"{name}: argmax mismatch", file=sys.stderr)
        return False
    if maximum > limit:
        print(f"{name}: max_abs {maximum:.6g} exceeds {limit:.6g}", file=sys.stderr)
        return False
    return True


def make_temporary_weights(checkpoint: Path, directory: Path) -> Path:
    """Use the normal tiny packer so the oracle tests the same model-loading path users run."""
    weights = directory / "qwen35-0.8b.bin"
    packer = Path(__file__).with_name("pack_weights.py")
    subprocess.run([sys.executable, str(packer), str(checkpoint), str(weights)], check=True)
    return weights


def run(args: argparse.Namespace, weights: Path) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("official-oracle needs a CUDA GPU because it tests qwen35_cuda")

    # `eager` makes official attention's math explicit instead of selecting an optional flash kernel.
    # Loading FP32 is deliberate; see the module docstring for the precision rationale.
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.float32, device_map="cuda:0", attn_implementation="eager"
    ).eval()
    # Transformers 5 returns a BatchEncoding by default; False preserves the plain `list[int]`
    # boundary used by both the old and new tokenizer APIs.
    prompt_ids = tokenizer.apply_chat_template(
        MESSAGES, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    print(f"official chat prompt: {len(prompt_ids)} tokens")

    reference_logits, reference_cache = official_prefill(model, prompt_ids)
    reference = reference_logits.cpu().numpy()
    reference_tokens = official_greedy(model, reference_logits, reference_cache, args.max_new_tokens)
    print("official greedy:", " ".join(map(str, reference_tokens)) or "<stop>")

    passed = True
    for name, engine in (("CPU", args.cpu_engine), ("CUDA", args.cuda_engine)):
        actual = command_logits(engine, weights, prompt_ids)
        passed = compare_logits(name, reference, actual, args.max_abs_error) and passed
        generated = command_generate(engine, weights, prompt_ids, args.max_new_tokens)
        print(f"{name:5} greedy:", " ".join(map(str, generated)) or "<stop>")
        if generated != reference_tokens:
            print(f"{name}: greedy tokens differ from official reference", file=sys.stderr)
            passed = False

    if passed:
        print("official oracle: passed (official FP32 == CPU == CUDA)")
    return passed


def main() -> None:
    args = arguments()
    if args.weights:
        if not args.weights.is_file():
            raise SystemExit(f"missing packed weights: {args.weights}")
        ok = run(args, args.weights)
    else:
        # Temporary conversion makes `make official-oracle MODEL=...` standalone, while the
        # optional --weights path keeps the quick edit/test loop from rewriting 1.4 GiB each time.
        with tempfile.TemporaryDirectory(prefix="qwen35-official-oracle-") as directory:
            ok = run(args, make_temporary_weights(args.checkpoint, Path(directory)))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
