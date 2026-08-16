#!/usr/bin/env python3
"""一个刻意放在 C++ core 外面的最小 chat frontend。

qwen35 / qwen35_cuda 只做一件事：token id -> next token id。这里才做两项文字
外围工作：让官方 tokenizer 套用官方 chat template，以及把生成的 ids 解码为 UTF-8。
因此读 qwen35.cpp 时，不必先理解 Jinja template、Unicode 或 Python subprocess。

这是教学用的逐回合 frontend：每次对话都会把完整 history 再喂给 executable，
所以 C++ 程序无需为 server、流式输出或跨进程 state 增添复杂度。
"""

import argparse
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer


def parse_generated(stdout: str) -> list[int]:
    """读取 qwen35 --generate 唯一的机器可读输出行：`generated: 1 2 3`。"""
    line = stdout.strip()
    if not line.startswith("generated:"):
        raise RuntimeError(f"unexpected engine output: {line!r}")
    words = line[len("generated:"):].split()
    return [int(word) for word in words]


def generate(tokenizer, messages, engine: str, weights: str, max_new_tokens: int,
             show_ids: bool) -> str:
    # `apply_chat_template` 从 checkpoint 的 tokenizer_config.json 读取模板；本项目
    # 不手写 <|im_start|>、<think> 等控制 token，也不将这类知识塞进 C++。
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    if len(prompt_ids) + max_new_tokens > 2048:
        raise RuntimeError(
            f"prompt ({len(prompt_ids)}) + generation ({max_new_tokens}) exceeds "
            "the teaching engine's 2048-token limit"
        )
    if show_ids:
        print("prompt ids:", ",".join(map(str, prompt_ids)), file=sys.stderr)

    # 这里是 Python 和 C++ 的唯一边界：一串逗号分隔的 token ids。C++ 不知道文本、
    # chat role 或 tokenizer 的存在，因而 CPU/CUDA forward 保持完全相同。
    command = [engine, "--generate", weights, ",".join(map(str, prompt_ids)),
               str(max_new_tokens)]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    output_ids = parse_generated(completed.stdout)
    if show_ids:
        print("output ids:", ",".join(map(str, output_ids)), file=sys.stderr)
    return tokenizer.decode(output_ids, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False).strip()


def add_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official-tokenizer chat wrapper around the qwen35 teaching engine."
    )
    parser.add_argument("--model", required=True,
                        help="official Qwen3.5-0.8B checkpoint directory (tokenizer only)")
    parser.add_argument("--weights", required=True,
                        help="qwen35 binary weights created by pack_weights.py")
    parser.add_argument("--prompt", help="one user turn; prints one answer and exits")
    parser.add_argument("--interactive", action="store_true",
                        help="read multiple user turns; /exit leaves the chat")
    parser.add_argument("--system", help="optional system message")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--engine", help="path to qwen35 or qwen35_cuda")
    parser.add_argument("--cuda", action="store_true",
                        help="shortcut for --engine ./qwen35_cuda")
    parser.add_argument("--show-ids", action="store_true",
                        help="also print the token ids crossing the Python/C++ boundary")
    args = parser.parse_args()
    if bool(args.prompt) == args.interactive:
        parser.error("pass exactly one of --prompt or --interactive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.engine and args.cuda:
        parser.error("use either --engine or --cuda, not both")
    return args


def main() -> None:
    args = add_arguments()
    # 默认 binary 相对本脚本定位，而非相对 shell 当前目录；README 里的 `cd` 只是让
    # 命令更短，用户从仓库根目录执行 `python qwen35-0.8b/chat.py ...` 也应当可用。
    default_engine = Path(__file__).with_name("qwen35_cuda" if args.cuda else "qwen35")
    engine = args.engine or str(default_engine)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # The Python-side history is deliberately ordinary text messages. Recomputing the full
    # prompt each turn is inefficient, but exposes prefill clearly and avoids server state.
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    def reply(user_text: str) -> None:
        messages.append({"role": "user", "content": user_text})
        answer = generate(tokenizer, messages, engine, args.weights,
                          args.max_new_tokens, args.show_ids)
        messages.append({"role": "assistant", "content": answer})
        print(answer)

    if args.prompt:
        reply(args.prompt)
        return

    print("type /exit to quit")
    while True:
        try:
            user_text = input("you> ").strip()
        except EOFError:
            print()
            return
        if user_text in {"/exit", "/quit"}:
            return
        if user_text:
            reply(user_text)


if __name__ == "__main__":
    main()
