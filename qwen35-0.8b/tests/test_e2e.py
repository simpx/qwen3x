#!/usr/bin/env python3
"""One real tokenizer -> HTTP -> SessionManager -> CPU model request."""

import argparse
from pathlib import Path

from fastapi.testclient import TestClient
from transformers import AutoTokenizer

from qwen35 import Engine
from server import Config, create_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--bin", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    with Engine(args.library, args.bin) as engine:
        manager = engine.create_session_manager(session_count=1, context_size=4096)
        config = Config(args.tokenizer, args.library, args.bin, api_key="test", slot_count=1)
        with TestClient(create_app(config, tokenizer=tokenizer, manager=manager)) as client:
            messages = [{"role": "user", "content": "用一句话介绍 DeltaNet。"}]
            checkpoint_at = len(tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=False, enable_thinking=False,
                preserve_thinking=True,
            ))
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer test"},
                json={
                    "model": "qwen3.5-0.8b",
                    "messages": messages,
                    "temperature": 0,
                    "max_completion_tokens": 8,
                },
            )
            response.raise_for_status()
            body = response.json()
            text = body["choices"][0]["message"]["content"]
            assert text
            assert "session_id" not in body

            messages.extend([
                {"role": "assistant", "content": text},
                {"role": "user", "content": "继续。"},
            ])
            continued = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer test"},
                json={
                    "model": "qwen3.5-0.8b",
                    "messages": messages,
                    "temperature": 0,
                    "max_completion_tokens": 1,
                },
            )
            continued.raise_for_status()
            cached = continued.json()["usage"]["prompt_tokens_details"]["cached_tokens"]
            assert cached >= checkpoint_at, (cached, checkpoint_at)
            print(f"native HTTP e2e: {text!r}, continued cached={cached}")

        # create_app does not own an injected manager.
        manager.close()


if __name__ == "__main__":
    main()
