#!/usr/bin/env python3
"""One real tokenizer -> HTTP -> Slot -> C++ Session -> CPU model request."""

import argparse
import asyncio
from pathlib import Path

from fastapi.testclient import TestClient
from transformers import AutoTokenizer

from qwen_runtime.binding import Engine
from qwen_runtime.server import Config, create_app
from qwen_runtime.slots import SlotPool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    with Engine(args.library, args.weights) as engine:
        pool = SlotPool(engine, slot_count=1, context_size=4096)
        config = Config(args.model, args.library, args.weights, api_key="test", slot_count=1)
        with TestClient(create_app(config, tokenizer=tokenizer, pool=pool)) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer test"},
                json={
                    "model": "qwen3.5-0.8b",
                    "session_id": "e2e-agent",
                    "messages": [{"role": "user", "content": "用一句话介绍 DeltaNet。"}],
                    "temperature": 0,
                    "max_completion_tokens": 8,
                },
            )
            response.raise_for_status()
            body = response.json()
            text = body["choices"][0]["message"]["content"]
            assert text
            assert body["session_id"] == "e2e-agent"
            assert any(slot.owner == "e2e-agent" for slot in pool.slots)
            print(f"native HTTP e2e: {text!r}")

        # create_app does not own injected pools.
        asyncio.run(pool.close())


if __name__ == "__main__":
    main()
