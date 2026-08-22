#!/usr/bin/env python3
"""真实 tokenizer + 常驻 C++ engine + OpenAI HTTP handler 的本机 e2e。"""

import argparse
from pathlib import Path

from fastapi.testclient import TestClient

from runtime import Config, create_app


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(args.model, args.engine, args.weights, api_key="test-key")
    with TestClient(create_app(config)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-key"},
            json={
                "model": "qwen3.5-0.8b",
                "messages": [{"role": "user", "content": "用一句话介绍 DeltaNet。"}],
                "temperature": 0,
                "max_completion_tokens": 8,
            },
        )
        response.raise_for_status()
        body = response.json()
        text = body["choices"][0]["message"]["content"]
        if not text.strip() or body["usage"]["completion_tokens"] <= 0:
            raise SystemExit(f"runtime returned an empty completion: {body}")
        print(f"runtime e2e: {text!r}")
        print("runtime e2e: passed")


if __name__ == "__main__":
    main()
