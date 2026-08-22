#!/usr/bin/env python3
"""不加载真实模型，验证鉴权、OpenAI JSON、参数拒绝和 SSE contract。"""

import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from runtime import Config, create_app


class FakeTokenizer:
    texts = {20: "你", 21: "你好", 22: "你好！"}

    def apply_chat_template(self, messages, **_kwargs):
        assert messages[-1]["role"] == "user"
        return [10, 11]

    def decode(self, ids, **_kwargs):
        return self.texts[ids[-1]] if ids else ""


class FakeWorker:
    alive = True

    def __init__(self):
        self.tokens = []
        self.resets = 0

    def start(self, prompt_ids):
        assert prompt_ids == [10, 11]
        self.tokens = [20, 21, 22]
        return self.tokens.pop(0)

    def next(self):
        return self.tokens.pop(0) if self.tokens else None

    def reset(self):
        self.resets += 1


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.worker = FakeWorker()
        config = Config(Path("model"), Path("engine"), Path("weights"), api_key="secret")
        self.client_context = TestClient(
            create_app(config, tokenizer=FakeTokenizer(), worker=self.worker)
        )
        self.client = self.client_context.__enter__()
        self.headers = {"Authorization": "Bearer secret"}

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def request(self, **extra):
        body = {
            "model": "qwen3.5-0.8b",
            "messages": [{"role": "user", "content": "你好"}],
            "temperature": 0,
            "max_completion_tokens": 3,
        }
        body.update(extra)
        return self.client.post("/v1/chat/completions", headers=self.headers, json=body)

    def test_health_models_and_auth(self):
        self.assertEqual(self.client.get("/healthz").status_code, 200)
        self.assertEqual(self.client.get("/readyz").status_code, 200)
        self.assertEqual(self.client.get("/v1/models").status_code, 401)
        response = self.client.get("/v1/models", headers=self.headers)
        self.assertEqual(response.json()["data"][0]["id"], "qwen3.5-0.8b")

    def test_non_streaming_completion(self):
        response = self.request()
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"],
                         {"role": "assistant", "content": "你好！", "refusal": None})
        self.assertEqual(body["choices"][0]["finish_reason"], "length")
        self.assertEqual(body["usage"],
                         {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5})
        self.assertEqual(self.worker.resets, 1)

    def test_streaming_completion(self):
        response = self.request(stream=True, stream_options={"include_usage": True})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn('"role":"assistant"', response.text)
        self.assertIn('"content":"你"', response.text)
        self.assertIn('"content":"好"', response.text)
        self.assertIn('"content":"！"', response.text)
        self.assertIn('"finish_reason":"length"', response.text)
        self.assertIn('"choices":[],"usage":', response.text)
        self.assertTrue(response.text.endswith("data: [DONE]\n\n"))

    def test_rejects_unsupported_sampling_and_model(self):
        response = self.request(temperature=0.7)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["param"], "temperature")
        response = self.request(model="missing")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "model_not_found")

    def test_stop_string_is_removed(self):
        response = self.request(stop="！")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "你好")
        self.assertEqual(response.json()["choices"][0]["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main()
