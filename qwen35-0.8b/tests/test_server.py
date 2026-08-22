import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from qwen_runtime.server import Config, create_app
from qwen_runtime.slots import SlotPool


class FakeTokenizer:
    texts = {20: "你", 21: "你好", 22: "你好！"}

    def apply_chat_template(self, messages, **_kwargs):
        assert messages[-1]["role"] == "user"
        return [10, 11]

    def decode(self, ids, **_kwargs):
        return self.texts[ids[-1]] if ids else ""


class FakeSession:
    def __init__(self, index):
        self.index = index
        self.step = 0
        self.sync_count = 0

    def sync(self, tokens):
        assert tokens == [10, 11]
        self.sync_count += 1
        self.step = 0

    def argmax(self):
        return [20, 21, 22][self.step]

    def eval(self, token):
        assert token == [20, 21, 22][self.step]
        self.step += 1

    def is_stop_token(self, _token):
        return False

    def reset(self):
        self.step = 0

    def close(self):
        pass


class FakeEngine:
    def __init__(self):
        self.sessions = []

    def create_session(self, _context_size):
        session = FakeSession(len(self.sessions))
        self.sessions.append(session)
        return session


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.pool = SlotPool(self.engine, 2, 128)
        config = Config(Path("model"), Path("library"), Path("weights"), api_key="secret",
                        max_context_tokens=128)
        self.context = TestClient(create_app(config, tokenizer=FakeTokenizer(), pool=self.pool))
        self.client = self.context.__enter__()
        self.headers = {"Authorization": "Bearer secret"}

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def request(self, **extra):
        body = {
            "model": "qwen3.5-0.8b",
            "messages": [{"role": "user", "content": "你好"}],
            "temperature": 0,
            "max_completion_tokens": 3,
        }
        body.update(extra)
        return self.client.post("/v1/chat/completions", headers=self.headers, json=body)

    def test_auth_health_and_models(self):
        self.assertEqual(self.client.get("/healthz").status_code, 200)
        self.assertEqual(self.client.get("/readyz").json()["slots"], 2)
        self.assertEqual(self.client.get("/v1/models").status_code, 401)
        self.assertEqual(self.client.get("/v1/models", headers=self.headers).status_code, 200)

    def test_non_streaming_uses_named_resident_session(self):
        response = self.request(session_id="agent-a")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "你好！")
        self.assertEqual(response.json()["session_id"], "agent-a")
        self.assertEqual(response.headers["x-qwen-session-id"], "agent-a")
        first = next(slot for slot in self.pool.slots if slot.owner == "agent-a")

        response = self.request(session_id="agent-a", max_completion_tokens=1)
        self.assertEqual(response.status_code, 200)
        second = next(slot for slot in self.pool.slots if slot.owner == "agent-a")
        self.assertIs(first.session, second.session)
        self.assertEqual(first.session.sync_count, 2)

    def test_streaming_contract(self):
        response = self.request(session_id="agent-stream", stream=True,
                                stream_options={"include_usage": True})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn('"content":"你"', response.text)
        self.assertIn('"content":"好"', response.text)
        self.assertIn('"content":"！"', response.text)
        self.assertIn('"session_id":"agent-stream"', response.text)
        self.assertTrue(response.text.endswith("data: [DONE]\n\n"))

    def test_ephemeral_request_does_not_keep_slot_owner(self):
        response = self.request(max_completion_tokens=1)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["x-qwen-session-id"].startswith("ephemeral-"))
        self.assertFalse(any(slot.owner is not None for slot in self.pool.slots))

    def test_delete_session(self):
        self.request(session_id="agent-delete", max_completion_tokens=1)
        response = self.client.delete("/v1/sessions/agent-delete", headers=self.headers)
        self.assertEqual(response.json(), {"id": "agent-delete", "deleted": True})


if __name__ == "__main__":
    unittest.main()
