import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from qwen35 import SessionBusy
from server import Config, create_app


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

    def sample(self, _temperature, _top_p, rng):
        return self.argmax(), rng + 1

    def eval(self, token):
        assert token == [20, 21, 22][self.step]
        self.step += 1

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

    def token_is_stop(self, _token):
        return False


class FakeManager:
    def __init__(self, engine, session_count):
        self.engine = engine
        self.session_count = session_count
        self.sessions = [engine.create_session(128) for _ in range(session_count)]
        self.named = {}
        self.active = set()

    def acquire(self, session_id, _tokens):
        if session_id in self.named:
            session = self.named[session_id]
            if session in self.active:
                raise SessionBusy("session is already busy")
        else:
            session = next(item for item in self.sessions if item not in self.active)
            if session_id is not None:
                for old_id, item in list(self.named.items()):
                    if item is session:
                        del self.named[old_id]
                self.named[session_id] = session
        self.active.add(session)
        return session

    def release(self, session, *, keep):
        self.active.remove(session)
        if not keep:
            for session_id, item in list(self.named.items()):
                if item is session:
                    del self.named[session_id]
            session.reset()

    def forget(self, session_id):
        session = self.named.get(session_id)
        if session is None:
            return False
        if session in self.active:
            raise SessionBusy("session is busy")
        del self.named[session_id]
        session.reset()
        return True


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.manager = FakeManager(self.engine, 2)
        config = Config(Path("model"), Path("library"), Path("weights"), api_key="secret",
                        max_context_tokens=128)
        self.context = TestClient(
            create_app(config, tokenizer=FakeTokenizer(), manager=self.manager)
        )
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

    def test_auth_is_disabled_by_default(self):
        manager = FakeManager(FakeEngine(), 1)
        config = Config(Path("model"), Path("library"), Path("weights"))
        with TestClient(create_app(config, tokenizer=FakeTokenizer(), manager=manager)) as client:
            self.assertEqual(client.get("/v1/models").status_code, 200)

    def test_non_streaming_uses_named_resident_session(self):
        response = self.request(session_id="agent-a")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "你好！")
        self.assertEqual(response.json()["session_id"], "agent-a")
        self.assertEqual(response.headers["x-qwen-session-id"], "agent-a")
        first = self.manager.named["agent-a"]

        response = self.request(session_id="agent-a", max_completion_tokens=1)
        self.assertEqual(response.status_code, 200)
        second = self.manager.named["agent-a"]
        self.assertIs(first, second)
        self.assertEqual(first.sync_count, 2)

    def test_streaming_contract(self):
        response = self.request(session_id="agent-stream", stream=True,
                                stream_options={"include_usage": True})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn('"content":"你"', response.text)
        self.assertIn('"content":"好"', response.text)
        self.assertIn('"content":"！"', response.text)
        self.assertIn('"session_id":"agent-stream"', response.text)
        self.assertTrue(response.text.endswith("data: [DONE]\n\n"))

    def test_sampling_options(self):
        response = self.request(temperature=0.8, top_p=0.9, seed=123,
                                max_completion_tokens=1)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "你")

    def test_anonymous_request_creates_no_named_binding(self):
        response = self.request(max_completion_tokens=1)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["x-qwen-session-id"].startswith("ephemeral-"))
        self.assertFalse(self.manager.named)

    def test_delete_session(self):
        self.request(session_id="agent-delete", max_completion_tokens=1)
        response = self.client.delete("/v1/sessions/agent-delete", headers=self.headers)
        self.assertEqual(response.json(), {"id": "agent-delete", "deleted": True})


if __name__ == "__main__":
    unittest.main()
