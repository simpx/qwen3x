import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from qwen35 import SessionBusy
from server import Config, LogHighlighter, create_app


class FakeTokenizer:
    texts = {20: "你", 21: "你好", 22: "你好！"}

    def apply_chat_template(self, messages, *, add_generation_prompt, **_kwargs):
        assert messages[-1]["role"] == "user"
        return [10, 11] if add_generation_prompt else [10]

    def decode(self, ids, **_kwargs):
        return self.texts[ids[-1]] if ids else ""


class FakeSession:
    def __init__(self, index):
        self.index = index
        self.step = 0
        self.sync_count = 0
        self.cached_tokens = 0
        self.checkpoint_at = None

    def sync(self, tokens, checkpoint_at):
        assert tokens == [10, 11]
        assert checkpoint_at == 1
        self.sync_count += 1
        self.step = 0
        self.checkpoint_at = checkpoint_at
        return self.cached_tokens

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
        self.active = set()

    def acquire(self, _tokens):
        session = next(
            (item for item in self.sessions if item not in self.active), None
        )
        if session is None:
            raise SessionBusy("all sessions are busy")
        self.active.add(session)
        return session

    def release(self, session, *, keep):
        self.active.remove(session)
        if not keep:
            session.reset()


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

    def test_log_highlighter_marks_structured_values(self):
        text = LogHighlighter()(
            "[info] cache hit reused_tokens=128 elapsed=0.125s reason=append"
        )
        highlighted = {
            (span.style, text.plain[span.start:span.end])
            for span in text.spans
        }
        self.assertIn(("qwen.info", "[info]"), highlighted)
        self.assertIn(("qwen.number", "128"), highlighted)
        self.assertIn(("qwen.number", "0.125s"), highlighted)
        self.assertIn(("qwen.value", "append"), highlighted)

    def test_auth_health_and_models(self):
        first = self.client.get("/healthz")
        second = self.client.get("/healthz")
        self.assertEqual(first.status_code, 200)
        self.assertTrue(first.headers["x-request-id"].startswith("req-"))
        self.assertNotEqual(first.headers["x-request-id"], second.headers["x-request-id"])
        self.assertEqual(self.client.get("/readyz").json()["slots"], 2)
        self.assertEqual(self.client.get("/v1/models").status_code, 401)
        self.assertEqual(self.client.get("/v1/models", headers=self.headers).status_code, 200)

    def test_auth_is_disabled_by_default(self):
        manager = FakeManager(FakeEngine(), 1)
        config = Config(Path("model"), Path("library"), Path("weights"))
        with TestClient(create_app(config, tokenizer=FakeTokenizer(), manager=manager)) as client:
            self.assertEqual(client.get("/v1/models").status_code, 200)

    def test_non_streaming_reuses_resident_session(self):
        with self.assertLogs("qwen35.runtime", level="INFO") as logs:
            response = self.request()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "你好！")
        self.assertNotIn("session_id", response.json())
        self.assertNotIn("x-qwen-session-id", response.headers)
        self.assertEqual(
            response.json()["usage"]["prompt_tokens_details"]["cached_tokens"], 0
        )
        self.assertTrue(any(
            "prompt_tokens=2 cached_tokens=0 completion_tokens=3 total_tokens=5" in line
            for line in logs.output
        ))
        self.assertTrue(any("decode started" in line for line in logs.output))
        self.assertTrue(any("decode completed" in line for line in logs.output))
        self.assertTrue(any(
            "ttft=" in line and "prefill_tps=" in line and "decode_tps=" in line
            for line in logs.output
        ))
        first = self.manager.sessions[0]

        response = self.request(max_completion_tokens=1)
        self.assertEqual(response.status_code, 200)
        second = self.manager.sessions[0]
        self.assertIs(first, second)
        self.assertEqual(first.sync_count, 2)

    def test_streaming_contract(self):
        with self.assertLogs("qwen35.runtime", level="INFO") as logs:
            response = self.request(stream=True,
                                    stream_options={"include_usage": True})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn('"content":"你"', response.text)
        self.assertIn('"content":"好"', response.text)
        self.assertIn('"content":"！"', response.text)
        self.assertNotIn('"session_id"', response.text)
        self.assertIn(
            '"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5,'
            '"prompt_tokens_details":{"cached_tokens":0}}',
            response.text,
        )
        self.assertTrue(response.text.endswith("data: [DONE]\n\n"))
        self.assertTrue(any(
            "prompt_tokens=2 cached_tokens=0 completion_tokens=3 total_tokens=5" in line
            for line in logs.output
        ))

    def test_streaming_omits_usage_without_option(self):
        response = self.request(stream=True)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn('"usage":', response.text)
        self.assertTrue(response.text.endswith("data: [DONE]\n\n"))

    def test_usage_reports_engine_cache_hit(self):
        self.manager.sessions[0].cached_tokens = 2
        response = self.request(max_completion_tokens=1)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json()["usage"]["prompt_tokens_details"]["cached_tokens"], 2
        )

    def test_sampling_options(self):
        response = self.request(temperature=0.8, top_p=0.9, seed=123,
                                max_completion_tokens=1)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "你")

    def test_response_has_no_session_extension(self):
        response = self.request(max_completion_tokens=1)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("session_id", response.json())
        self.assertNotIn("x-qwen-session-id", response.headers)

        rejected = self.request(session_id="custom-session", max_completion_tokens=1)
        self.assertEqual(rejected.status_code, 400)
        self.assertEqual(rejected.json()["error"]["param"], "session_id")

    def test_sessions_extension_is_absent(self):
        response = self.client.delete("/v1/sessions/anything", headers=self.headers)
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
