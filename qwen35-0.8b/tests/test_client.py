#!/usr/bin/env python3
"""Regression for the standalone multi-turn client."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading
import unittest


CLIENT = Path(__file__).resolve().parents[1] / "client"


class Handler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        self.requests.append(body)
        turn = len(self.requests)
        text = f"reply-{turn}"
        chunks = [
            {"choices": [{"delta": {"content": text}}]},
            {"choices": [], "usage": {
                "prompt_tokens": turn * 10,
                "completion_tokens": 2,
                "total_tokens": turn * 10 + 2,
                "prompt_tokens_details": {"cached_tokens": (turn - 1) * 5},
            }},
        ]
        payload = "".join(
            "data: " + json.dumps(chunk, separators=(",", ":")) + "\n\n"
            for chunk in chunks
        ) + "data: [DONE]\n\n"
        encoded = payload.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format, *_args):
        pass


class ClientTest(unittest.TestCase):
    def test_two_prompts_are_sent_as_one_conversation(self):
        Handler.requests = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = subprocess.run(
                [sys.executable, CLIENT, "--no-color", "-u",
                 f"http://127.0.0.1:{server.server_port}", "hello", "hello2"],
                check=True, capture_output=True, text=True,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

        self.assertIn("req: hello\nresp: reply-1", result.stdout)
        self.assertIn("req: hello2\nresp: reply-2", result.stdout)
        self.assertIn("usage: prompt=20  cached=5  completion=2  total=22", result.stdout)
        self.assertEqual(len(Handler.requests), 2)
        first, second = Handler.requests
        self.assertEqual(first["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(second["messages"], [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "reply-1"},
            {"role": "user", "content": "hello2"},
        ])
        self.assertNotIn("session_id", first)
        self.assertNotIn("session_id", second)
        self.assertTrue(first["stream_options"]["include_usage"])


if __name__ == "__main__":
    unittest.main()
