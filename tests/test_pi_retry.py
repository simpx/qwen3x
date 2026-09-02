#!/usr/bin/env python3
"""Verify that Pi retries the SSE error emitted by qwen3x."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import subprocess
import tempfile
import threading


ERROR = {
    "error": {
        "message": (
            "incomplete generated tool call; please retry your request"
        ),
        "type": "server_error",
        "param": None,
        "code": "incomplete_tool_call",
    }
}


def sse(*events):
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()


class RetryServer(ThreadingHTTPServer):
    def __init__(self, address):
        super().__init__(address, Handler)
        self.requests = []


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        self.server.requests.append(request)
        attempt = len(self.server.requests)
        if attempt == 1:
            body = sse(ERROR)
        else:
            completion_id = "chatcmpl-pi-retry"
            common = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "retry-model",
            }
            body = sse(
                {**common, "choices": [{"index": 0,
                                         "delta": {"role": "assistant"},
                                         "finish_reason": None}]},
                {**common, "choices": [{"index": 0,
                                         "delta": {"content": "retry succeeded"},
                                         "finish_reason": None}]},
                {**common, "choices": [{"index": 0, "delta": {},
                                         "finish_reason": "stop"}]},
            ) + b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi", default="pi")
    args = parser.parse_args()

    server = RetryServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory() as directory:
            models = {
                "providers": {
                    "retry-test": {
                        "baseUrl": f"http://127.0.0.1:{server.server_port}/v1",
                        "api": "openai-completions",
                        "apiKey": "local",
                        "compat": {
                            "supportsDeveloperRole": False,
                            "supportsUsageInStreaming": False,
                        },
                        "models": [{
                            "id": "retry-model",
                            "input": ["text"],
                            "contextWindow": 8192,
                            "maxTokens": 64,
                        }],
                    }
                }
            }
            settings = {
                "retry": {"enabled": True, "maxRetries": 1,
                          "baseDelayMs": 1},
            }
            with open(os.path.join(directory, "models.json"), "w",
                      encoding="utf-8") as stream:
                json.dump(models, stream)
            with open(os.path.join(directory, "settings.json"), "w",
                      encoding="utf-8") as stream:
                json.dump(settings, stream)

            environment = dict(
                os.environ,
                PI_CODING_AGENT_DIR=directory,
                PI_OFFLINE="1",
                PI_TELEMETRY="0",
            )
            result = subprocess.run(
                [args.pi, "--provider", "retry-test", "--model", "retry-model",
                 "--print", "--no-session", "--no-tools", "--no-extensions",
                 "--no-skills", "--no-context-files", "--offline",
                 "--system-prompt", "Answer briefly.", "hi"],
                cwd=directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            assert result.returncode == 0, result.stderr
            assert "retry succeeded" in result.stdout, result.stdout
            assert len(server.requests) == 2, len(server.requests)
            assert server.requests[0]["messages"] == server.requests[1]["messages"]
            assert all(message["role"] != "assistant"
                       for message in server.requests[1]["messages"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    print("pi-retry-test: ok")


if __name__ == "__main__":
    main()
