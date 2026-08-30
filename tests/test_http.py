#!/usr/bin/env python3
"""Black-box regression for the standalone C++ HTTP runtime."""

import argparse
import http.client
import json
import os
import socket
import subprocess
import time


def request(port, method, path, body=None, *, authorized=True):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {}
    if body is not None:
        body = json.dumps(body, ensure_ascii=False).encode()
        headers["Content-Type"] = "application/json"
    if authorized:
        headers["Authorization"] = "Bearer test"
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    data = response.read().decode()
    result = response.status, dict(response.getheaders()), data
    connection.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    args = parser.parse_args()

    direct = subprocess.run(
        [args.program, "--prompt", "hello", "--max-tokens", "3",
         "--session-context", "256", "--mock", "--log-level", "error"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert direct.returncode == 0, direct.stderr

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    environment = dict(os.environ, QWEN_API_KEY="test")
    process = subprocess.Popen(
        [args.program, "--listen", "--host", "127.0.0.1", "--port", str(port),
         "--session-slots", "2", "--session-context", "256", "--mock",
         "--log-level", "error"],
        env=environment,
    )
    try:
        for _ in range(100):
            try:
                status, first_headers, data = request(
                    port, "GET", "/healthz", authorized=False)
                if status == 200:
                    break
            except OSError:
                time.sleep(0.02)
        else:
            raise AssertionError("server did not become healthy")
        status, second_headers, _ = request(
            port, "GET", "/healthz", authorized=False)
        assert json.loads(data) == {"status": "ok"}
        assert first_headers["X-Request-Id"] != second_headers["X-Request-Id"]
        assert request(port, "GET", "/v1/models", authorized=False)[0] == 401
        assert request(port, "GET", "/v1/models")[0] == 200

        body = {
            "model": "qwen3.5-0.8b",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
            "max_completion_tokens": 3,
        }
        status, _, data = request(port, "POST", "/v1/chat/completions", body)
        assert status == 200, data
        completion = json.loads(data)
        assert completion["choices"][0]["message"]["content"]
        assert completion["usage"]["completion_tokens"] == 3

        body.update({
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": 4,
        })
        status, _, data = request(port, "POST", "/v1/chat/completions", body)
        assert status == 200, data
        assert "data: [DONE]\n\n" in data
        assert '"choices":[]' in data and '"cached_tokens":' in data

        body["top_p"] = 0
        status, _, data = request(port, "POST", "/v1/chat/completions", body)
        assert status == 400 and "invalid chat request" in data
    finally:
        process.terminate()
        process.wait(timeout=10)
    print("http-test: ok")


if __name__ == "__main__":
    main()
