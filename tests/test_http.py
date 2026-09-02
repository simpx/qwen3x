#!/usr/bin/env python3
"""Black-box regression for the standalone C++ HTTP runtime."""

import argparse
import http.client
import json
import os
import socket
import subprocess
import tempfile
import time
import uuid


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
         "--session-context", "256", "--mock"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert direct.returncode == 0, direct.stderr
    assert direct.stderr == "", direct.stderr

    with tempfile.TemporaryDirectory() as directory:
        log_file = os.path.join(directory, "qwen35.log")
        logged = subprocess.run(
            [args.program, "--prompt", "hello", "--max-tokens", "1",
             "--session-context", "256", "--mock", "--log-level", "info",
             "--log-file", log_file],
            capture_output=True,
            text=True,
            check=False,
        )
        assert logged.returncode == 0, logged.stderr
        assert logged.stderr == "", logged.stderr
        with open(log_file, encoding="utf-8") as stream:
            assert "mock compute enabled" in stream.read()

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    environment = dict(os.environ, QWEN_API_KEY="test")
    server_logs = tempfile.TemporaryDirectory()
    server_log = os.path.join(server_logs.name, "server.log")
    audit_log = os.path.join(server_logs.name, "audit.log")
    process = subprocess.Popen(
        [args.program, "--listen", "--host", "127.0.0.1", "--port", str(port),
         "--session-slots", "2", "--session-context", "1024", "--mock",
         "--log-file", server_log, "--audit-log", audit_log],
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
        status, _, data = request(port, "GET", "/v1/models")
        assert status == 200
        served_model = json.loads(data)["data"][0]["id"]
        assert served_model in ("qwen3.5-0.8b", "qwen3.5-4b", "qwen3.5-9b")

        body = {
            "model": served_model,
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
            "max_completion_tokens": 3,
        }
        status, headers, data = request(
            port, "POST", "/v1/chat/completions", body)
        assert status == 200, data
        session_id = headers["X-Session-Id"]
        assert uuid.UUID(session_id).version == 4
        completion = json.loads(data)
        assert completion["choices"][0]["message"]["content"]
        assert completion["usage"]["completion_tokens"] == 3
        status, headers, _ = request(
            port, "POST", "/v1/chat/completions", body)
        assert status == 200
        assert headers["X-Session-Id"] == session_id

        thinking_body = {
            "model": served_model,
            "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": {"enable_thinking": True},
            "stream": True,
            "temperature": 0,
            "max_completion_tokens": 32,
        }
        status, thinking_headers, data = request(
            port, "POST", "/v1/chat/completions", thinking_body)
        assert status == 200, data
        events = [json.loads(line[6:]) for line in data.splitlines()
                  if line.startswith("data: {")]
        assert events[-1]["error"]["code"] == "incomplete_generation"
        assert "please retry your request" in events[-1]["error"]["message"]
        assert "data: [DONE]" not in data

        status, retry_headers, retry_data = request(
            port, "POST", "/v1/chat/completions", thinking_body)
        assert status == 200
        assert retry_headers["X-Session-Id"] == thinking_headers["X-Session-Id"]
        assert '"code":"incomplete_generation"' in retry_data
        assert "data: [DONE]" not in retry_data

        tool = {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
        tool_body = {
            "model": served_model,
            "messages": [{"role": "user", "content": "read README.md"}],
            "tools": [tool],
            "stream": False,
            "temperature": 0,
            "max_completion_tokens": 1,
        }
        status, _, data = request(
            port, "POST", "/v1/chat/completions", tool_body)
        assert status == 200, data

        tool_body["messages"].extend([
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"README.md"}',
                    },
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "hello"},
        ])
        status, _, data = request(
            port, "POST", "/v1/chat/completions", tool_body)
        assert status == 200, data

        tool_body["stream"] = True
        status, _, data = request(
            port, "POST", "/v1/chat/completions", tool_body)
        assert status == 200, data
        assert "data: [DONE]\n\n" in data
        assert '"role":"assistant"' in data

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
    with open(server_log, encoding="utf-8") as stream:
        logs = stream.read()
        assert "server ready" in logs
        assert "renderer load started" in logs
        assert "renderer load completed" in logs
        assert "access started request_id=" in logs
        assert "request prepared request_id=" in logs
        assert "parse_ms=" in logs and "render_ms=" in logs
        assert "generation started request_id=" in logs
        assert "prefill_ms=" in logs and "ttft_ms=" in logs
        assert "duration_ms=" in logs and "decode_tps=" in logs
        assert "access completed request_id=" in logs
        assert logs.count("access started request_id=") == \
            logs.count("access completed request_id=")
        assert "request accepted" not in logs
        assert "event=request" not in logs
    assert os.stat(audit_log).st_mode & 0o777 == 0o600
    with open(audit_log, encoding="utf-8") as stream:
        audit = stream.read()
        assert "event=request" in audit and '"content": "hello"' in audit
        assert "event=render" in audit
        assert f"event=session_acquired" in audit
        assert f"session_id={session_id}" in audit
        assert "event=generation_end" in audit and "stop_cause=" in audit
        retryable_generations = [
            line for line in audit.splitlines()
            if "event=generation_end" in line and
               "stop_cause=incomplete_generation" in line
        ]
        assert len(retryable_generations) == 2
        assert all("retryable=1" in line for line in retryable_generations)
        assert "cached_tokens=0" not in retryable_generations[1]
        assert "event=tool_parse" in audit
        assert "event=response_chunk" in audit and "data: [DONE]" in audit
        assert "event=response_end" in audit
        assert "Bearer test" not in audit
    server_logs.cleanup()
    print("http-test: ok")


if __name__ == "__main__":
    main()
