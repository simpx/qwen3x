#!/usr/bin/env python3
"""单用户 OpenAI-compatible runtime：Python HTTP/tokenizer -> 常驻 qwen3x engine。"""

import argparse
import asyncio
import hmac
import json
import logging
import os
import select
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


HERE = Path(__file__).resolve().parent
LOG = logging.getLogger("qwen3x.runtime")


class APIError(Exception):
    def __init__(self, status: int, message: str, *, param=None,
                 code=None, error_type="invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.message = message
        self.param = param
        self.code = code
        self.error_type = error_type


class EngineError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    model_path: Path
    engine_path: Path
    weights_path: Path
    served_model_name: str = "qwen3.5-0.8b"
    api_key: Optional[str] = None
    max_context_tokens: int = 4096
    default_max_tokens: int = 128
    max_request_bytes: int = 1024 * 1024
    worker_start_timeout: float = 120.0
    worker_step_timeout: float = 600.0
    request_timeout: float = 600.0


class EngineWorker:
    """一个常驻 C++ 子进程；一条 start/next 命令恰好推进一个生成 token。"""

    def __init__(self, engine: Path, weights: Path, start_timeout: float = 120.0,
                 step_timeout: float = 600.0):
        self.engine = engine.resolve()
        self.weights = weights.resolve()
        self.start_timeout = start_timeout
        self.step_timeout = step_timeout
        self.process = None
        self._lock = threading.Lock()
        self._start_process()

    @property
    def alive(self):
        return self.process is not None and self.process.poll() is None

    def _start_process(self):
        if not self.engine.is_file():
            raise EngineError(f"engine not found: {self.engine}")
        if not self.weights.is_file():
            raise EngineError(f"packed weights not found: {self.weights}")
        self.process = subprocess.Popen(
            [str(self.engine), "--worker", str(self.weights)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        readable, _, _ = select.select([self.process.stdout], [], [], self.start_timeout)
        if not readable:
            self.process.kill()
            self.process.wait()
            raise EngineError("engine did not become ready before the startup timeout")
        line = self.process.stdout.readline().rstrip("\n")
        if line != "ready":
            code = self.process.poll()
            self.close(force=True)
            raise EngineError(f"engine startup failed (exit={code}, output={line!r})")
        LOG.info("engine ready: pid=%s", self.process.pid)

    def _command(self, command: str):
        with self._lock:
            if not self.alive:
                raise EngineError("engine worker is not running")
            try:
                self.process.stdin.write(command + "\n")
                self.process.stdin.flush()
                readable, _, _ = select.select(
                    [self.process.stdout], [], [], self.step_timeout
                )
                if not readable:
                    self.close(force=True)
                    raise EngineError("engine step timed out")
                line = self.process.stdout.readline()
            except (BrokenPipeError, OSError) as error:
                raise EngineError("engine worker pipe failed") from error
            if not line:
                raise EngineError(f"engine worker exited with code {self.process.poll()}")
            line = line.rstrip("\n")
            if line.startswith("error\t"):
                raise EngineError(line.split("\t", 1)[1])
            return line

    @staticmethod
    def _parse_step(line):
        if line == "done\tstop":
            return None
        if line.startswith("token\t"):
            try:
                return int(line.split("\t", 1)[1])
            except ValueError as error:
                raise EngineError(f"bad token response: {line!r}") from error
        raise EngineError(f"unexpected engine response: {line!r}")

    def start(self, prompt_ids):
        if not self.alive:
            self._start_process()
        ids = ",".join(str(token) for token in prompt_ids)
        return self._parse_step(self._command(f"start\t{ids}"))

    def next(self):
        return self._parse_step(self._command("next"))

    def reset(self):
        if self.alive:
            line = self._command("reset")
            if line != "ok":
                raise EngineError(f"unexpected reset response: {line!r}")

    def close(self, force=False):
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        if not force:
            try:
                process.stdin.write("quit\n")
                process.stdin.flush()
                process.wait(timeout=5)
                return
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                pass
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def error_body(message, *, param=None, code=None, error_type="invalid_request_error"):
    return {"error": {"message": message, "type": error_type, "param": param, "code": code}}


def text_content(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                raise APIError(400, "only text message content is supported", param="messages")
            parts.append(part["text"])
        return "".join(parts)
    raise APIError(400, "message content must be text", param="messages")


def normalize_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "messages must be a non-empty array", param="messages")
    result = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise APIError(400, "each message must be an object", param=f"messages[{index}]")
        role = message.get("role")
        if role == "developer":
            role = "system"
        if role not in {"system", "user", "assistant"}:
            raise APIError(400, f"unsupported message role: {role!r}", param=f"messages[{index}].role")
        if role == "system" and index != 0:
            raise APIError(400, "system/developer message must be first", param="messages")
        result.append({"role": role, "content": text_content(message)})
    if not any(message["role"] == "user" for message in result):
        raise APIError(400, "messages must contain a user message", param="messages")
    return result


def parse_request(body, config, tokenizer):
    if not isinstance(body, dict):
        raise APIError(400, "request body must be a JSON object")
    if body.get("model") != config.served_model_name:
        raise APIError(404, f"model {body.get('model')!r} not found", param="model",
                       code="model_not_found")
    for key in ("tools", "tool_choice", "functions", "function_call", "response_format"):
        if key in body and body[key] not in (None, [], "none"):
            raise APIError(400, f"{key} is not supported by this text-only runtime", param=key)
    if body.get("n", 1) != 1:
        raise APIError(400, "only n=1 is supported", param="n")
    if body.get("temperature") not in (None, 0, 0.0):
        raise APIError(400, "only greedy temperature=0 is supported", param="temperature")
    if body.get("top_p") not in (None, 1, 1.0):
        raise APIError(400, "top_p sampling is not implemented", param="top_p")
    if body.get("logprobs") not in (None, False):
        raise APIError(400, "logprobs are not implemented", param="logprobs")

    messages = normalize_messages(body.get("messages"))
    try:
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False
        )
    except Exception as error:
        raise APIError(400, f"chat template rejected messages: {error}", param="messages") from error
    prompt_ids = [int(token) for token in prompt_ids]

    requested = body.get("max_completion_tokens", body.get("max_tokens", config.default_max_tokens))
    if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
        raise APIError(400, "max_completion_tokens must be a positive integer",
                       param="max_completion_tokens")
    if len(prompt_ids) + requested > config.max_context_tokens:
        raise APIError(
            400,
            f"prompt ({len(prompt_ids)}) + completion ({requested}) exceeds "
            f"the {config.max_context_tokens}-token runtime limit",
            param="max_completion_tokens",
            code="context_length_exceeded",
        )

    stop = body.get("stop", [])
    if isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list) or len(stop) > 4 or not all(isinstance(item, str) and item for item in stop):
        raise APIError(400, "stop must be a string or an array of at most 4 non-empty strings", param="stop")

    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise APIError(400, "stream must be boolean", param="stream")
    stream_options = body.get("stream_options") or {}
    if not isinstance(stream_options, dict):
        raise APIError(400, "stream_options must be an object", param="stream_options")
    return prompt_ids, requested, stop, stream, bool(stream_options.get("include_usage", False))


def stop_view(text, stops, final=False):
    positions = [text.find(stop) for stop in stops]
    positions = [position for position in positions if position >= 0]
    if positions:
        return text[:min(positions)], True
    if final or not stops:
        return text, False
    hold = max(len(stop) for stop in stops) - 1
    return text[:-hold] if hold else text, False


def chunk(completion_id, created, model, delta, finish_reason=None, usage=None):
    result = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "logprobs": None,
                     "finish_reason": finish_reason}],
    }
    if usage is not None:
        result["usage"] = usage
    return result


def sse(data):
    return "data: " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n\n"


def create_app(config: Config, *, tokenizer=None, worker=None):
    owns_worker = worker is None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal tokenizer, worker
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = await asyncio.to_thread(AutoTokenizer.from_pretrained, config.model_path)
        if worker is None:
            worker = await asyncio.to_thread(
                EngineWorker, config.engine_path, config.weights_path,
                config.worker_start_timeout, config.worker_step_timeout
            )
        app.state.tokenizer = tokenizer
        app.state.worker = worker
        app.state.generation_lock = asyncio.Lock()
        try:
            yield
        finally:
            if owns_worker and worker is not None:
                await asyncio.to_thread(worker.close)

    app = FastAPI(title="qwen3x runtime", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(APIError)
    async def api_error_handler(_request, error):
        return JSONResponse(
            error_body(error.message, param=error.param, code=error.code,
                       error_type=error.error_type),
            status_code=error.status,
        )

    async def authenticate(request):
        if config.api_key is None:
            return
        header = request.headers.get("authorization", "")
        expected = "Bearer " + config.api_key
        if not hmac.compare_digest(header, expected):
            raise APIError(401, "invalid API key", code="invalid_api_key",
                           error_type="authentication_error")

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready():
        if not app.state.worker.alive:
            return JSONResponse({"status": "not ready"}, status_code=503)
        return {"status": "ready"}

    @app.get("/v1/models")
    async def models(request: Request):
        await authenticate(request)
        return {
            "object": "list",
            "data": [{"id": config.served_model_name, "object": "model",
                      "created": 0, "owned_by": "qwen3x"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        await authenticate(request)
        chunks = []
        body_bytes = 0
        async for data in request.stream():
            body_bytes += len(data)
            if body_bytes > config.max_request_bytes:
                raise APIError(413, "request body is too large", code="request_too_large")
            chunks.append(data)
        try:
            body = json.loads(b"".join(chunks))
        except Exception as error:
            raise APIError(400, "request body is not valid JSON") from error
        prompt_ids, max_tokens, stops, stream, include_usage = parse_request(
            body, config, app.state.tokenizer
        )
        lock = app.state.generation_lock
        if lock.locked():
            raise APIError(429, "the single-user inference worker is busy; retry later",
                           code="engine_busy", error_type="rate_limit_error")
        await lock.acquire()

        completion_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        started = time.monotonic()

        async def run_non_streaming():
            output_ids = []
            finish_reason = "length"
            try:
                token = await asyncio.to_thread(app.state.worker.start, prompt_ids)
                while token is not None and len(output_ids) < max_tokens:
                    if await request.is_disconnected():
                        raise EngineError("client disconnected")
                    if time.monotonic() - started > config.request_timeout:
                        raise EngineError("generation request timed out")
                    output_ids.append(token)
                    decoded = app.state.tokenizer.decode(output_ids, skip_special_tokens=True)
                    visible, stopped = stop_view(decoded, stops)
                    if stopped:
                        finish_reason = "stop"
                        break
                    if len(output_ids) == max_tokens:
                        break
                    token = await asyncio.to_thread(app.state.worker.next)
                else:
                    if token is None:
                        finish_reason = "stop"
                decoded = app.state.tokenizer.decode(output_ids, skip_special_tokens=True)
                text, stopped = stop_view(decoded, stops, final=True)
                if stopped:
                    finish_reason = "stop"
                usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": len(output_ids),
                         "total_tokens": len(prompt_ids) + len(output_ids)}
                LOG.info("request=%s prompt=%d completion=%d elapsed=%.3fs reason=%s",
                         completion_id, len(prompt_ids), len(output_ids),
                         time.monotonic() - started, finish_reason)
                return {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": config.served_model_name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text,
                                                           "refusal": None},
                                 "logprobs": None, "finish_reason": finish_reason}],
                    "usage": usage,
                }
            finally:
                try:
                    await asyncio.to_thread(app.state.worker.reset)
                finally:
                    lock.release()

        async def run_streaming():
            output_ids = []
            published = ""
            finish_reason = "length"
            try:
                yield sse(chunk(completion_id, created, config.served_model_name,
                                {"role": "assistant", "content": ""}))
                token = await asyncio.to_thread(app.state.worker.start, prompt_ids)
                while token is not None and len(output_ids) < max_tokens:
                    if await request.is_disconnected():
                        return
                    if time.monotonic() - started > config.request_timeout:
                        raise EngineError("generation request timed out")
                    output_ids.append(token)
                    decoded = app.state.tokenizer.decode(output_ids, skip_special_tokens=True)
                    visible, stopped = stop_view(decoded, stops)
                    if not visible.startswith(published):
                        raise EngineError("tokenizer streaming output rewrote already published text")
                    delta = visible[len(published):]
                    if delta:
                        yield sse(chunk(completion_id, created, config.served_model_name,
                                        {"content": delta}))
                        published = visible
                    if stopped:
                        finish_reason = "stop"
                        break
                    if len(output_ids) == max_tokens:
                        break
                    token = await asyncio.to_thread(app.state.worker.next)
                else:
                    if token is None:
                        finish_reason = "stop"

                decoded = app.state.tokenizer.decode(output_ids, skip_special_tokens=True)
                visible, stopped = stop_view(decoded, stops, final=True)
                if stopped:
                    finish_reason = "stop"
                if not visible.startswith(published):
                    raise EngineError("tokenizer final output rewrote already published text")
                tail = visible[len(published):]
                if tail:
                    yield sse(chunk(completion_id, created, config.served_model_name,
                                    {"content": tail}))
                yield sse(chunk(completion_id, created, config.served_model_name,
                                {}, finish_reason=finish_reason))
                usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": len(output_ids),
                         "total_tokens": len(prompt_ids) + len(output_ids)}
                if include_usage:
                    usage_chunk = chunk(completion_id, created, config.served_model_name, {})
                    usage_chunk["choices"] = []
                    usage_chunk["usage"] = usage
                    yield sse(usage_chunk)
                yield "data: [DONE]\n\n"
                LOG.info("request=%s prompt=%d completion=%d elapsed=%.3fs reason=%s",
                         completion_id, len(prompt_ids), len(output_ids),
                         time.monotonic() - started, finish_reason)
            except Exception as error:
                LOG.exception("streaming request failed: %s", completion_id)
                yield sse(error_body(str(error), error_type="server_error"))
                yield "data: [DONE]\n\n"
            finally:
                try:
                    await asyncio.to_thread(app.state.worker.reset)
                finally:
                    lock.release()

        if stream:
            return StreamingResponse(
                run_streaming(), media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            return await run_non_streaming()
        except EngineError as error:
            raise APIError(500, str(error), error_type="server_error") from error

    return app


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=HERE.parent / "models/Qwen3.5-0.8B")
    parser.add_argument("--engine", type=Path, default=HERE.parent / "02-cpu-0.8b/qwen35")
    parser.add_argument("--weights", type=Path,
                        default=HERE.parent / "02-cpu-0.8b/build/qwen35-0.8b.bin")
    parser.add_argument("--served-model-name", default="qwen3.5-0.8b")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--default-max-tokens", type=int, default=128)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--engine-step-timeout", type=float, default=600.0)
    parser.add_argument("--no-auth", action="store_true",
                        help="仅限可信本机开发；否则必须设置 QWEN_API_KEY")
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = None if args.no_auth else os.environ.get("QWEN_API_KEY")
    if not args.no_auth and not api_key:
        raise SystemExit("set QWEN_API_KEY or explicitly use --no-auth for local development")
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = Config(
        model_path=args.model,
        engine_path=args.engine,
        weights_path=args.weights,
        served_model_name=args.served_model_name,
        api_key=api_key,
        max_context_tokens=args.max_context_tokens,
        default_max_tokens=args.default_max_tokens,
        request_timeout=args.request_timeout,
        worker_step_timeout=args.engine_step_timeout,
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port, workers=1,
                log_level=args.log_level, proxy_headers=False)


if __name__ == "__main__":
    main()
