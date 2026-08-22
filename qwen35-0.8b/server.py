#!/usr/bin/env python3
"""OpenAI-compatible HTTP runtime over the in-process Qwen3.5 C ABI."""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.datastructures import MutableHeaders

from qwen35 import (
    Engine,
    EngineError,
    Q35_LOG_DEBUG,
    Q35_LOG_ERROR,
    Q35_LOG_INFO,
    Q35_LOG_WARN,
    SessionBusy,
    set_log_callback,
)


HERE = Path(__file__).resolve().parent
LOG = logging.getLogger("qwen35.runtime")
SESSION_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
REQUEST_ID_CONTEXT = ContextVar("request_id", default="-")
SESSION_ID_CONTEXT = ContextVar("session_id", default="-")

PYTHON_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}
NATIVE_LOG_LEVELS = {
    "debug": Q35_LOG_DEBUG,
    "info": Q35_LOG_INFO,
    "warning": Q35_LOG_WARN,
    "error": Q35_LOG_ERROR,
}
NATIVE_TO_PYTHON_LEVEL = {
    Q35_LOG_DEBUG: logging.DEBUG,
    Q35_LOG_INFO: logging.INFO,
    Q35_LOG_WARN: logging.WARNING,
    Q35_LOG_ERROR: logging.ERROR,
}


class LogContextFilter(logging.Filter):
    def filter(self, record):
        request_id = REQUEST_ID_CONTEXT.get()
        session_id = SESSION_ID_CONTEXT.get()
        context = []
        if request_id != "-":
            context.append(f"request_id={request_id}")
        if session_id != "-":
            context.append(f"session_id={session_id}")
        record.context = (" ".join(context) + " ") if context else ""
        record.tid = threading.get_native_id()
        record.level_lower = record.levelname.lower()
        if not hasattr(record, "source"):
            record.source = f"{record.filename}:{record.lineno}"
        return True


def configure_logging(level: str, log_file: Path | None,
                      max_megabytes: int, backups: int) -> None:
    numeric_level = PYTHON_LOG_LEVELS[level]
    formatter = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] [%(tid)d] [%(level_lower)s] [%(source)s] "
        "%(context)s%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    context_filter = LogContextFilter()

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(
            log_file,
            maxBytes=max_megabytes * 1024 * 1024,
            backupCount=backups,
            encoding="utf-8",
        ))
    for handler in handlers:
        handler.setLevel(numeric_level)
        handler.setFormatter(formatter)
        handler.addFilter(context_filter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric_level)
    for handler in handlers:
        root.addHandler(handler)


def native_log(level: int, file: str, line: int, message: str) -> None:
    logging.getLogger("qwen35.native").log(
        NATIVE_TO_PYTHON_LEVEL.get(level, logging.INFO),
        message,
        extra={"source": f"{file}:{line}"},
    )


class RequestContextMiddleware:
    """Give every HTTP request one server-issued ID, including streaming."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = "req-" + uuid.uuid4().hex
        context = REQUEST_ID_CONTEXT.set(request_id)
        started = time.monotonic()
        status = 500

        async def send_with_request_id(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message).append("X-Request-Id", request_id)
            await send(message)

        LOG.info("request started method=%s path=%s", scope["method"], scope["path"])
        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception:
            LOG.exception("request failed method=%s path=%s",
                          scope["method"], scope["path"])
            raise
        finally:
            LOG.info("request completed method=%s path=%s status=%d elapsed=%.3fs",
                     scope["method"], scope["path"], status,
                     time.monotonic() - started)
            REQUEST_ID_CONTEXT.reset(context)


class APIError(Exception):
    def __init__(self, status: int, message: str, *, param=None,
                 code=None, error_type="invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.message = message
        self.param = param
        self.code = code
        self.error_type = error_type


@dataclass(frozen=True)
class Config:
    tokenizer_path: Path
    library_path: Path
    bin_path: Path
    served_model_name: str = "qwen3.5-0.8b"
    api_key: Optional[str] = None
    slot_count: int = 2
    max_context_tokens: int = 4096
    default_max_tokens: int = 128
    max_request_bytes: int = 1024 * 1024
    request_timeout: float = 600.0
    native_log_level: int = Q35_LOG_INFO


def error_body(message, *, param=None, code=None, error_type="invalid_request_error"):
    return {"error": {"message": message, "type": error_type, "param": param, "code": code}}


def text_content(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if (not isinstance(part, dict) or part.get("type") != "text"
                    or not isinstance(part.get("text"), str)):
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
        role = "system" if message.get("role") == "developer" else message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise APIError(400, f"unsupported message role: {role!r}", param=f"messages[{index}].role")
        if role == "system" and index != 0:
            raise APIError(400, "system/developer message must be first", param="messages")
        result.append({"role": role, "content": text_content(message)})
    if not any(message["role"] == "user" for message in result):
        raise APIError(400, "messages must contain a user message", param="messages")
    return result


def parse_request(body, request: Request, config: Config, tokenizer):
    if not isinstance(body, dict):
        raise APIError(400, "request body must be a JSON object")
    if body.get("model") != config.served_model_name:
        raise APIError(404, f"model {body.get('model')!r} not found", param="model",
                       code="model_not_found")
    for key in ("tools", "tool_choice", "functions", "function_call", "response_format"):
        if key in body and body[key] not in (None, [], "none"):
            raise APIError(400, f"{key} is not supported", param=key)
    if body.get("n", 1) != 1:
        raise APIError(400, "only n=1 is supported", param="n")
    temperature = body.get("temperature", 1.0)
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not 0 <= temperature <= 2):
        raise APIError(400, "temperature must be between 0 and 2", param="temperature")
    top_p = body.get("top_p", 1.0)
    if (isinstance(top_p, bool) or not isinstance(top_p, (int, float))
            or not 0 < top_p <= 1):
        raise APIError(400, "top_p must be in (0, 1]", param="top_p")
    seed = body.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise APIError(400, "seed must be an integer", param="seed")
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

    maximum = body.get("max_completion_tokens", body.get("max_tokens", config.default_max_tokens))
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise APIError(400, "max_completion_tokens must be a positive integer",
                       param="max_completion_tokens")
    if len(prompt_ids) + maximum > config.max_context_tokens:
        raise APIError(400, "prompt and completion exceed the Session context",
                       param="max_completion_tokens", code="context_length_exceeded")

    stops = body.get("stop", [])
    if isinstance(stops, str):
        stops = [stops]
    if (not isinstance(stops, list) or len(stops) > 4
            or not all(isinstance(item, str) and item for item in stops)):
        raise APIError(400, "stop must contain at most 4 non-empty strings", param="stop")
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise APIError(400, "stream must be boolean", param="stream")
    stream_options = body.get("stream_options") or {}
    if not isinstance(stream_options, dict):
        raise APIError(400, "stream_options must be an object", param="stream_options")

    body_session = body.get("session_id")
    header_session = request.headers.get("x-qwen-session-id")
    if body_session is not None and header_session is not None and body_session != header_session:
        raise APIError(400, "body and header session_id disagree", param="session_id")
    session_id = body_session if body_session is not None else header_session
    persistent = session_id is not None
    if session_id is None:
        session_id = "ephemeral-" + uuid.uuid4().hex
    if not isinstance(session_id, str) or not SESSION_ID.fullmatch(session_id):
        raise APIError(400, "session_id must be 1..128 safe ASCII characters", param="session_id")

    return {
        "prompt_ids": prompt_ids,
        "max_tokens": maximum,
        "stops": stops,
        "stream": stream,
        "include_usage": bool(stream_options.get("include_usage", False)),
        "session_id": session_id,
        "persistent": persistent,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "rng": seed if seed is not None else secrets.randbits(64),
    }


def stop_view(text: str, stops: list[str], *, final=False):
    positions = [text.find(stop) for stop in stops]
    positions = [position for position in positions if position >= 0]
    if positions:
        return text[:min(positions)], True
    if final or not stops:
        return text, False
    hold = max(len(stop) for stop in stops) - 1
    return text[:-hold] if hold else text, False


def sse(data) -> str:
    return "data: " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n\n"


def stream_chunk(completion_id, created, model, session_id, delta,
                 finish_reason=None, usage=None):
    result = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "session_id": session_id,
        "choices": [{"index": 0, "delta": delta, "logprobs": None,
                     "finish_reason": finish_reason}],
    }
    if usage is not None:
        result["usage"] = usage
    return result


async def next_token(session, temperature, top_p, rng, input_token=None):
    if input_token is not None:
        await asyncio.to_thread(session.eval, input_token)
    if temperature == 0:
        return await asyncio.to_thread(session.argmax), rng
    return await asyncio.to_thread(session.sample, temperature, top_p, rng)


def create_app(config: Config, *, tokenizer=None, manager=None):
    owns_runtime = manager is None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal tokenizer, manager
        engine = None
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained, config.tokenizer_path
            )
        if manager is None:
            await asyncio.to_thread(
                set_log_callback, config.library_path, native_log, config.native_log_level
            )
            try:
                engine = await asyncio.to_thread(
                    Engine, config.library_path, config.bin_path
                )
                manager = await asyncio.to_thread(
                    engine.create_session_manager, config.slot_count, config.max_context_tokens
                )
            except Exception:
                await asyncio.to_thread(set_log_callback, config.library_path, None)
                raise
        app.state.tokenizer = tokenizer
        app.state.manager = manager
        app.state.engine = engine if engine is not None else manager.engine
        LOG.info("server ready model=%s slots=%d context=%d", config.served_model_name,
                 config.slot_count, config.max_context_tokens)
        try:
            yield
        finally:
            LOG.info("server shutting down")
            if owns_runtime:
                await asyncio.to_thread(manager.close)
                engine.close()
                set_log_callback(config.library_path, None)

    app = FastAPI(title="qwen35 runtime", version="0.1.0", lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware)

    @app.exception_handler(APIError)
    async def api_error_handler(_request, error):
        return JSONResponse(error_body(error.message, param=error.param, code=error.code,
                                       error_type=error.error_type), status_code=error.status)

    async def authenticate(request):
        if config.api_key is None:
            return
        expected = "Bearer " + config.api_key
        if not hmac.compare_digest(request.headers.get("authorization", ""), expected):
            raise APIError(401, "invalid API key", code="invalid_api_key",
                           error_type="authentication_error")

    async def read_json(request):
        chunks, size = [], 0
        async for data in request.stream():
            size += len(data)
            if size > config.max_request_bytes:
                raise APIError(413, "request body is too large", code="request_too_large")
            chunks.append(data)
        try:
            return json.loads(b"".join(chunks))
        except Exception as error:
            raise APIError(400, "request body is not valid JSON") from error

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready():
        return {"status": "ready", "slots": app.state.manager.session_count}

    @app.get("/v1/models")
    async def models(request: Request):
        await authenticate(request)
        return {"object": "list", "data": [{"id": config.served_model_name,
                "object": "model", "created": 0, "owned_by": "qwen3x"}]}

    @app.delete("/v1/sessions/{session_id}")
    async def delete_session(session_id: str, request: Request):
        await authenticate(request)
        session_context = SESSION_ID_CONTEXT.set(session_id)
        try:
            removed = await asyncio.to_thread(app.state.manager.forget, session_id)
        except SessionBusy as error:
            raise APIError(409, str(error), code="session_busy") from error
        finally:
            SESSION_ID_CONTEXT.reset(session_context)
        if not removed:
            raise APIError(404, "session not found", code="session_not_found")
        return {"id": session_id, "deleted": True}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        await authenticate(request)
        parsed = parse_request(await read_json(request), request, config, app.state.tokenizer)
        session_context = SESSION_ID_CONTEXT.set(parsed["session_id"])
        try:
            native_session_id = parsed["session_id"] if parsed["persistent"] else None
            session = await asyncio.to_thread(
                app.state.manager.acquire, native_session_id, parsed["prompt_ids"]
            )
        except SessionBusy as error:
            SESSION_ID_CONTEXT.reset(session_context)
            raise APIError(429, str(error), code="engine_busy",
                           error_type="rate_limit_error") from error
        except Exception:
            SESSION_ID_CONTEXT.reset(session_context)
            raise

        completion_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        started = time.monotonic()
        headers = {"X-Qwen-Session-Id": parsed["session_id"]}
        LOG.info("generation started completion_id=%s prompt_tokens=%d max_tokens=%d stream=%s",
                 completion_id, len(parsed["prompt_ids"]), parsed["max_tokens"],
                 parsed["stream"])

        async def release(error=None):
            await asyncio.to_thread(app.state.manager.release, session, keep=error is None)

        async def generate_tokens():
            await asyncio.to_thread(session.sync, parsed["prompt_ids"])
            output_ids = []
            finish_reason = "length"
            token, rng = await next_token(
                session, parsed["temperature"], parsed["top_p"], parsed["rng"]
            )
            while len(output_ids) < parsed["max_tokens"]:
                if await request.is_disconnected():
                    raise EngineError("client disconnected")
                if time.monotonic() - started > config.request_timeout:
                    raise EngineError("generation request timed out")
                if app.state.engine.token_is_stop(token):
                    finish_reason = "stop"
                    break
                output_ids.append(token)
                decoded = app.state.tokenizer.decode(output_ids, skip_special_tokens=True)
                _visible, stopped = stop_view(decoded, parsed["stops"])
                if stopped:
                    finish_reason = "stop"
                    break
                if len(output_ids) == parsed["max_tokens"]:
                    break
                token, rng = await next_token(
                    session, parsed["temperature"], parsed["top_p"], rng, token
                )
            return output_ids, finish_reason

        async def non_streaming():
            try:
                output_ids, finish_reason = await generate_tokens()
                decoded = app.state.tokenizer.decode(output_ids, skip_special_tokens=True)
                text, stopped = stop_view(decoded, parsed["stops"], final=True)
                if stopped:
                    finish_reason = "stop"
                usage = {"prompt_tokens": len(parsed["prompt_ids"]),
                         "completion_tokens": len(output_ids),
                         "total_tokens": len(parsed["prompt_ids"]) + len(output_ids)}
                return {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": config.served_model_name,
                    "session_id": parsed["session_id"],
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text,
                                                               "refusal": None},
                                 "logprobs": None, "finish_reason": finish_reason}],
                    "usage": usage,
                }
            except Exception as error:
                await release(error)
                raise
            finally:
                LOG.info("generation completed completion_id=%s elapsed=%.3fs",
                         completion_id, time.monotonic() - started)

        async def streaming():
            stream_context = SESSION_ID_CONTEXT.set(parsed["session_id"])
            failure = None
            output_ids, published = [], ""
            finish_reason = "length"
            try:
                yield sse(stream_chunk(completion_id, created, config.served_model_name,
                                       parsed["session_id"], {"role": "assistant", "content": ""}))
                await asyncio.to_thread(session.sync, parsed["prompt_ids"])
                token, rng = await next_token(
                    session, parsed["temperature"], parsed["top_p"], parsed["rng"]
                )
                while len(output_ids) < parsed["max_tokens"]:
                    if await request.is_disconnected():
                        raise EngineError("client disconnected")
                    if time.monotonic() - started > config.request_timeout:
                        raise EngineError("generation request timed out")
                    if app.state.engine.token_is_stop(token):
                        finish_reason = "stop"
                        break
                    output_ids.append(token)
                    decoded = app.state.tokenizer.decode(output_ids, skip_special_tokens=True)
                    visible, stopped = stop_view(decoded, parsed["stops"])
                    if not visible.startswith(published):
                        raise EngineError("tokenizer rewrote already streamed text")
                    delta = visible[len(published):]
                    if delta:
                        yield sse(stream_chunk(completion_id, created, config.served_model_name,
                                               parsed["session_id"], {"content": delta}))
                        published = visible
                    if stopped:
                        finish_reason = "stop"
                        break
                    if len(output_ids) == parsed["max_tokens"]:
                        break
                    token, rng = await next_token(
                        session, parsed["temperature"], parsed["top_p"], rng, token
                    )

                decoded = app.state.tokenizer.decode(output_ids, skip_special_tokens=True)
                visible, stopped = stop_view(decoded, parsed["stops"], final=True)
                if stopped:
                    finish_reason = "stop"
                if not visible.startswith(published):
                    raise EngineError("tokenizer rewrote already streamed text")
                if visible != published:
                    yield sse(stream_chunk(completion_id, created, config.served_model_name,
                                           parsed["session_id"], {"content": visible[len(published):]}))
                yield sse(stream_chunk(completion_id, created, config.served_model_name,
                                       parsed["session_id"], {}, finish_reason=finish_reason))
                if parsed["include_usage"]:
                    usage = {"prompt_tokens": len(parsed["prompt_ids"]),
                             "completion_tokens": len(output_ids),
                             "total_tokens": len(parsed["prompt_ids"]) + len(output_ids)}
                    item = stream_chunk(completion_id, created, config.served_model_name,
                                        parsed["session_id"], {})
                    item["choices"] = []
                    item["usage"] = usage
                    yield sse(item)
                yield "data: [DONE]\n\n"
            except Exception as error:
                failure = error
                LOG.exception("generation failed completion_id=%s", completion_id)
                yield sse(error_body(str(error), error_type="server_error"))
                yield "data: [DONE]\n\n"
            finally:
                try:
                    await release(failure)
                    LOG.info("generation completed completion_id=%s elapsed=%.3fs",
                             completion_id, time.monotonic() - started)
                finally:
                    SESSION_ID_CONTEXT.reset(stream_context)

        if parsed["stream"]:
            response = StreamingResponse(streaming(), media_type="text/event-stream", headers={
                **headers, "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
            })
            SESSION_ID_CONTEXT.reset(session_context)
            return response
        try:
            response = await non_streaming()
            await release()
            return JSONResponse(response, headers=headers)
        except EngineError as error:
            raise APIError(500, str(error), error_type="server_error") from error
        finally:
            SESSION_ID_CONTEXT.reset(session_context)

    return app


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path,
                        default=HERE.parent / "models/Qwen3.5-0.8B")
    parser.add_argument("--library", type=Path, default=HERE / "build/libqwen35.so")
    parser.add_argument("--bin", type=Path, default=HERE / "build/qwen35-0.8b.bin")
    parser.add_argument("--served-model-name", default="qwen3.5-0.8b")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--default-max-tokens", type=int, default=128)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--log-level", choices=PYTHON_LOG_LEVELS, default="debug")
    parser.add_argument("--log-file", type=Path, default=HERE / "logs/qwen35.log")
    parser.add_argument("--log-max-mb", type=int, default=20)
    parser.add_argument("--log-backups", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.log_max_mb <= 0:
        raise SystemExit("--log-max-mb must be positive")
    if args.log_backups < 0:
        raise SystemExit("--log-backups must not be negative")
    configure_logging(args.log_level, args.log_file, args.log_max_mb, args.log_backups)
    # Authentication is opt-in: setting QWEN_API_KEY enables Bearer auth.
    api_key = os.environ.get("QWEN_API_KEY")
    config = Config(
        tokenizer_path=args.tokenizer,
        library_path=args.library,
        bin_path=args.bin,
        served_model_name=args.served_model_name,
        api_key=api_key,
        slot_count=args.slots,
        max_context_tokens=args.max_context_tokens,
        default_max_tokens=args.default_max_tokens,
        request_timeout=args.request_timeout,
        native_log_level=NATIVE_LOG_LEVELS[args.log_level],
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port, workers=1,
                log_level=args.log_level, proxy_headers=False,
                log_config=None, access_log=False)


if __name__ == "__main__":
    main()
