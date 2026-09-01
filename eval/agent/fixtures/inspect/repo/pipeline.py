"""Small request pipeline used by the coding-agent inspect fixture.

The file is deliberately longer than one read_file call.  Its code is simple,
but the caller, orchestration function, registry, and response encoder live in
different sections so an answer based on only the first section is incomplete.
"""

from dataclasses import dataclass, field
import json
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Boundary data


@dataclass
class Packet:
    """Bytes accepted from the transport plus a request correlation id."""

    request_id: str
    body: bytes


@dataclass
class Request:
    """Validated input understood by handlers."""

    request_id: str
    operation: str
    values: list[int]
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class Response:
    """Internal response; the transport encoder owns JSON formatting."""

    request_id: str
    status: int
    result: Optional[int] = None
    error: Optional[str] = None


class RequestError(ValueError):
    """Expected client input error with a stable public message."""


class HandlerError(RuntimeError):
    """Expected operation error with a stable public message."""


def error_response(request_id: str, status: int, message: str) -> Response:
    return Response(request_id=request_id, status=status, error=message)


# ---------------------------------------------------------------------------
# Parse and normalization


def parse_packet(packet: Packet) -> dict:
    """Decode transport bytes, preserving only JSON-object requests."""

    try:
        text = packet.body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RequestError("body is not UTF-8") from error

    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise RequestError("body is not valid JSON") from error

    if not isinstance(value, dict):
        raise RequestError("request must be a JSON object")
    return value


def require_operation(value: dict) -> str:
    operation = value.get("operation")
    if not isinstance(operation, str) or not operation:
        raise RequestError("operation must be a non-empty string")
    return operation.strip().lower()


def require_values(value: dict) -> list[int]:
    values = value.get("values")
    if not isinstance(values, list) or not values:
        raise RequestError("values must be a non-empty array")
    if any(isinstance(item, bool) or not isinstance(item, int)
           for item in values):
        raise RequestError("values must contain only integers")
    return list(values)


def optional_labels(value: dict) -> dict[str, str]:
    labels = value.get("labels", {})
    if not isinstance(labels, dict):
        raise RequestError("labels must be an object")
    if any(not isinstance(key, str) or not isinstance(item, str)
           for key, item in labels.items()):
        raise RequestError("labels must map strings to strings")
    return dict(labels)


def normalize_request(packet: Packet, value: dict) -> Request:
    """Turn untrusted parsed fields into the handler-facing Request."""

    operation = require_operation(value)
    values = require_values(value)
    labels = optional_labels(value)
    return Request(
        request_id=packet.request_id,
        operation=operation,
        values=values,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# Handlers and registry


Handler = Callable[[Request], int]


def add_values(request: Request) -> int:
    return sum(request.values)


def multiply_values(request: Request) -> int:
    result = 1
    for value in request.values:
        result *= value
    return result


def minimum_value(request: Request) -> int:
    return min(request.values)


def maximum_value(request: Request) -> int:
    return max(request.values)


class HandlerRegistry:
    """Own operation-name lookup, leaving orchestration free of branches."""

    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}

    def add(self, name: str, handler: Handler) -> None:
        if not name:
            raise ValueError("handler name must not be empty")
        if name in self._handlers:
            raise ValueError(f"duplicate handler: {name}")
        self._handlers[name] = handler

    def dispatch(self, request: Request) -> int:
        handler = self._handlers.get(request.operation)
        if handler is None:
            raise HandlerError(f"unknown operation: {request.operation}")
        return handler(request)


def default_registry() -> HandlerRegistry:
    registry = HandlerRegistry()
    registry.add("add", add_values)
    registry.add("multiply", multiply_values)
    registry.add("minimum", minimum_value)
    registry.add("maximum", maximum_value)
    return registry


# ---------------------------------------------------------------------------
# Orchestration


def run_request(packet: Packet, registry: HandlerRegistry) -> Response:
    """Run one request and translate expected failures into Responses."""

    try:
        parsed = parse_packet(packet)
        request = normalize_request(packet, parsed)
    except RequestError as error:
        return error_response(packet.request_id, 400, str(error))

    try:
        result = registry.dispatch(request)
    except HandlerError as error:
        return error_response(request.request_id, 404, str(error))
    except ArithmeticError:
        return error_response(request.request_id, 422, "arithmetic failed")

    return Response(
        request_id=request.request_id,
        status=200,
        result=result,
    )


# ---------------------------------------------------------------------------
# Encoding and transport entry point


def response_document(response: Response) -> dict:
    document: dict[str, object] = {
        "request_id": response.request_id,
        "status": response.status,
    }
    if response.error is not None:
        document["error"] = response.error
    else:
        document["result"] = response.result
    return document


def encode_response(response: Response) -> bytes:
    """Serialize internal response state only at the transport boundary."""

    document = response_document(response)
    return json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def serve_once(
    request_id: str,
    body: bytes,
    registry: Optional[HandlerRegistry] = None,
) -> tuple[int, bytes]:
    """Transport adapter used by the fixture's imaginary HTTP server."""

    active_registry = registry if registry is not None else default_registry()
    packet = Packet(request_id=request_id, body=body)
    response = run_request(packet, active_registry)
    encoded = encode_response(response)
    return response.status, encoded


def example() -> tuple[int, bytes]:
    """Keep one concrete top-level caller visible at the end of the file."""

    body = json.dumps({
        "operation": "add",
        "values": [2, 3, 5],
        "labels": {"source": "example"},
    }).encode("utf-8")
    return serve_once("example-1", body)


if __name__ == "__main__":
    status, payload = example()
    print(status, payload.decode("utf-8"))
