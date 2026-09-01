"""Qwen tool-call boundary helpers shared by the reference evaluation server."""

from __future__ import annotations

import copy
import json


class ToolCallParseError(ValueError):
    pass


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI encoded arguments to objects expected by Qwen's template."""

    normalized = copy.deepcopy(messages)
    for message in normalized:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as error:
                    raise ToolCallParseError(
                        "tool call arguments contain malformed JSON"
                    ) from error
            if not isinstance(arguments, dict):
                raise ToolCallParseError("tool call arguments must be an object")
            function["arguments"] = arguments
    return normalized


def _trim_one_newline(value: str) -> str:
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def _coerce_arguments(
    calls: list[dict[str, object]], tools: list[dict] | None
) -> None:
    if not tools:
        return
    schemas = {}
    for tool in tools:
        function = tool.get("function") or {}
        name = function.get("name")
        properties = (function.get("parameters") or {}).get("properties") or {}
        if isinstance(name, str) and isinstance(properties, dict):
            schemas[name] = properties
    for call in calls:
        properties = schemas.get(call["name"], {})
        arguments = call["arguments"]
        for name, value in list(arguments.items()):
            expected = (properties.get(name) or {}).get("type")
            if expected in (None, "string"):
                continue
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            valid = (
                (expected == "integer" and isinstance(parsed, int) and
                 not isinstance(parsed, bool)) or
                (expected == "number" and isinstance(parsed, (int, float)) and
                 not isinstance(parsed, bool)) or
                (expected == "boolean" and isinstance(parsed, bool)) or
                (expected == "object" and isinstance(parsed, dict)) or
                (expected == "array" and isinstance(parsed, list)) or
                (expected == "null" and parsed is None)
            )
            if valid:
                arguments[name] = parsed


def parse_generated_tool_calls(
    text: str, tools: list[dict] | None = None
) -> tuple[str, list[dict[str, object]]]:
    """Parse Qwen's XML-like generated calls, matching render.cpp semantics."""

    tool_start = "<tool_call>"
    tool_end = "</tool_call>"
    function_start = "<function="
    function_end_tag = "</function>"
    parameter_start = "<parameter="
    parameter_end_tag = "</parameter>"

    first = text.find(tool_start)
    if first < 0:
        first = text.find(function_start)
    if first < 0:
        return text, []

    content = text[:first].rstrip()
    calls: list[dict[str, object]] = []
    cursor = first
    while True:
        function = text.find(function_start, cursor)
        if function < 0:
            break
        name_begin = function + len(function_start)
        name_end = text.find(">", name_begin)
        function_end = (
            text.find(function_end_tag, name_end + 1) if name_end >= 0 else -1
        )
        if name_end < 0 or function_end < 0:
            raise ToolCallParseError("incomplete generated tool call")
        name = text[name_begin:name_end].strip()
        if not name:
            raise ToolCallParseError("generated tool name is empty")

        arguments: dict[str, str] = {}
        argument = name_end + 1
        while True:
            parameter = text.find(parameter_start, argument)
            if parameter < 0 or parameter >= function_end:
                break
            parameter_name = parameter + len(parameter_start)
            parameter_name_end = text.find(">", parameter_name)
            if parameter_name_end < 0 or parameter_name_end >= function_end:
                raise ToolCallParseError("incomplete generated parameter name")
            value_begin = parameter_name_end + 1
            next_parameter = text.find(parameter_start, value_begin)
            parameter_end = text.find(parameter_end_tag, value_begin)
            if (next_parameter >= 0 and next_parameter < function_end and
                    (parameter_end < 0 or next_parameter < parameter_end)):
                value_end = next_parameter
                argument = next_parameter
            else:
                if parameter_end < 0 or parameter_end > function_end:
                    raise ToolCallParseError("incomplete generated parameter value")
                value_end = parameter_end
                argument = parameter_end + len(parameter_end_tag)

            key = text[parameter_name:parameter_name_end]
            if not key:
                raise ToolCallParseError("generated parameter name is empty")
            arguments[key] = _trim_one_newline(text[value_begin:value_end])

        calls.append({"name": name, "arguments": arguments})
        cursor = function_end + len(function_end_tag)

    if not calls:
        raise ToolCallParseError("generated tool wrapper has no function")
    if text.startswith(tool_start, first) and text.rfind(tool_end) < 0:
        raise ToolCallParseError("incomplete generated tool wrapper")
    _coerce_arguments(calls, tools)
    return content, calls
