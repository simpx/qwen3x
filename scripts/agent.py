#!/usr/bin/env python3
"""Minimal read/write/bash agent for a local qwen35 server."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request


DEFAULT_MODEL = "qwen3.5-0.8b"
DEFAULT_URL = "http://127.0.0.1:8000/v1/chat/completions"

SYSTEM_PROMPT = """You are a small local coding agent.
The current working directory is the repository the user wants you to work on.
All tools already start there. Run commands directly, for example `git show`.
Never use `cd` or search other directories unless the user explicitly asks.
Use the available tools when you need to inspect or change files.
Read large files in sections and request only the lines you need.
When asked to explain or review a whole file, keep reading sections to its end.
After using tools, finish with a plain-text answer instead of a tool call.
Keep tool calls and final answers concise."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file relative to the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path"},
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read, starting at 1",
                        "default": 1,
                    },
                    "line_count": {
                        "type": "integer",
                        "description": "Number of lines to read",
                        "default": 200,
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a complete UTF-8 text file relative to the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path"},
                    "content": {"type": "string", "description": "Complete file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run one Bash command in the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash command"},
                },
                "required": ["command"],
            },
        },
    },
]


class AgentError(RuntimeError):
    pass


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Run a minimal coding agent against a qwen35 server."
    )
    parser.add_argument("prompt", nargs="?", help="one-shot task; omit for a REPL")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--api-key", default=os.environ.get("QWEN_API_KEY"))
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--command-timeout", type=float, default=30.0)
    parser.add_argument("--max-output", type=int, default=16000)
    parser.add_argument("-y", "--yes", action="store_true",
                        help="approve write_file and bash without prompting")
    return parser


def request_body(args, messages):
    return {
        "model": args.model,
        "messages": messages,
        "tools": TOOLS,
        "stream": False,
        "temperature": 0,
        "max_completion_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def completion(args, messages):
    body = json.dumps(
        request_body(args, messages), ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = "Bearer " + args.api_key
    request = urllib.request.Request(args.url, body, headers, method="POST")
    try:
        with urllib.request.urlopen(request) as response:
            value = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise AgentError(f"HTTP {error.code}: {detail.strip()}") from error
    except urllib.error.URLError as error:
        raise AgentError(f"request failed: {error.reason}") from error
    except json.JSONDecodeError as error:
        raise AgentError(f"invalid response JSON: {error.msg}") from error

    if "error" in value:
        raise AgentError(value["error"].get("message", "server returned an error"))
    try:
        return value["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise AgentError("response does not contain an assistant message") from error


def resolve_path(cwd, value):
    if not isinstance(value, str) or not value:
        raise AgentError("path must be a non-empty string")
    root = Path(cwd).resolve()
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise AgentError(f"path escapes working directory: {value}") from error
    return path


def limit_output(text, maximum):
    if len(text) <= maximum:
        return text
    half = maximum // 2
    removed = len(text) - half * 2
    return text[:half] + f"\n... {removed} characters omitted ...\n" + text[-half:]


def approve(args, description, input_fn=input):
    if args.yes:
        return True
    try:
        answer = input_fn(f"Allow {description}? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def execute_tool(args, name, arguments, input_fn=input):
    if not isinstance(arguments, dict):
        raise AgentError("tool arguments must be a JSON object")

    if name == "read_file":
        path = resolve_path(args.cwd, arguments.get("path", ""))
        start_line = arguments.get("start_line", 1)
        line_count = arguments.get("line_count", 200)
        if (not isinstance(start_line, int) or isinstance(start_line, bool)
                or start_line <= 0):
            raise AgentError("start_line must be a positive integer")
        if (not isinstance(line_count, int) or isinstance(line_count, bool)
                or line_count <= 0):
            raise AgentError("line_count must be a positive integer")
        try:
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        except (OSError, UnicodeError) as error:
            raise AgentError(f"cannot read {path}: {error}") from error
        total = len(lines)
        if start_line > total and total > 0:
            raise AgentError(
                f"start_line {start_line} is past end of file ({total} lines)"
            )
        first = start_line - 1
        selected = lines[first:first + line_count]
        end_line = first + len(selected)
        relative = path.relative_to(Path(args.cwd).resolve())
        if total == 0:
            return f"{relative}: empty file"
        content = "".join(selected)
        header = f"{relative}: lines {start_line}-{end_line} of {total}\n"
        footer = ""
        if end_line < total:
            separator = "" if content.endswith("\n") else "\n"
            footer = f"{separator}next_line: {end_line + 1}"
        result = header + content + footer
        if len(result) > args.max_output:
            raise AgentError(
                "selected lines exceed --max-output; request fewer lines"
            )
        return result

    if name == "write_file":
        path = resolve_path(args.cwd, arguments.get("path", ""))
        content = arguments.get("content")
        if not isinstance(content, str):
            raise AgentError("write_file content must be a string")
        relative = path.relative_to(Path(args.cwd).resolve())
        size = len(content.encode("utf-8"))
        if not approve(args, f"write_file {relative} ({size} bytes)", input_fn):
            return "DENIED: user rejected write_file"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as error:
            raise AgentError(f"cannot write {path}: {error.strerror}") from error
        return f"Wrote {relative} ({size} bytes)"

    if name == "bash":
        command = arguments.get("command")
        if not isinstance(command, str) or not command:
            raise AgentError("bash command must be a non-empty string")
        if not approve(args, f"bash: {command}", input_fn):
            return "DENIED: user rejected bash"
        try:
            result = subprocess.run(
                ["bash", "-lc", command], cwd=Path(args.cwd).resolve(),
                capture_output=True, text=True, timeout=args.command_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise AgentError(
                f"bash timed out after {args.command_timeout:g} seconds"
            ) from error
        output = f"exit_code={result.returncode}\n"
        if result.stdout:
            output += "stdout:\n" + result.stdout
        if result.stderr:
            output += "stderr:\n" + result.stderr
        return limit_output(output, args.max_output)

    raise AgentError(f"unknown tool: {name}")


def run_task(args, messages, prompt, complete=completion, input_fn=input):
    messages.append({"role": "user", "content": prompt})
    for _ in range(args.max_turns):
        message = complete(args, messages)
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            return message.get("content") or ""

        for call in calls:
            call_id = call.get("id", "")
            function = call.get("function") or {}
            name = function.get("name", "")
            raw_arguments = function.get("arguments", "{}")
            print(f"tool: {name}({raw_arguments})", file=sys.stderr)
            try:
                arguments = json.loads(raw_arguments)
                result = execute_tool(args, name, arguments, input_fn)
            except (json.JSONDecodeError, AgentError) as error:
                result = "ERROR: " + str(error)
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": result,
            })
    raise AgentError(f"agent exceeded {args.max_turns} model turns")


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    if args.max_tokens <= 0 or args.max_turns <= 0 or args.max_output <= 0:
        parser.error("limits must be positive")
    if args.command_timeout <= 0:
        parser.error("--command-timeout must be positive")
    try:
        args.cwd = str(Path(args.cwd).resolve(strict=True))
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if args.prompt is not None:
            answer = run_task(args, messages, args.prompt)
            if answer:
                print(answer)
            return 0

        while True:
            try:
                prompt = input("you> ")
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not prompt.strip():
                continue
            answer = run_task(args, messages, prompt)
            print("agent> " + answer)
    except (AgentError, OSError) as error:
        print(f"agent.py: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
