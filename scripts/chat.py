#!/usr/bin/env python3
"""Thin command-line client for the qwen35 Chat Completions endpoint."""

import argparse
import json
import os
import shlex
import sys
import urllib.error
import urllib.request


DEFAULT_MODEL = "qwen3.5-0.8b"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


class ChatError(ValueError):
    pass


class TraceAction(argparse.Action):
    """Preserve the order shared by -s, -u, -a, -c and -t."""

    def __call__(self, parser, namespace, values, option_string=None):
        events = getattr(namespace, self.dest, None)
        if events is None:
            events = []
            setattr(namespace, self.dest, events)
        events.append((option_string, values))


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Send a message trace to a local OpenAI-compatible qwen35 server."
    )
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("-p", "--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--api-key", default=os.environ.get("QWEN_API_KEY"))
    parser.add_argument("--tools", metavar="FILE",
                        help="JSON value for the OpenAI tools field")
    parser.add_argument("--think", action="store_true",
                        help="enable Qwen3.5 thinking")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--raw", action="store_true",
                        help="print the unprocessed HTTP response")
    parser.add_argument("--dry-run", action="store_true",
                        help="print an executable curl command without sending")
    parser.add_argument("--max-tokens", type=int, metavar="N")
    parser.add_argument("--temperature", type=float)

    parser.set_defaults(trace=[])
    parser.add_argument("-s", dest="trace", action=TraceAction, metavar="TEXT",
                        help="append a system message")
    parser.add_argument("-u", dest="trace", action=TraceAction, metavar="TEXT",
                        help="append a user message")
    parser.add_argument("-a", dest="trace", action=TraceAction, metavar="TEXT",
                        help="append an assistant message")
    parser.add_argument(
        "-c", dest="trace", action=TraceAction, nargs=3,
        metavar=("ID", "NAME", "ARGUMENTS"),
        help="append a function call; ARGUMENTS is passed verbatim or read from @file",
    )
    parser.add_argument(
        "-t", dest="trace", action=TraceAction, nargs=2,
        metavar=("ID", "CONTENT"), help="append a tool result",
    )
    return parser


def fail(message):
    raise ChatError(message)


def read_json(path, description):
    try:
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
    except OSError as error:
        fail(f"cannot read {description} {path!r}: {error.strerror}")
    except json.JSONDecodeError as error:
        fail(f"invalid JSON in {description} {path!r}: {error.msg}")


def call_arguments(text):
    if not text.startswith("@"):
        return text
    path = text[1:]
    try:
        with open(path, encoding="utf-8") as stream:
            return stream.read()
    except OSError as error:
        fail(f"cannot read tool arguments file {path!r}: {error.strerror}")


def build_messages(events):
    messages = []

    for option, values in events:
        if option in ("-s", "-u", "-a"):
            role = {"-s": "system", "-u": "user", "-a": "assistant"}[option]
            messages.append({"role": role, "content": values})
            continue

        if option == "-c":
            call_id, name, arguments = values
            if not messages or messages[-1]["role"] != "assistant":
                messages.append({
                    "role": "assistant", "content": None, "tool_calls": []
                })
            message = messages[-1]
            message.setdefault("tool_calls", []).append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": call_arguments(arguments),
                },
            })
            continue

        if option == "-t":
            call_id, content = values
            messages.append({
                "role": "tool", "tool_call_id": call_id, "content": content
            })
            continue

        fail(f"unsupported trace option: {option}")

    return messages


def build_request(args):
    body = {
        "model": args.model,
        "messages": build_messages(args.trace),
        "stream": not args.no_stream,
    }
    if args.tools:
        body["tools"] = read_json(args.tools, "tools file")
    if not args.no_stream:
        body["stream_options"] = {"include_usage": True}
    if args.think:
        body["chat_template_kwargs"] = {"enable_thinking": True}
    if args.max_tokens is not None:
        body["max_completion_tokens"] = args.max_tokens
    if args.temperature is not None:
        body["temperature"] = args.temperature
    return body


def endpoint(args):
    host = args.host
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{args.port}/v1/chat/completions"


def request_headers(args):
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = "Bearer " + args.api_key
    return headers


def curl_command(args, body):
    command = ["curl", "-sS"]
    if not args.no_stream:
        command.append("-N")
    command.append(endpoint(args))
    for name, value in request_headers(args).items():
        command.extend(("-H", f"{name}: {value}"))
    command.extend(("--data-binary", body))
    return shlex.join(command)


class TextOutput:
    def __init__(self):
        self.reasoning = False
        self.content = False
        self.last = ""

    def write(self, field, text):
        if not text:
            return
        if field == "reasoning_content" and not self.reasoning:
            sys.stdout.write("<think>\n")
            self.reasoning = True
        if field == "content" and self.reasoning and not self.content:
            sys.stdout.write("\n</think>\n\n")
        if field == "content":
            self.content = True
        sys.stdout.write(text)
        sys.stdout.flush()
        self.last = text[-1]

    def finish(self):
        if self.reasoning and not self.content:
            sys.stdout.write("\n</think>")
            self.last = ">"
        if self.reasoning or self.content:
            if self.last != "\n":
                sys.stdout.write("\n")
            sys.stdout.flush()


def print_completion(value):
    try:
        message = value["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        fail("response does not contain an assistant message")
    output = TextOutput()
    output.write("reasoning_content", message.get("reasoning_content", ""))
    output.write("content", message.get("content", ""))
    output.finish()


def stream_completion(response):
    output = TextOutput()
    data_lines = []
    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
            continue
        if line or not data_lines:
            continue

        data = "\n".join(data_lines)
        data_lines.clear()
        if data == "[DONE]":
            break
        try:
            value = json.loads(data)
        except json.JSONDecodeError as error:
            fail(f"invalid SSE JSON: {error.msg}")
        if "error" in value:
            fail(value["error"].get("message", "server returned an error"))
        for choice in value.get("choices", []):
            delta = choice.get("delta", {})
            output.write("reasoning_content", delta.get("reasoning_content", ""))
            output.write("content", delta.get("content", ""))
    output.finish()


def send(args, body):
    request = urllib.request.Request(
        endpoint(args), data=body.encode("utf-8"),
        headers=request_headers(args), method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            if args.raw:
                for chunk in response:
                    sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                return
            if args.no_stream:
                try:
                    value = json.load(response)
                except json.JSONDecodeError as error:
                    fail(f"invalid response JSON: {error.msg}")
                print_completion(value)
            else:
                stream_completion(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        fail(f"HTTP {error.code}: {detail.strip()}")
    except urllib.error.URLError as error:
        fail(f"request failed: {error.reason}")


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    try:
        request = build_request(args)
        body = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
        if args.dry_run:
            print(curl_command(args, body))
        else:
            send(args, body)
    except ChatError as error:
        print(f"chat.py: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
