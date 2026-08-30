#!/usr/bin/env python3
"""Tests for the thin Chat Completions client."""

import contextlib
import io
import json
import shlex
import tempfile
import unittest

from scripts import chat


def parse(*arguments):
    return chat.argument_parser().parse_args(arguments)


class ChatTest(unittest.TestCase):
    def test_message_order(self):
        args = parse("-s", "rules", "-u", "hello", "-a", "hi",
                     "-u", "again")
        request = chat.build_request(args)
        self.assertEqual(request["messages"], [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "again"},
        ])
        self.assertTrue(request["stream"])
        self.assertEqual(request["stream_options"], {"include_usage": True})

    def test_tool_trace(self):
        args = parse(
            "-u", "weather",
            "-a", "checking",
            "-c", "call_a", "weather", '{"city":"杭州"}',
            "-c", "call_b", "weather", '{"city":"北京"}',
            "-t", "call_b", "sunny",
            "-t", "call_a", "rain",
            "-u", "thanks",
        )
        messages = chat.build_request(args)["messages"]
        self.assertEqual(len(messages), 5)
        assistant = messages[1]
        self.assertEqual(assistant["content"], "checking")
        self.assertEqual(
            [call["id"] for call in assistant["tool_calls"]],
            ["call_a", "call_b"],
        )
        self.assertEqual(
            json.loads(assistant["tool_calls"][0]["function"]["arguments"]),
            {"city": "杭州"},
        )
        self.assertEqual(messages[2]["tool_call_id"], "call_b")
        self.assertEqual(messages[3]["tool_call_id"], "call_a")

    def test_call_creates_empty_assistant(self):
        args = parse("-u", "weather", "-c", "call_1", "weather", "{}",
                     "-t", "call_1", "sunny")
        messages = chat.build_request(args)["messages"]
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertIsNone(messages[1]["content"])

    def test_unusual_tool_trace_is_not_rejected(self):
        messages = chat.build_request(parse(
            "-t", "missing", "orphan result",
            "-u", "x",
            "-c", "same", "one", "not JSON",
            "-c", "same", "two", "[]",
        ))["messages"]
        self.assertEqual(messages[0]["tool_call_id"], "missing")
        calls = messages[2]["tool_calls"]
        self.assertEqual([call["id"] for call in calls], ["same", "same"])
        self.assertEqual(calls[0]["function"]["arguments"], "not JSON")
        self.assertEqual(calls[1]["function"]["arguments"], "[]")

    def test_arguments_and_tools_files(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments_path = directory + "/arguments.json"
            tools_path = directory + "/tools.json"
            with open(arguments_path, "w", encoding="utf-8") as stream:
                stream.write('{ "count": 2 }\n')
            with open(tools_path, "w", encoding="utf-8") as stream:
                json.dump([{"type": "function", "function": {"name": "f"}}],
                          stream)
            args = parse(
                "--tools", tools_path,
                "-u", "x",
                "-c", "call_1", "f", "@" + arguments_path,
                "-t", "call_1", "done",
            )
            request = chat.build_request(args)
        self.assertEqual(request["tools"][0]["function"]["name"], "f")
        arguments = request["messages"][1]["tool_calls"][0]["function"][
            "arguments"
        ]
        self.assertEqual(arguments, '{ "count": 2 }\n')

    def test_empty_trace_and_invalid_values_are_translated(self):
        request = chat.build_request(parse(
            "--model", "", "--port", "70000", "--temperature", "3"
        ))
        self.assertEqual(request["model"], "")
        self.assertEqual(request["messages"], [])
        self.assertEqual(request["temperature"], 3)

    def test_dry_run_is_executable_shell_syntax(self):
        args = parse("-u", "it's ready", "--dry-run")
        request = chat.build_request(args)
        body = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
        command = chat.curl_command(args, body)
        words = shlex.split(command)
        self.assertEqual(words[:3], ["curl", "-sS", "-N"])
        self.assertEqual(words[words.index("--data-binary") + 1], body)

    def test_stream_output(self):
        stream = io.BytesIO(
            b'data: {"choices":[{"delta":{"reasoning_content":"why"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
            b'data: [DONE]\n\n'
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            chat.stream_completion(stream)
        self.assertEqual(output.getvalue(), "<think>\nwhy\n</think>\n\nanswer\n")


if __name__ == "__main__":
    unittest.main()
