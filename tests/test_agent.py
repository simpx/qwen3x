#!/usr/bin/env python3
"""Tests for the minimal local coding agent."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts import agent


def arguments(directory, *extra):
    return agent.argument_parser().parse_args([
        "--cwd", directory, "--yes", *extra,
    ])


class AgentTest(unittest.TestCase):
    def test_request_is_small_and_non_streaming(self):
        args = arguments(".")
        body = agent.request_body(args, [{"role": "user", "content": "x"}])
        self.assertFalse(body["stream"])
        self.assertEqual(body["temperature"], 0)
        self.assertIn("Never use `cd`", agent.SYSTEM_PROMPT)
        self.assertEqual(
            [tool["function"]["name"] for tool in body["tools"]],
            ["read_file", "write_file", "bash"],
        )

    def test_read_write_and_path_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(directory)
            result = agent.execute_tool(
                args, "write_file", {"path": "sub/a.txt", "content": "hello"}
            )
            self.assertEqual(result, "Wrote sub/a.txt (5 bytes)")
            self.assertEqual(
                agent.execute_tool(args, "read_file", {"path": "sub/a.txt"}),
                "sub/a.txt: lines 1-1 of 1\nhello",
            )
            with self.assertRaises(agent.AgentError):
                agent.execute_tool(args, "read_file", {"path": "../outside"})

    def test_read_file_in_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "lines.txt").write_text(
                "one\ntwo\nthree\nfour\n", encoding="utf-8"
            )
            args = arguments(directory)
            self.assertEqual(
                agent.execute_tool(args, "read_file", {
                    "path": "lines.txt", "start_line": 2, "line_count": 2,
                }),
                "lines.txt: lines 2-3 of 4\ntwo\nthree\nnext_line: 4",
            )
            with self.assertRaises(agent.AgentError):
                agent.execute_tool(args, "read_file", {
                    "path": "lines.txt", "start_line": 5,
                })
            small = arguments(directory, "--max-output", "20")
            with self.assertRaises(agent.AgentError):
                agent.execute_tool(small, "read_file", {
                    "path": "lines.txt", "line_count": 4,
                })

            Path(directory, "empty.txt").write_text("", encoding="utf-8")
            self.assertEqual(
                agent.execute_tool(args, "read_file", {"path": "empty.txt"}),
                "empty.txt: empty file",
            )

    def test_bash_result_and_output_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(directory, "--max-output", "80")
            result = agent.execute_tool(
                args, "bash", {"command": "printf '%0100d' 0"}
            )
            self.assertIn("exit_code=0", result)
            self.assertIn("characters omitted", result)

    def test_tool_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "input.txt").write_text("hello", encoding="utf-8")
            args = arguments(directory)
            responses = iter([
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "input.txt"}),
                        },
                    }],
                },
                {"role": "assistant", "content": "The file says hello."},
            ])

            def complete(_args, _messages):
                return next(responses)

            messages = [{"role": "system", "content": "test"}]
            with contextlib.redirect_stderr(io.StringIO()):
                answer = agent.run_task(
                    args, messages, "read it", complete=complete
                )
            self.assertEqual(answer, "The file says hello.")
            self.assertEqual(messages[2]["tool_calls"][0]["id"], "call_1")
            self.assertEqual(messages[3], {
                "role": "tool", "tool_call_id": "call_1",
                "content": "input.txt: lines 1-1 of 1\nhello",
            })


if __name__ == "__main__":
    unittest.main()
