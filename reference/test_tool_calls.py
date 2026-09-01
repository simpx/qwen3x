#!/usr/bin/env python3
"""Tests for the Transformers reference server's Qwen tool boundary."""

import json
import unittest

from reference.tool_calls import (
    ToolCallParseError,
    normalize_messages,
    parse_generated_tool_calls,
)


class ToolCallsTest(unittest.TestCase):
    def test_normalize_openai_arguments_without_mutating_input(self):
        messages = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path":"README.md","start_line":2}',
                },
            }],
        }]
        normalized = normalize_messages(messages)
        arguments = normalized[0]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(arguments, {"path": "README.md", "start_line": 2})
        self.assertIsInstance(
            messages[0]["tool_calls"][0]["function"]["arguments"], str
        )
        with self.assertRaises(ToolCallParseError):
            normalize_messages([{
                "role": "assistant",
                "tool_calls": [{"function": {"arguments": "[]"}}],
            }])

    def test_parse_multiple_calls_and_multiline_arguments(self):
        text = """I will inspect it.
<tool_call>
<function=read_file>
<parameter=path>
README.md
</parameter>
</function>
</tool_call>
<tool_call>
<function=bash>
<parameter=command>
printf 'hello\\nworld'
</parameter>
</function>
</tool_call>"""
        content, calls = parse_generated_tool_calls(text)
        self.assertEqual(content, "I will inspect it.")
        self.assertEqual(calls[0], {
            "name": "read_file", "arguments": {"path": "README.md"},
        })
        self.assertEqual(
            calls[1]["arguments"]["command"], "printf 'hello\\nworld'"
        )
        self.assertEqual(
            json.loads(json.dumps(calls[0]["arguments"])),
            {"path": "README.md"},
        )

    def test_plain_fallback_and_malformed_call(self):
        self.assertEqual(
            parse_generated_tool_calls("plain answer"), ("plain answer", [])
        )
        content, calls = parse_generated_tool_calls(
            "<function=bash><parameter=command>pwd</parameter></function>"
        )
        self.assertEqual(content, "")
        self.assertEqual(calls[0]["arguments"], {"command": "pwd"})
        with self.assertRaises(ToolCallParseError):
            parse_generated_tool_calls("<tool_call><function=bash>")

    def test_generated_arguments_follow_tool_schema(self):
        text = """<tool_call>
<function=read_file>
<parameter=path>
README.md
</parameter>
<parameter=start_line>
201
</parameter>
</function>
</tool_call>"""
        tools = [{
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer"},
                    },
                },
            },
        }]
        _, calls = parse_generated_tool_calls(text, tools)
        self.assertEqual(calls[0]["arguments"], {
            "path": "README.md", "start_line": 201,
        })


if __name__ == "__main__":
    unittest.main()
