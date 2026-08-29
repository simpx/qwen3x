#!/usr/bin/env python3
"""Differential tests for dependency-free render.cpp against Transformers."""

import os
import json
import random
import struct
import subprocess
import tempfile
import unittest
import unicodedata
from pathlib import Path

from transformers import AutoTokenizer


PROJECT = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get(
    "TOKENIZER", PROJECT / "build/models/Qwen3.5-0.8B"
))
RENDER_BIN = Path(os.environ.get(
    "RENDER_BIN", PROJECT / "build/qwen35-render.bin"
))
DRIVER = Path(os.environ.get("RENDER_TEST", PROJECT / "build/render-test"))
CHAT_TEMPLATE = (PROJECT / "reference/chat_template.jinja").read_text(
    encoding="utf-8"
)


def chat_case(name):
    options = {
        "add_generation_prompt": True,
        "enable_thinking": False,
        "preserve_thinking": True,
        "add_vision_id": False,
    }
    tools = None
    if name == "basic":
        messages = [{"role": "user", "content": "你好"}]
    elif name == "system-thinking":
        messages = [
            {"role": "system", "content": "  Be useful.  "},
            {"role": "user", "content": "Explain NFC."},
        ]
        options["enable_thinking"] = True
    elif name == "history-preserve":
        messages = [
            {"role": "user", "content": "question one"},
            {"role": "assistant", "content": "answer one",
             "reasoning_content": "thought one"},
            {"role": "user", "content": "question two"},
        ]
    elif name == "history-drop":
        messages = [
            {"role": "user", "content": "question one"},
            {"role": "assistant", "content": "answer one",
             "reasoning_content": "thought one"},
            {"role": "user", "content": "question two"},
        ]
        options["preserve_thinking"] = False
    elif name == "embedded-thinking":
        messages = [
            {"role": "user", "content": "one"},
            {"role": "assistant",
             "content": "<think>\ninside\n</think>\n\nvisible"},
            {"role": "user", "content": "two"},
        ]
    elif name == "tools":
        tools = [{
            "type": "function",
            "function": {
                "name": "weather",
                "description": "天气\n查询 <>&\u2028",
                "parameters": {
                    "type": "object", "required": ["city"], "default": 1.0,
                    "maximum": 18446744073709551615,
                },
            },
        }]
        messages = [
            {"role": "system", "content": "Keep it short."},
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "I will check.", "tool_calls": [{
                "name": "weather",
                "arguments": {
                    "city": "杭州", "days": 2,
                    "units": ["C", "km/h"], "detail": True,
                    "ratio": 1e-7, "offset": -3,
                    "maximum": 18446744073709551615,
                },
            }]},
            {"role": "tool", "content": "sunny"},
            {"role": "user", "content": "thanks"},
        ]
    elif name == "tool-group":
        messages = [
            {"role": "user", "content": "look up both"},
            {"role": "tool", "content": " first "},
            {"role": "tool", "content": "second"},
            {"role": "user", "content": "continue"},
        ]
    elif name == "vision":
        messages = [{"role": "user", "content": [
            {"text": "compare "}, {"type": "image"},
            {"text": " and "}, {"type": "video"},
        ]}]
        options["add_vision_id"] = True
    elif name == "null-content":
        messages = [{"role": "user", "content": None}]
    elif name == "no-generation-prompt":
        messages = [{"role": "user", "content": "only render history"}]
        options["add_generation_prompt"] = False
    elif name == "tool-first":
        messages = [
            {"role": "tool", "content": "orphan result"},
            {"role": "user", "content": "continue"},
        ]
    elif name == "multiple-tool-calls":
        messages = [
            {"role": "user", "content": "call both"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"name": "alpha", "arguments": {}},
                {"function": {"name": "beta", "arguments": {
                    "none": None, "false": False,
                    "nested": {"b": 2, "a": 1},
                }}},
            ]},
            {"role": "tool", "content": "done"},
            {"role": "user", "content": "thanks"},
        ]
    elif name == "vision-keys":
        messages = [{"role": "user", "content": [
            {"image_url": {"url": "unused"}},
            {"image": "unused"},
            {"video": "unused"},
        ]}]
        options["add_vision_id"] = True
    else:
        raise ValueError(name)
    return messages, tools, options


class RenderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for path in (MODEL / "tokenizer.json", RENDER_BIN, DRIVER):
            if not path.exists():
                raise unittest.SkipTest(f"missing render test input: {path}")
        # Explicit type is intentional: the checkpoint's legacy
        # tokenizer_class still says Qwen2Tokenizer and overrides its own
        # Qwen3.5 regex when loaded without this argument.
        cls.tokenizer = AutoTokenizer.from_pretrained(
            MODEL, tokenizer_type="qwen3_5"
        )
        cls.tokenizer.chat_template = CHAT_TEMPLATE

    @classmethod
    def driver(cls, mode, payload=b""):
        command = [str(DRIVER), str(RENDER_BIN), mode]
        return subprocess.run(
            command, input=payload, check=True, capture_output=True
        ).stdout

    def assert_bad_render_bin(self, data, expected):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-render.bin"
            path.write_bytes(data)
            result = subprocess.run(
                [str(DRIVER), str(path), "batch-encode"],
                input=b"61\n", capture_output=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(expected, result.stderr.decode())

    def assert_encodings(self, cases):
        payload = "".join(text.encode().hex() + "\n" for text in cases).encode()
        lines = self.driver("batch-encode", payload).decode().splitlines()
        self.assertEqual(len(lines), len(cases))
        for text, line in zip(cases, lines):
            actual = [] if not line else list(map(int, line.split()))
            expected = self.tokenizer.encode(text, add_special_tokens=False)
            self.assertEqual(actual, expected, ascii(text))

    def test_mixed_text_and_added_tokens_match(self):
        randomizer = random.Random(123)
        pools = [
            list("abcXYZ  \t\n\r!?'123"),
            list("你好世界中文测试"),
            ["é", "e\u0301", "Å", "A\u030a", "\u0301", "\u2003", "\u00a0"],
            ["👋", "🏽", "🙂", "❤️", "🇨🇳"],
            ["<|im_start|>", "<|im_end|>", "<think>", "</think>",
             "<|audio_pad|>"],
        ]
        cases = [
            "", "hello", "  a", "a  b", "  \n  ", "e\u0301",
            "<|im_start|>user\n你好<|im_end|>\n",
            "".join(
                str(self.tokenizer.added_tokens_decoder[token])
                for token in sorted(self.tokenizer.added_tokens_decoder)
            ),
        ]
        for _ in range(5000):
            cases.append("".join(
                randomizer.choice(randomizer.choice(pools))
                for _ in range(randomizer.randrange(40))
            ))
        self.assert_encodings(cases)

    def test_unicode_normalization_and_categories_match(self):
        randomizer = random.Random(9283)
        codepoints = [
            codepoint for codepoint in range(0x110000)
            if not 0xD800 <= codepoint <= 0xDFFF
        ]
        # Chunks cover every valid Unicode scalar value while keeping each
        # individual BPE call bounded. Exact token equality simultaneously
        # checks normalization, category boundaries and raw byte encoding.
        cases = [
            "".join(map(chr, codepoints[first:first + 8192]))
            for first in range(0, len(codepoints), 8192)
        ]
        randomizer.shuffle(codepoints)
        marks = [
            chr(codepoint) for codepoint in codepoints
            if unicodedata.combining(chr(codepoint))
        ][:800]
        bases = ["A", "e", "Ω", "क", "가"]
        for _ in range(5000):
            cases.append(
                randomizer.choice(bases) +
                "".join(randomizer.choices(marks, k=randomizer.randrange(1, 5)))
            )
        cases += [
            "A" * 100000,
            "\u0301" * 10000 + "a",
            (" \t\n" * 10000) + "end",
        ]
        self.assert_encodings(cases)

    def test_every_model_id_decodes_like_python(self):
        request = []
        expected = []
        for skip in (False, True):
            for token in range(248320):
                request.append(f"{int(skip)} {token}\n")
                expected.append(self.tokenizer.decode(
                    [token], skip_special_tokens=skip,
                    clean_up_tokenization_spaces=False,
                ).encode().hex())
        actual = self.driver("batch-decode", "".join(request).encode()).decode().splitlines()
        self.assertEqual(actual, expected)

    def test_fragmented_utf8_decode_matches(self):
        cases = [
            [9008, 239, 233], [9008], [239], [233],
            [248045, 846, 198, 109266, 248046],
            [248058, 248068, 248069, 248059],
            [248077, 248319],
        ]
        randomizer = random.Random(991)
        cases += [
            [randomizer.randrange(248320)
             for _ in range(randomizer.randrange(20))]
            for _ in range(5000)
        ]
        request = "".join("0 " + " ".join(map(str, case)) + "\n" for case in cases)
        actual = self.driver("batch-decode", request.encode()).decode().splitlines()
        expected = [
            self.tokenizer.decode(case, clean_up_tokenization_spaces=False).encode().hex()
            for case in cases
        ]
        self.assertEqual(actual, expected)

    def test_complete_chat_template_matches(self):
        names = (
            "basic", "system-thinking", "history-preserve", "history-drop",
            "embedded-thinking", "tools", "tool-group", "vision", "null-content",
            "no-generation-prompt", "tool-first", "multiple-tool-calls",
            "vision-keys",
        )
        for name in names:
            with self.subTest(name=name):
                messages, tools, options = chat_case(name)
                request = {"messages": messages, "tools": tools, **options}
                payload = json.dumps(request, ensure_ascii=False).encode()
                expected_text = self.tokenizer.apply_chat_template(
                    messages, tools=tools, tokenize=False, **options
                )
                actual_text = self.driver("chat", payload).decode()
                self.assertEqual(actual_text, expected_text)

                expected_ids = self.tokenizer.apply_chat_template(
                    messages, tools=tools, tokenize=True,
                    return_dict=False, **options
                )
                actual_ids = list(map(int, self.driver(
                    "chat-ids", payload
                ).decode().split()))
                self.assertEqual(actual_ids, expected_ids)

    def test_random_chat_histories_match(self):
        randomizer = random.Random(1701)
        text = [
            "hello", "  padded  ", "你好", "e\u0301", "line one\nline two",
            "symbols <>&", "emoji 👋🏽",
        ]
        requests = []
        expected = []

        for case in range(500):
            messages = []
            if randomizer.random() < 0.4:
                messages.append({
                    "role": "system",
                    "content": randomizer.choice(text),
                })

            turn_count = randomizer.randrange(1, 5)
            for turn in range(turn_count):
                if randomizer.random() < 0.25:
                    content = [
                        {"type": "text", "text": randomizer.choice(text)},
                        {"type": randomizer.choice(("image", "video"))},
                    ]
                else:
                    content = randomizer.choice(text)
                messages.append({"role": "user", "content": content})

                if turn + 1 == turn_count and randomizer.random() < 0.35:
                    continue

                assistant = {
                    "role": "assistant",
                    "content": randomizer.choice(text),
                }
                reasoning_kind = randomizer.randrange(3)
                if reasoning_kind == 1:
                    assistant["reasoning_content"] = randomizer.choice(text)
                elif reasoning_kind == 2:
                    assistant["content"] = (
                        "<think>\n" + randomizer.choice(text) +
                        "\n</think>\n\n" + randomizer.choice(text)
                    )

                if randomizer.random() < 0.3:
                    call_count = randomizer.randrange(1, 3)
                    assistant["tool_calls"] = [
                        {
                            "type": "function",
                            "function": {
                                "name": f"lookup_{index}",
                                "arguments": {
                                    "query": randomizer.choice(text),
                                    "count": index + 1,
                                    "flags": [True, None],
                                },
                            },
                        }
                        for index in range(call_count)
                    ]
                    messages.append(assistant)
                    for index in range(call_count):
                        messages.append({
                            "role": "tool",
                            "content": f" result {case}:{turn}:{index} ",
                        })
                else:
                    messages.append(assistant)

            tools = None
            if randomizer.random() < 0.4:
                tools = [{
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "lookup 说明",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                            },
                            "required": ["query"],
                        },
                    },
                }]
            options = {
                "add_generation_prompt": randomizer.choice((False, True)),
                "enable_thinking": randomizer.choice((False, True)),
                "preserve_thinking": randomizer.choice((False, True)),
                "add_vision_id": randomizer.choice((False, True)),
            }
            request = {"messages": messages, "tools": tools, **options}
            requests.append(json.dumps(request, ensure_ascii=False))
            expected.append(self.tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False, **options
            ))

        payload = ("\n".join(requests) + "\n").encode()
        lines = self.driver("batch-chat", payload).decode().splitlines()
        actual = [bytes.fromhex(line).decode() for line in lines]
        self.assertEqual(actual, expected)

    def test_openai_tool_arguments_are_normalized_at_the_boundary(self):
        arguments = {
            "z": [2, 1], "a": "杭州", "enabled": True,
        }
        messages = [
            {"role": "user", "content": "call it"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "type": "function",
                "function": {"name": "lookup", "arguments": arguments},
            }]},
            {"role": "tool", "content": "done"},
            {"role": "user", "content": "continue"},
        ]
        expected = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False, preserve_thinking=True,
            add_vision_id=False,
        )
        messages[1]["tool_calls"][0]["function"]["arguments"] = json.dumps(
            arguments, ensure_ascii=False
        )
        request = {
            "messages": messages,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "preserve_thinking": True,
            "add_vision_id": False,
        }
        actual = self.driver(
            "chat", json.dumps(request, ensure_ascii=False).encode()
        ).decode()
        self.assertEqual(actual, expected)

    def test_malformed_inputs_are_rejected(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.driver("batch-encode", b"ff\n")

        for request in (
            b"not json",
            b"[]",
            b"{}",
            b'{"messages":{}}',
            b'{"messages":[]}',
            b'{"messages":[{"role":"alien","content":"x"}]}',
            b'{"messages":[{"role":"user","content":42}]}',
            b'{"messages":[{"role":"assistant","content":"x"}]}',
            b'{"messages":[{"role":"user","content":"x"},'
            b'{"role":"system","content":"late"}]}',
            b'{"messages":[{"role":"system","content":['
            b'{"type":"image"}]},{"role":"user","content":"x"}]}',
            b'{"messages":[{"role":"user","content":['
            b'{"type":"audio"}]}]}',
            b'{"messages":[{"role":"user","content":"x",'
            b'"tool_calls":[{"name":"f","arguments":[]}]}]}',
            b'{"messages":[{"role":"user","content":"x"},'
            b'{"role":"assistant","content":"","tool_calls":['
            b'{"name":"f","arguments":"not-json"}]}]}',
        ):
            with self.subTest(request=request):
                with self.assertRaises(subprocess.CalledProcessError):
                    self.driver("chat", request)

    def test_corrupt_render_data_is_rejected(self):
        original = RENDER_BIN.read_bytes()
        self.assert_bad_render_bin(original[:100], "size does not match header")
        self.assert_bad_render_bin(original + b"x", "size does not match header")

        damaged = bytearray(original)
        damaged[0] ^= 0xff
        self.assert_bad_render_bin(damaged, "invalid render data magic")

        damaged = bytearray(original)
        struct.pack_into("<I", damaged, 12, 0)
        self.assert_bad_render_bin(damaged, "unsupported render data header")

        damaged = bytearray(original)
        struct.pack_into("<Q", damaged, 16, len(original) + 1)
        self.assert_bad_render_bin(damaged, "size does not match header")

        damaged = bytearray(original)
        struct.pack_into("<I", damaged, 44, 0xffffffff)
        self.assert_bad_render_bin(damaged, "impossible Unicode counts")


if __name__ == "__main__":
    unittest.main()
