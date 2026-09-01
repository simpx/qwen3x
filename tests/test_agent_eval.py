#!/usr/bin/env python3
"""Tests for the stable coding-agent eval fixtures and verdicts."""

from pathlib import Path
import tempfile
import unittest

from scripts import agent_eval


class AgentEvalTest(unittest.TestCase):
    def test_review_fixture_applies_known_diff(self):
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            agent_eval.prepare_fixture("review", cwd)
            snapshot = agent_eval.git_snapshot(cwd)
            self.assertEqual(snapshot["changed_files"], ["calculator.py"])
            self.assertIn("return sum(values) // len(values)", snapshot["diff"])

    def test_bugfix_fixture_starts_failing(self):
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            agent_eval.prepare_fixture("bugfix", cwd)
            result = agent_eval.command_result(
                ["python3", "-m", "unittest", "-v"], cwd
            )
            self.assertNotEqual(result["exit_code"], 0)

    def test_inspect_verdict_requires_sectioned_read_to_end(self):
        rubric = {
            "required_terms": ["run_request", "serve_once"],
            "expected_lines": {"run_request": 179, "serve_once": 230},
            "minimum_successful_reads": 2,
            "must_read_to_end": "pipeline.py",
        }
        trace = {"tools": [
            {"name": "read_file", "result":
             "pipeline.py: lines 1-200 of 250\n...\nnext_line: 201"},
            {"name": "read_file", "result":
             "pipeline.py: lines 201-250 of 250\n..."},
        ]}
        verdict = agent_eval.inspect_verdict(
            "serve_once line 230 calls run_request line 179", trace, rubric,
            {"changed_files": []},
        )
        self.assertTrue(verdict["passed"])

    def test_inspect_verdict_accepts_line_ranges_and_qualified_aliases(self):
        rubric = {
            "required_terms": ["HandlerRegistry.dispatch", "parse_packet"],
            "expected_lines": {
                "HandlerRegistry.dispatch": 159,
                "parse_packet": 61,
            },
            "aliases": {
                "HandlerRegistry.dispatch": ["registry.dispatch"],
            },
            "minimum_successful_reads": 1,
        }
        trace = {"tools": [{
            "name": "read_file",
            "result": "pipeline.py: lines 1-257 of 257\n...",
        }]}
        verdict = agent_eval.inspect_verdict(
            "parse_packet (lines 61-76) calls registry.dispatch "
            "(lines 159-163)",
            trace,
            rubric,
            {"changed_files": []},
        )
        self.assertTrue(verdict["passed"])


if __name__ == "__main__":
    unittest.main()
