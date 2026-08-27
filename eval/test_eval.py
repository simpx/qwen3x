import json
import tempfile
import unittest
from pathlib import Path

import compare
import run as eval_run


class EvalToolTest(unittest.TestCase):
    def test_run_directory_name_includes_sampling_seed(self):
        self.assertEqual(
            eval_run.run_directory_name(
                "20260824-120000", "mmlu_pro", False, False, 43
            ),
            "20260824-120000-mmlu_pro-non-thinking-official-seed43",
        )
        self.assertEqual(
            eval_run.run_directory_name(
                "20260824-120000", "ifeval", True, True, 44
            ),
            "20260824-120000-ifeval-thinking-smoke-seed44",
        )

    def test_expected_sample_count_matches_planned_runs(self):
        self.assertEqual(
            eval_run.expected_sample_count("mmlu_pro", False, 36), 504
        )
        self.assertEqual(
            eval_run.expected_sample_count("ceval", False, 2), 104
        )
        self.assertEqual(
            eval_run.expected_sample_count("ifeval", False, 100), 100
        )
        self.assertEqual(
            eval_run.expected_sample_count("mmlu_pro", False, None), 12032
        )
        self.assertEqual(
            eval_run.expected_sample_count("ceval", False, None), 1346
        )
        self.assertEqual(
            eval_run.expected_sample_count("ifeval", False, None), 541
        )
        self.assertEqual(
            eval_run.expected_sample_count("ceval", True, 20), 19
        )
        self.assertIsNone(
            eval_run.expected_sample_count("ceval", False, 13)
        )

    def test_mark_manifest_running_removes_stale_terminal_fields(self):
        manifest = {
            "result": "completed",
            "error": "old failure",
            "completed_at": "yesterday",
            "report_sha256": {"report": "hash"},
            "prediction_summary": {"samples": 112},
            "prediction_input_sha256": "input-hash",
            "review_summary": {"scored_samples": 112},
            "attempts": [{"result": "completed"}],
        }
        eval_run.mark_manifest_running(manifest)
        self.assertEqual(manifest, {
            "result": "running",
            "attempts": [{"result": "completed"}],
        })

    def test_sampling_parameters_follow_qwen_benchmark_recipe(self):
        self.assertEqual(eval_run.sampling_parameters(False, False), {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 1.5,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
        })
        self.assertEqual(eval_run.sampling_parameters(True, False), {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 1.5,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
        })
        self.assertEqual(eval_run.sampling_parameters(True, True)["top_k"], 0)
        self.assertEqual(eval_run.sampling_parameters(False, True)["temperature"], 0)

    def test_paired_primary_reports_binary_disagreements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "engine/reviews/model"
            reference = root / "reference/reviews/model"
            engine.mkdir(parents=True)
            reference.mkdir(parents=True)

            def write(path, values):
                rows = [
                    {
                        "index": index,
                        "sample_score": {"score": {"value": {"acc": value}}},
                    }
                    for index, value in enumerate(values)
                ]
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))

            write(engine / "subset.jsonl", [1, 1, 0, 0])
            write(reference / "subset.jsonl", [1, 0, 1, 0])
            result = compare.paired_primary(
                root / "engine", root / "reference", "ceval"
            )
            self.assertEqual(result["both_correct"], 1)
            self.assertEqual(result["engine_only_correct"], 1)
            self.assertEqual(result["reference_only_correct"], 1)
            self.assertEqual(result["both_wrong"], 1)
            self.assertEqual(result["score_delta"], 0)
            self.assertEqual(result["mcnemar_exact_p"], 1)

    def test_semantic_server_contract_ignores_execution_knobs(self):
        info = {
            "status": "ready",
            "device": "cuda",
            "dtype": "float32",
            "cache": "static",
            "max_context_tokens": 40960,
            "slots": 4,
        }
        self.assertEqual(eval_run.semantic_server_contract("transformers", info), {
            "device": "cuda",
            "dtype": "float32",
            "max_context_tokens": 40960,
        })

    def test_old_manifest_server_contract_is_merged_and_conflicts_fail(self):
        old = {"attempts": [
            {"server_info": {"device": "cuda", "dtype": "float32"}},
            {"server_info": {
                "device": "cuda", "dtype": "float32", "cache": "static",
                "max_context_tokens": 40960,
            }},
        ]}
        self.assertEqual(
            eval_run.previous_server_contract(old, "transformers", {}),
            {"device": "cuda", "dtype": "float32", "max_context_tokens": 40960},
        )
        old["attempts"].append({"server_info": {"dtype": "bfloat16"}})
        with self.assertRaises(SystemExit):
            eval_run.previous_server_contract(old, "transformers", {})

    def test_prediction_summary_and_hash_are_order_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            predictions = run / "predictions/model"
            predictions.mkdir(parents=True)
            path = predictions / "subset.jsonl"
            rows = [
                {
                    "index": 1,
                    "messages": [
                        {"role": "user", "content": "B"},
                        {"role": "assistant", "content": "two"},
                    ],
                    "metadata": {"answer": "B"},
                    "model_output": {
                        "choices": [{
                            "message": {"content": "two"},
                            "stop_reason": "length",
                        }],
                        "usage": {
                            "input_tokens": 7,
                            "input_tokens_cache_read": 3,
                            "output_tokens": 4,
                        },
                        "time": 2.5,
                    },
                },
                {
                    "index": 0,
                    "messages": [{"role": "user", "content": "A"}],
                    "metadata": {"answer": "A"},
                    "model_output": {
                        "choices": [{
                            "message": {"content": "one"},
                            "stop_reason": "stop",
                        }],
                        "usage": {"input_tokens": 5, "output_tokens": 2},
                        "time": 1.0,
                    },
                },
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            first_hash = eval_run.prediction_input_sha256(run)
            path.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)))
            self.assertEqual(eval_run.prediction_input_sha256(run), first_hash)

            summary = eval_run.summarize_predictions(run)
            self.assertEqual(summary["samples"], 2)
            self.assertEqual(summary["input_tokens"], 12)
            self.assertEqual(summary["cached_tokens"], 3)
            self.assertEqual(summary["output_tokens"], 6)
            self.assertEqual(summary["max_output_tokens"], 4)
            self.assertEqual(summary["length_limited_samples"], 1)
            self.assertEqual(summary["finish_reasons"], {"length": 1, "stop": 1})

    def test_completion_requires_every_prediction_to_be_scored(self):
        complete_predictions = {"samples": 2, "failed_samples": 0}
        complete_reviews = {
            "samples": 2,
            "scored_samples": 2,
            "unscored_samples": 0,
        }
        self.assertEqual(
            eval_run.completion_errors(
                complete_predictions, complete_reviews
            ),
            [],
        )

        errors = eval_run.completion_errors(
            {"samples": 2, "failed_samples": 1},
            {"samples": 1, "scored_samples": 0, "unscored_samples": 1},
        )
        self.assertIn("failed predictions=1", errors)
        self.assertIn(
            "prediction/review count mismatch: predictions=2 reviews=1",
            errors,
        )
        self.assertIn("unscored reviews=1", errors)
        self.assertEqual(
            eval_run.completion_errors(
                complete_predictions, complete_reviews, report_count=0
            ),
            ["no JSON report was saved"],
        )
        self.assertEqual(
            eval_run.completion_errors(
                complete_predictions,
                complete_reviews,
                expected_samples=3,
            ),
            ["unexpected prediction count: expected=3 actual=2"],
        )


if __name__ == "__main__":
    unittest.main()
