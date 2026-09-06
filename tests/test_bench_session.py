import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts import bench_session as bench


class SessionBenchmarkTest(unittest.TestCase):
    def run_bench(self, output, session):
        engine = MagicMock()
        engine.create_session.return_value = session
        args = ["bench_session.py", "--library", "test.dylib", "--model", "test.bin",
                "--prompts", "128", "512", "--output", str(output)]
        snapshot = {"swap_used_bytes": 1234, "pressure_level": 1}
        with patch.object(bench.sys, "argv", args), \
             patch.object(bench, "Engine", return_value=engine), \
             patch.object(bench, "set_log_callback"), \
             patch.object(bench, "system_snapshot", return_value=snapshot), \
             patch.object(bench, "mac_memory", return_value={"peak_phys_footprint_bytes": 5678}), \
             patch.object(bench.time, "perf_counter", side_effect=range(100)), \
             contextlib.redirect_stdout(io.StringIO()):
            bench.main()
        return engine

    def test_warmup_reset_and_first_decode_are_not_counted_as_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bench.json"
            session = MagicMock()
            session.sync.return_value = 0
            engine = self.run_bench(output, session)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(session.reset.call_count, 8)
            self.assertEqual(session.eval.call_count, 8 * 129)
            self.assertEqual([len(call.args[0]) for call in session.sync.call_args_list], [128] * 4 + [512] * 4)
            self.assertEqual([call.kwargs["checkpoint_at"] for call in session.sync.call_args_list], [-1] * 8)
            for case in report["cases"]:
                self.assertEqual(len(case["runs"]), 4)
                self.assertEqual([run["warmup"] for run in case["runs"]], [True, False, False, False])
                self.assertEqual(case["median"]["decode_tps"], 128)
                self.assertEqual(case["median"]["prefill_tps"], case["prompt_tokens"])
            session.close.assert_called_once()
            engine.close.assert_called_once()
            # Evidence is never overwritten by a repeat invocation.
            previous = output.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.run_bench(output, session)
            self.assertEqual(output.read_bytes(), previous)

    def test_cache_hit_is_failure_and_partial_evidence_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed.json"
            session = MagicMock()
            session.sync.return_value = 128
            with self.assertRaisesRegex(AssertionError, "must not reuse"):
                self.run_bench(output, session)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("must not reuse", report["error"])
            session.close.assert_called_once()

    def test_load_log_uses_the_current_metal_fields(self):
        current = bench.metal_load_fields(
            "Metal ready device=Apple M5 Pro weights=10 recommended_working_set=20")
        self.assertEqual(current, {
            "metal_device": "Apple M5 Pro", "model_weight_bytes": 10,
            "metal_recommended_working_set_bytes": 20,
        })


if __name__ == "__main__":
    unittest.main()
