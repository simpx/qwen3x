from array import array
import json
from pathlib import Path
import tempfile
import unittest

import backend_smoke as smoke


class BackendSmokeTest(unittest.TestCase):
    def test_compare_full_logits_not_just_argmax(self):
        self.assertEqual(smoke.compare([2.0, 1.0], [2.0, 1.0], "exact"), 0)
        with self.assertRaisesRegex(AssertionError, "token=1"):
            smoke.compare([2.0, 1.01], [2.0, 1.0], "wrong lower logit")

    def test_nonfinite_shape_and_argmax_rejected(self):
        for row in ([float("nan"), 1.0], [float("inf"), 1.0]):
            with self.assertRaisesRegex(AssertionError, "non-finite"):
                smoke.compare(row, [0, 1], "bad")
        with self.assertRaisesRegex(AssertionError, "shape"):
            smoke.compare([1], [1, 2], "bad")
        with self.assertRaisesRegex(AssertionError, "argmax"):
            smoke.compare([1, 1.0001], [1.0001, 1], "tie crossed")

    def test_checksums_checked_before_loading_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.bin"
            model.write_bytes(b"a pack")
            logits = root / "logits.f32"
            logits.write_bytes(array("f", [0] * sum(len(c["tokens"]) for c in smoke.CASES)).tobytes())
            metadata = {"format": "qwen3x-backend-smoke-v1", "cases": smoke.CASES,
                        "atol": smoke.ATOL, "model_bytes": model.stat().st_size,
                        "model_sha256": smoke.sha256(model), "vocab": 1,
                        "logits_sha256": smoke.sha256(logits)}
            (root / "vectors.json").write_text(json.dumps(metadata))
            (root / "check.json").write_text('{"result":"passed"}')
            model.write_bytes(b"b pack")
            with self.assertRaisesRegex(ValueError, "model.bin differs"):
                smoke.check(root / "nonexistent.so", model, root)
            self.assertEqual(json.loads((root / "check.json").read_text())["result"], "not_completed")
            model.write_bytes(b"a pack")
            logits.write_bytes(b"\1" * logits.stat().st_size)
            with self.assertRaisesRegex(ValueError, "checksum"):
                smoke.check(root / "nonexistent.so", model, root)
            logits.write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "truncated"):
                smoke.check(root / "nonexistent.so", model, root)


if __name__ == "__main__":
    unittest.main()
