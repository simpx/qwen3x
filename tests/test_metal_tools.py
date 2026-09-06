import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import compile_metal


class MetalToolsTest(unittest.TestCase):
    def test_commands_pin_xcode_metal4_and_macos_26_3(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "kernels.metallib"

            def run(command, **kwargs):
                self.assertEqual(kwargs, {"check": True})
                Path(command[-1]).write_bytes(b"fresh artifact")

            with patch.object(compile_metal.subprocess, "run", side_effect=run) as calls:
                with contextlib.redirect_stdout(io.StringIO()):
                    compile_metal.compile_shaders(output)
            self.assertEqual(calls.call_count, 2)
            compile_command = calls.call_args_list[0].args[0]
            link_command = calls.call_args_list[1].args[0]
            self.assertEqual(compile_command[:6], compile_metal.COMPILER)
            self.assertIn("air64-apple-macosx26.3", compile_command)
            self.assertIn("-std=metal4.0", compile_command)
            self.assertIn("-fno-fast-math", compile_command)
            self.assertEqual(link_command[:6], compile_metal.LINKER)
            self.assertEqual(output.read_bytes(), b"fresh artifact")

    def test_compiler_failure_stops_before_link(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "kernels.metallib"
            with patch.object(
                compile_metal.subprocess, "run",
                side_effect=subprocess.CalledProcessError(1, "metal"),
            ) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    compile_metal.compile_shaders(output)
            self.assertEqual(run.call_count, 1)

    def test_missing_output_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(compile_metal.subprocess, "run"):
                with self.assertRaisesRegex(RuntimeError, "produced no AIR"):
                    compile_metal.compile_shaders(
                        Path(directory) / "kernels.metallib")

    def test_stale_library_cannot_mask_linker_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "kernels.metallib"
            output.write_bytes(b"previous good library")

            def run(command, **_kwargs):
                if "-c" in command:
                    Path(command[-1]).write_bytes(b"fresh AIR")

            with patch.object(compile_metal.subprocess, "run", side_effect=run):
                with self.assertRaisesRegex(RuntimeError, "linker produced no library"):
                    compile_metal.compile_shaders(output)
            self.assertEqual(output.read_bytes(), b"previous good library")


if __name__ == "__main__":
    unittest.main()
