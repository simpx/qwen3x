import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import compile_metal


class MetalToolsTest(unittest.TestCase):
    def test_explicit_tools_and_spaces(self):
        with tempfile.TemporaryDirectory(prefix="metal tools ") as directory:
            root = Path(directory)
            (root / "metal.exe").touch()
            (root / "metallib.exe").touch()
            self.assertEqual(compile_metal.windows_tools(root),
                             ([str(root / "metal.exe")], [str(root / "metallib.exe")]))

    def test_missing_linker_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "metal.exe").touch()
            with self.assertRaisesRegex(RuntimeError, "Expected metal.exe and metallib.exe"):
                compile_metal.windows_tools(directory)

    def test_missing_tools_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "Apple Metal compiler not found"):
                compile_metal.windows_tools(directory)

    @patch.object(compile_metal.sys, "platform", "darwin")
    @patch.object(compile_metal.shutil, "which", return_value="/usr/bin/xcrun")
    def test_compiler_failure_stops_before_link(self, _which):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "kernels.metallib"
            with patch.object(compile_metal.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "metal")) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    compile_metal.compile_shaders(output)
            self.assertEqual(run.call_count, 1)
            self.assertIn("-fno-fast-math", run.call_args.args[0])

    @patch.object(compile_metal.sys, "platform", "darwin")
    @patch.object(compile_metal.shutil, "which", return_value="/usr/bin/xcrun")
    def test_missing_output_is_not_success(self, _which):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(compile_metal.subprocess, "run"):
                with self.assertRaisesRegex(RuntimeError, "produced no AIR"):
                    compile_metal.compile_shaders(Path(directory) / "kernels.metallib")

    @patch.object(compile_metal.sys, "platform", "linux")
    @patch.object(compile_metal.shutil, "which", return_value="/usr/bin/wslpath")
    @patch.object(compile_metal.Path, "is_dir", return_value=True)
    @patch.object(compile_metal, "windows_tools", return_value=(["metal.exe"], ["metallib.exe"]))
    def test_wsl_translates_each_path_without_shell(self, _tools, _is_dir, _which):
        with tempfile.TemporaryDirectory(prefix="metal test ") as directory:
            output = Path(directory) / "kernels.metallib"
            translated = {}
            def run(command, **kwargs):
                self.assertEqual(kwargs, {"check": True})
                translated[command[-1]].write_bytes(b"test artifact, not a real Metal library")
            def native(command, **kwargs):
                self.assertEqual(command[:2], ["wslpath", "-w"])
                result = "C:\\test path\\" + Path(command[2]).name
                translated[result] = Path(command[2])
                return result + "\n"
            with patch.object(compile_metal.subprocess, "run", side_effect=run) as calls:
                with patch.object(compile_metal.subprocess, "check_output", side_effect=native):
                    with contextlib.redirect_stdout(io.StringIO()):
                        compile_metal.compile_shaders(output)
            self.assertEqual(calls.call_count, 2)
            self.assertIn("C:\\test path\\kernels.metal", calls.call_args_list[0].args[0])

    @patch.object(compile_metal.sys, "platform", "darwin")
    @patch.object(compile_metal.shutil, "which", return_value="/usr/bin/xcrun")
    def test_stale_library_cannot_mask_linker_no_output(self, _which):
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
