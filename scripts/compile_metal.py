#!/usr/bin/env python3
"""Compile native MSL on macOS or with Apple's Windows tools from Windows/WSL.

This only compiles shaders. It cannot run Metal or build the macOS executable.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def windows_tools(directory=None):
    compiler = shutil.which("metal.exe") if directory is None else None
    if compiler:
        root = Path(compiler).parent
    else:
        roots = [Path(directory)] if directory else sorted(
            Path(os.environ.get("ProgramFiles", "/mnt/c/Program Files")).glob("*Metal*")
        )
        matches = [p for root in roots for p in sorted(root.rglob("metal.exe"))]
        if not matches:
            raise RuntimeError(
                "Apple Metal compiler not found. Install Metal Developer Tools for Windows "
                "from https://developer.apple.com/metal/tools/ (Apple login required), "
                "or pass --tools /path/to/its/bin. WSL cannot execute Metal GPU tests."
            )
        root = matches[-1].parent
    compiler, linker = root / "metal.exe", root / "metallib.exe"
    if not compiler.is_file() or not linker.is_file():
        raise RuntimeError(f"Expected metal.exe and metallib.exe in {root}")
    return [str(compiler)], [str(linker)]


def compile_shaders(output, tools=None):
    output = Path(output).resolve()
    source = ROOT / "arch/metal/kernels.metal"
    wsl = False
    if sys.platform == "darwin":
        if not shutil.which("xcrun"):
            raise RuntimeError("Install Xcode including the Metal toolchain first")
        compiler, linker = ["xcrun", "-sdk", "macosx", "metal"], ["xcrun", "-sdk", "macosx", "metallib"]
    elif os.name == "nt" or (shutil.which("wslpath") and Path("/mnt/c/Windows").is_dir()):
        compiler, linker = windows_tools(tools)
        wsl = os.name != "nt"
    else:
        raise RuntimeError("Metal compilation needs macOS/Xcode or Windows/WSL with Apple's Metal tools")

    def native(path):
        if wsl:
            return subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()
        return str(path)

    output.parent.mkdir(parents=True, exist_ok=True)
    # An old output must never turn a broken/no-output compiler into success.
    # Keep the previous good library until both stages produce fresh artifacts.
    with tempfile.TemporaryDirectory(prefix="metal-", dir=output.parent) as directory:
        air = Path(directory) / "kernels.air"
        library = Path(directory) / "kernels.metallib"
        subprocess.run(compiler + ["-target", "air64-apple-macosx13.3", "-std=macos-metal2.4",
                                  "-fno-fast-math", "-c", native(source), "-o", native(air)], check=True)
        if not air.is_file() or air.stat().st_size == 0:
            raise RuntimeError("Metal compiler produced no AIR")
        subprocess.run(linker + [native(air), "-o", native(library)], check=True)
        if not library.is_file() or library.stat().st_size == 0:
            raise RuntimeError("Metal linker produced no library")
        library.replace(output)
    print(f"Metal shaders compiled: {output} (GPU execution not tested)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/metal/kernels.metallib")
    parser.add_argument("--tools", type=Path, help="Windows Metal tools directory; use a Linux path in WSL")
    args = parser.parse_args()
    try:
        compile_shaders(args.output, args.tools)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"metal-shaders: {error}\n")


if __name__ == "__main__":
    main()
