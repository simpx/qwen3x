#!/usr/bin/env python3
"""Compile the project's Metal 4 shaders with Xcode 26 on macOS."""

import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
COMPILER = ["xcrun", "--toolchain", "Metal", "-sdk", "macosx", "metal"]
LINKER = ["xcrun", "--toolchain", "Metal", "-sdk", "macosx", "metallib"]


def compile_shaders(output):
    output = Path(output).resolve()
    source = ROOT / "arch/metal/kernels.metal"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Keep the previous library until both commands produce fresh artifacts.
    with tempfile.TemporaryDirectory(prefix="metal-", dir=output.parent) as directory:
        air = Path(directory) / "kernels.air"
        library = Path(directory) / "kernels.metallib"
        subprocess.run(COMPILER + [
            "-target", "air64-apple-macosx26.3", "-std=metal4.0",
            "-fno-fast-math", "-c", str(source), "-o", str(air),
        ], check=True)
        if not air.is_file() or air.stat().st_size == 0:
            raise RuntimeError("Metal compiler produced no AIR")
        subprocess.run(LINKER + [str(air), "-o", str(library)], check=True)
        if not library.is_file() or library.stat().st_size == 0:
            raise RuntimeError("Metal linker produced no library")
        library.replace(output)
    print(f"Metal shaders compiled: {output} (GPU execution not tested)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "build/metal/kernels.metallib")
    args = parser.parse_args()
    try:
        compile_shaders(args.output)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"metal-shaders: {error}\n")


if __name__ == "__main__":
    main()
