#!/usr/bin/env python3
"""Run a command with a portable TERM/KILL deadline on POSIX systems."""

import os
from pathlib import Path
import signal
import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 4:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} SECONDS GRACE COMMAND...")
    seconds, grace = float(sys.argv[1]), float(sys.argv[2])
    process = subprocess.Popen(sys.argv[3:], start_new_session=True)
    try:
        return process.wait(timeout=seconds)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if isinstance(error, KeyboardInterrupt):
            return 130
        print(f"timeout after {seconds:g}s: {' '.join(sys.argv[3:])}", file=sys.stderr)
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
