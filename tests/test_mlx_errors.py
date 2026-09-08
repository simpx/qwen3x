"""Verify fatal forward exceptions and checkpoint invariant failures."""
from pathlib import Path
import resource
import signal
import subprocess
import sys


if __name__ == "__main__":
    # The child inherits this limit; intentional aborts must not create cores.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    program = str(Path(sys.argv[1]).resolve())
    for argument, marker in [("--fail-forward", "MLX forward failed: "),
                             ("--fail-restore", "MLX checkpoint restore: no checkpoint")]:
        result = subprocess.run([program, argument], capture_output=True,
                                text=True, timeout=30)
        assert result.returncode == -signal.SIGABRT, result
        assert marker in result.stderr, result.stderr
        if argument == "--fail-forward":
            assert result.stderr.split(marker, 1)[1].strip(), result.stderr
    print("mlx forward exception and checkpoint invariant: passed")
