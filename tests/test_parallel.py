import os
from pathlib import Path
import shutil
import subprocess

import pytest


def launcher(tmp_path, worker):
    source = Path(__file__).resolve().parents[1] / "run_parallel.sh"
    shutil.copyfile(source, tmp_path / "run_parallel.sh")
    run = tmp_path / "run.sh"
    run.write_text("#!/usr/bin/env bash\n" + worker + "\n")
    run.chmod(0o755)
    return subprocess.run(["bash", str(tmp_path / "run_parallel.sh"), "0,1"],
                          env={**os.environ, "WORKER_LOG_DIR": str(tmp_path / "logs")},
                          capture_output=True, text=True, timeout=8)


@pytest.mark.parametrize("second,expected", [(0, 0), (3, 3), (7, 7)])
def test_parallel_preserves_status_of_instantly_completed_workers(tmp_path, second, expected):
    result = launcher(tmp_path, f'if [ "$CUDA_VISIBLE_DEVICES" = 1 ]; then exit {second}; fi\nexit 0')
    assert result.returncode == expected, result.stdout + result.stderr


def test_parallel_failure_stops_a_running_peer(tmp_path):
    result = launcher(tmp_path, 'if [ "$CUDA_VISIBLE_DEVICES" = 0 ]; then exec sleep 30; fi\nexit 7')
    assert result.returncode == 7, result.stdout + result.stderr
