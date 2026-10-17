"""Shared utilities: seeding, logging, JSON I/O, environment capture."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import config


# --------------------------------------------------------------------------
# Seeding and determinism
# --------------------------------------------------------------------------
def seed_all(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG that influences a run (methodology Sec. 8)."""
    random.seed(seed)
    import numpy as np
    np.random.seed(seed % (2**32))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        else:
            torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except ImportError:
        pass


def head_seed(run_seed: int, task_index: int) -> int:
    """Derived seed for a task's classification head.

    Depends on the run seed and the task position ONLY, so heads are
    identical across ranks and learning rates for a given seed
    (methodology Sec. 3.4 / proposal Step 1).
    """
    return (run_seed * 1000 + task_index + 1) % (2**31)


# --------------------------------------------------------------------------
# Formatting / parsing
# --------------------------------------------------------------------------
def fmt_lr(lr: float) -> str:
    """Compact, filename-safe learning-rate string (e.g. '1.85e-4')."""
    return f"{lr:.2e}".replace("+", "")


def parse_number_list(values) -> list:
    """Parse CLI numbers; accepts '1e-4'/'0.0002', commas or spaces."""
    if values is None:
        return []
    if isinstance(values, (int, float)):
        return [float(values)]
    out = []
    for token in values:
        for part in str(token).replace(",", " ").split():
            if part:
                out.append(float(part))
    return out


def parse_int_list(values) -> list:
    if values is None:
        return []
    if isinstance(values, int):
        return [int(values)]
    out = []
    for token in values:
        for part in str(token).replace(",", " ").split():
            if part:
                out.append(int(part))
    return out


# --------------------------------------------------------------------------
# JSON / JSONL I/O
# --------------------------------------------------------------------------
def save_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", delete=False) as f:
        tmp = Path(f.name)
        try:
            json.dump(obj, f, indent=2, sort_keys=True, default=_json_default, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def append_jsonl(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=_json_default) + "\n")


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "rank_suite", log_file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter(_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


# --------------------------------------------------------------------------
# Environment capture (reproducibility record, methodology Sec. 9.3)
# --------------------------------------------------------------------------
def capture_environment() -> dict:
    env = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "suite": {"name": config.SUITE_NAME, "version": config.SUITE_VERSION},
        "packages": {},
        "cuda": {},
    }
    for mod_name in ("torch", "transformers", "peft", "datasets",
                     "accelerate", "bitsandbytes", "numpy", "pandas",
                     "scipy", "matplotlib", "huggingface-hub", "tokenizers",
                     "safetensors", "pyarrow", "threadpoolctl"):
        try:
            env["packages"][mod_name] = version(mod_name)
        except PackageNotFoundError:
            env["packages"][mod_name] = "unavailable"
    try:
        import torch

        env["cuda"]["available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda"]["device_count"] = torch.cuda.device_count()
            env["cuda"]["devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            env["cuda"]["torch_version"] = torch.version.cuda
            env["cuda"]["capability"] = list(
                torch.cuda.get_device_capability(0)
            )
    except Exception as e:  # noqa: BLE001
        env["cuda"]["error"] = str(e)
    env["nvidia_smi"] = _nvidia_smi()
    return env


def _nvidia_smi() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        return out.stdout.strip() or out.stderr.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({e.__class__.__name__})"


def pip_freeze() -> str:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=120,
        )
        return out.stdout
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({e})"


class Timer:
    """Simple wall-clock timer for per-run durations."""

    def __init__(self):
        self.start = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start
