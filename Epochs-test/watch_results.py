"""Wait for the existing cloud coordinator, then collect its small results.

This script never launches GPUs or retries training. It can run under systemd
without Codex or a terminal. Remote data remains available if collection fails.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def log(message):
    value = f"{datetime.now(timezone.utc).isoformat()}  {message}"
    print(value, flush=True)
    with (HERE / "execution" / "collection-progress.log").open("a") as stream:
        stream.write(value + "\n")


def watch(poll_seconds=60, timeout_seconds=10800):
    (HERE / "execution").mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    receipt = HERE / "execution" / "launch.json"
    completed = False
    while time.monotonic() - start < timeout_seconds:
        if not receipt.is_file():
            log("Waiting for launch receipt; no training is started by this collector")
        else:
            identity = json.loads(receipt.read_text())
            try:
                result = modal.FunctionCall.from_id(identity["call_id"]).get(timeout=5)
                (HERE / "execution" / "coordinator-returned.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                log(f"Coordinator returned status={result.get('status')}; collecting committed results")
                completed = True
                break
            except (TimeoutError, modal.exception.TimeoutError):
                log("Training/report coordinator is still running")
            except Exception as exc:
                log(
                    f"Coordinator status unavailable ({type(exc).__name__}); will retry without launching training"
                )
        time.sleep(poll_seconds)
    if not completed:
        log("Collector wait deadline reached; collecting available committed evidence")
    for attempt in range(1, 4):
        succeeded = False
        try:
            with (HERE / "execution" / "collection-cli.log").open("a") as stream:
                outcome = subprocess.run(
                    ["modal", "run", "Epochs-test/modal_epoch_study.py", "--stage", "collect"],
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=1200,
                    check=False,
                )
            succeeded = outcome.returncode == 0
        except subprocess.TimeoutExpired:
            log(f"CPU-only collection attempt {attempt}/3 exceeded its timeout")
        if succeeded:
            log("Collection complete; remote evidence retained and local documents saved")
            return
        log(f"CPU-only collection attempt {attempt}/3 failed; remote evidence is preserved")
        if attempt < 3:
            time.sleep(30)
    raise RuntimeError("Collection failed after three CPU-only attempts; inspect collection-cli.log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    args = parser.parse_args()
    if args.poll_seconds < 10 or args.timeout_seconds < 60:
        parser.error("Use poll-seconds >=10 and timeout-seconds >=60")
    watch(args.poll_seconds, args.timeout_seconds)
