"""Analyze verified downloaded runs on local CPU while GPU training continues.

A temporary view selects only pending runs. Its symlinks point to the actual
saved tensors and caches; the suite's unchanged spectral implementation writes
its normal outputs back into the downloaded run directories.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import tempfile
import time

from rank_suite.intruder import analyze_intruders
from rank_suite.utils import get_logger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="Finetune-result")
    parser.add_argument("--model", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    # Load SciPy's BLAS before the suite enters threadpool_limits. A library
    # first loaded inside that context otherwise escapes its thread limit.
    importlib.import_module("scipy.linalg")
    get_logger("rank_suite")
    while True:
        state_file = root / "execution" / "retrieval.json"
        state = json.loads(state_file.read_text()) if state_file.exists() else {}
        pending = []
        for run_id in state.get("verified_runs", []):
            path = root / "runs" / run_id / "result.json"
            result = json.loads(path.read_text())
            if result["model"] == args.model and result.get("intruder_status") != "completed":
                pending.append(run_id)
        if pending:
            with tempfile.TemporaryDirectory(prefix="fyp-spectra-") as temporary:
                view = Path(temporary)
                (view / "runs").mkdir()
                (view / "cache").symlink_to(root / "cache", target_is_directory=True)
                (view / "protocol.json").symlink_to(root / "protocol.json")
                for run_id in pending:
                    (view / "runs" / run_id).symlink_to(root / "runs" / run_id, target_is_directory=True)
                print(json.dumps({"model": args.model, "pending_verified_runs": len(pending)}), flush=True)
                result = analyze_intruders(view, models=[args.model], threads=args.threads)
                print(json.dumps({"model": args.model, **result}), flush=True)
        if not args.watch or (state.get("training_finished") and not pending):
            break
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    main()
