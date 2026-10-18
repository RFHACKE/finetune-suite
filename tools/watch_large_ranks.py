"""Monitor saved extension calls and retrieve diagnostic summaries, never train.

Raw tensors stay remotely preserved until a separate checksum-verified archive
collection. This watcher can run independently of an interactive Codex turn.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import modal


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="Finetune-result/large-ranks")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--max-seconds", type=int, default=21600)
    parser.add_argument("--fetch-metadata", action="store_true")
    args = parser.parse_args()
    if args.interval < 15 or args.max_seconds < 60:
        parser.error("interval >= 15 and max-seconds >= 60 are required")
    workspace = Path(__file__).resolve().parents[1]
    root = Path(args.root).resolve()
    execution = root / "execution"
    launch = json.loads((execution / "main-calls.json").read_text())
    recovery = json.loads((execution / "archive-recovery-call.json").read_text())
    calls = {"main": modal.FunctionCall.from_id(launch["call_id"]),
             "archive_recovery": modal.FunctionCall.from_id(recovery["call_id"])}
    volume = modal.Volume.from_name(launch["results_volume"], version=2)
    started, last_snapshot = time.monotonic(), 0
    results, retrieved, errors = {}, set(), []
    for path in (root / "runs").glob("*/epoch_diagnostics.jsonl"):
        if (path.parent / "result.json").exists():
            retrieved.add(path.parent.name)

    def fetch(row):
        run_id, model = row["run_id"], row["model"]
        prefix = f"/{launch['run']}/main/{model}/runs/{run_id}/"
        payloads = {name: b"".join(volume.read_file(prefix + name))
                    for name in ("result.json", "epoch_diagnostics.jsonl")}
        result = json.loads(payloads["result.json"])
        epochs = [json.loads(line) for line in payloads["epoch_diagnostics.jsonl"].decode().splitlines()]
        if result["run_id"] != run_id or result["status"] != "completed" or len(epochs) != 9:
            raise ValueError("Incomplete/mismatched live diagnostic snapshot")
        for name, raw in payloads.items():
            target = root / "runs" / run_id / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_bytes(raw)
            os.replace(temporary, target)
        return run_id

    while time.monotonic() - started < args.max_seconds:
        for key, call in calls.items():
            if key in results:
                continue
            try:
                results[key] = call.get(timeout=0)
                save(execution / f"{key}-observed-returned.json", results[key])
            except (TimeoutError, modal.exception.TimeoutError):
                pass
            except Exception as exc:
                results[key] = {"status": "call_failed", "error": repr(exc)}
                save(execution / f"{key}-observed-returned.json", results[key])
        if time.monotonic() - last_snapshot >= 120 or len(results) == len(calls):
            try:
                raw = b"".join(volume.read_file(f"/{launch['run']}/execution/status.json"))
                status = json.loads(raw)
                save(execution / "status.json", status)
                pending = [row for row in status["runs"]
                           if row["state"] == "completed" and row["run_id"] not in retrieved]
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = [(row["run_id"], pool.submit(fetch, row)) for row in pending]
                    for run_id, future in futures:
                        try:
                            retrieved.add(future.result())
                        except Exception as exc:
                            errors.append({"run_id": run_id, "error": repr(exc), "unix": time.time()})
                if pending and retrieved:
                    environment = os.environ.copy()
                    environment.update({"PYTHONPATH": str(execution / "source"),
                                        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                                        "OPENBLAS_NUM_THREADS": "2"})
                    with (execution / "live-chart-generation.log").open("a") as log:
                        subprocess.run([str(workspace / ".venv/bin/python"), "-c",
                                        "import sys; from rank_suite.overfitting import write_reports; "
                                        "print(write_reports(sys.argv[1]))", str(root)],
                                       cwd=execution / "source", env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=180, check=True)
                report = {"updated_unix": time.time(), "counts": status["counts"],
                          "completed_sequences": status["completed_sequences"],
                          "archived_sequences": status["archived_sequences"],
                          "training_finished": status["training_finished"],
                          "diagnostic_sequences_retrieved": len(retrieved), "errors": errors[-30:],
                          "automatic_training_retries": False, "raw_exports_retrieved": False}
                summary = root / "overfitting/summary.json"
                if summary.exists():
                    report["possible_overfitting_sequences"] = json.loads(summary.read_text())["possible_overfitting_sequences"]
                save(execution / "progress.json", report)
                print(json.dumps(report), flush=True)
            except Exception as exc:
                errors.append({"error": repr(exc), "unix": time.time()})
                save(execution / "watch-errors.json", errors)
            last_snapshot = time.monotonic()
        if len(results) == len(calls):
            if args.fetch_metadata:
                status = json.loads((execution / "status.json").read_text())
                if status["completed_sequences"] == 40 and status["archived_sequences"] == 40:
                    environment = os.environ.copy()
                    environment["FYP_MODAL_RUN"] = launch["run"]
                    command = [sys.executable, "-m", "modal", "run", "--detach", "tools/modal_delivery.py",
                               "--target", str(root), "--inline-metadata"]
                    with (execution / "metadata-retrieval.log").open("a") as log:
                        completed = subprocess.run(command, cwd=workspace, env=environment,
                                                   stdout=log, stderr=subprocess.STDOUT, check=False)
                    if completed.returncode:
                        save(execution / "metadata-retrieval-error.json", {"returncode": completed.returncode,
                             "raw_exports_preserved_remotely": True, "automatic_training_retries": False})
                else:
                    save(execution / "metadata-retrieval-deferred.json", {
                        "reason": "Exact 40-run cohort incomplete; review retry-plan before scientific delivery",
                        "raw_exports_preserved_remotely": True})
            save(execution / "watch-finished.json", {"updated_unix": time.time(), "calls": results,
                 "raw_exports_preserved_remotely": True, "automatic_training_retries": False})
            return
        time.sleep(args.interval)
    save(execution / "watch-timeout.json", {"updated_unix": time.time(),
         "cloud_work_continues_independently": True, "calls_observed": list(results)})


if __name__ == "__main__":
    main()
