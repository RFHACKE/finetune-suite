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
import signal
import subprocess
import sys
import time

import modal


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def start_verified_cpu_analysis(workspace, root):
    """Start local spectral work only after that model has verified raw data."""
    state_path = root / "execution/retrieval.json"
    if not state_path.exists():
        return
    state = json.loads(state_path.read_text())
    journal_path = root / "execution/cpu-workers.json"
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else {}
    environment = os.environ.copy()
    environment.update({"PYTHONPATH": str(root / "execution/source"), "OMP_NUM_THREADS": "2",
                        "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
    for model in ("llama3.2-1b", "qwen2.5-1.5b", "granite3.3-2b", "qwen2.5-3b"):
        pending = False
        for run_id in state.get("verified_runs", []):
            if run_id.startswith(model + "__"):
                result = json.loads((root / "runs" / run_id / "result.json").read_text())
                pending |= result.get("intruder_status") != "completed"
        if not pending:
            continue
        prior = journal.get(model, {})
        process_file = Path("/proc") / str(prior.get("pid", "missing")) / "cmdline"
        if process_file.exists():
            command = process_file.read_bytes().split(b"\0")
            if str(root).encode() in command and model.encode() in command:
                continue
        attempts = prior.get("attempts", 0) + 1
        if attempts > 3:
            continue
        command = [str(workspace / ".venv/bin/python"), "-u",
                   str(workspace / "tools/analyze_collected_runs.py"), "--root", str(root),
                   "--model", model, "--threads", "2", "--watch"]
        with (root / "execution" / f"cpu-{model}.log").open("a") as log:
            child = subprocess.Popen(command, cwd=workspace, env=environment,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        journal[model] = {"pid": child.pid, "started_unix": time.time(), "command": command,
                          "local_cpu_only": True, "attempts": attempts}
        save(journal_path, journal)


def collect_raw(workspace, root, launch):
    """Wait for the original download, then collect within the shared USD cap."""
    execution = root / "execution"
    original = workspace / "Finetune-result/execution"
    catalog = execution / "delivery-catalog.json"
    if not catalog.exists():
        save(execution / "raw-delivery-deferred.json", {"reason": "Checked delivery catalog missing"})
        return
    while True:
        state_path = original / "retrieval.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get("retrieval_finished"):
            break
        save(execution / "raw-delivery-waiting.json", {"updated_unix": time.time(),
             "reason": "Original cohort downloads have priority", "verified_original_runs":
                 len(state.get("verified_runs", [])), "raw_exports_preserved_remotely": True})
        time.sleep(30)
    stream_app = "fyp-large-rank-stream"
    environment = os.environ.copy()
    environment.update({"FYP_MODAL_RUN": launch["run"], "FYP_STREAM_APP": stream_app})
    with (execution / "raw-delivery.log").open("a") as log:
        subprocess.run([sys.executable, "-m", "modal", "deploy", "tools/modal_stream.py"],
                       cwd=workspace, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        try:
            url = modal.Function.from_name(stream_app, "stream_archives").get_web_url()
            command = [sys.executable, "-u", "-m", "tools.collect_delivery", "--catalog", str(catalog),
                       "--target", str(root), "--archive-url", url, "--workers", "16", "--connections", "16"]
            for attempt in range(1, 4):
                child = subprocess.Popen(command, cwd=workspace, stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
                stopped, last_billing = False, 0
                try:
                    while child.poll() is None:
                        try:
                            start_verified_cpu_analysis(workspace, root)
                        except Exception as exc:
                            save(execution / "cpu-start-error.json", {"error": repr(exc),
                                 "raw_collection_continues": True})
                        if time.monotonic() - last_billing >= 300:
                            cap = json.loads((original / "active-budget.json").read_text())["approved_total_usd"]
                            billing = subprocess.run([sys.executable, "-m", "modal", "billing", "summary", "--json"],
                                                     capture_output=True, text=True, timeout=60)
                            if billing.returncode:
                                raise RuntimeError("Cannot verify the remaining download budget; partial data retained")
                            metered = json.loads(billing.stdout)
                            save(execution / "billing-raw-delivery.json", metered)
                            if float(metered["metered_cost"]) >= float(cap) - .25:
                                stopped = True
                                save(execution / "raw-delivery-budget-stop.json", {
                                    "metered_cost": metered["metered_cost"], "approved_total_usd": cap,
                                    "partial_downloads_retained": True, "remote_raw_exports_retained": True})
                                break
                            last_billing = time.monotonic()
                        time.sleep(30)
                finally:
                    if child.poll() is None:
                        # One signal reaches both Python and native aria2 children;
                        # aria2 persists its resumption controls on SIGINT.
                        os.killpg(child.pid, signal.SIGINT)
                        try:
                            child.wait(timeout=45)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGTERM)
                            child.wait(timeout=10)
                if stopped:
                    return
                if child.returncode == 0:
                    try:
                        start_verified_cpu_analysis(workspace, root)
                    except Exception as exc:
                        save(execution / "cpu-start-error.json", {"error": repr(exc)})
                    save(execution / "raw-delivery-finished.json", {"updated_unix": time.time(),
                         "verified_runs": 40, "cpu_spectral_audit": "pending_separate_analysis"})
                    return
                save(execution / "raw-delivery-attempt.json", {"attempt": attempt,
                     "returncode": child.returncode, "partial_data_retained": True})
            save(execution / "raw-delivery-deferred.json", {
                 "reason": "Three transport attempts exhausted; resume from saved partials",
                 "remote_raw_exports_retained": True})
        finally:
            subprocess.run([sys.executable, "-m", "modal", "app", "stop", "--yes", stream_app],
                           cwd=workspace, stdout=log, stderr=subprocess.STDOUT, check=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="Finetune-result/large-ranks")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--max-seconds", type=int, default=21600)
    parser.add_argument("--fetch-metadata", action="store_true")
    parser.add_argument("--fetch-raw", action="store_true")
    parser.add_argument("--resume-download", action="store_true",
                        help="Resume the saved raw delivery with endpoint lifecycle and budget monitoring; no training")
    args = parser.parse_args()
    if args.interval < 15 or args.max_seconds < 60:
        parser.error("interval >= 15 and max-seconds >= 60 are required")
    if args.fetch_raw and not args.fetch_metadata:
        parser.error("--fetch-raw requires --fetch-metadata")
    if args.resume_download and (args.fetch_metadata or args.fetch_raw):
        parser.error("Use --resume-download by itself")
    workspace = Path(__file__).resolve().parents[1]
    root = Path(args.root).resolve()
    execution = root / "execution"
    launch = json.loads((execution / "main-calls.json").read_text())
    if args.resume_download:
        from tools.collect_delivery import active_download
        pid = active_download(root)
        if pid:
            print(f"Raw delivery is already running (PID {pid}); follow {execution / 'download-progress.log'}", flush=True)
            return
        state_path = execution / "retrieval.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get("retrieval_finished"):
            print("Raw delivery is already complete; no transfer started.", flush=True)
            return
        print(f"Resuming saved downloads. Progress log: {execution / 'download-progress.log'}", flush=True)
        collect_raw(workspace, root, launch)
        return
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
        snapshot_good = False
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
                status_path = launch.get("status_path", "execution/status.json")
                raw = b"".join(volume.read_file(f"/{launch['run']}/{status_path}"))
                status = json.loads(raw)
                snapshot_good = True
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
        if len(results) == len(calls) and snapshot_good:
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
                    elif args.fetch_raw:
                        try:
                            collect_raw(workspace, root, launch)
                        except Exception as exc:
                            save(execution / "raw-delivery-error.json", {"error": repr(exc),
                                 "remote_raw_exports_retained": True, "automatic_training_retries": False})
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
