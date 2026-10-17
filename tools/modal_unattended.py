"""Private CPU-only preservation observer; never launches or retries training.

Deploy with `modal deploy tools/modal_unattended.py`, then use this file's
`start` command once. The spawning client exits; the deployed function keeps
its own heartbeat, run inventory and failure/retry reports on durable storage.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import modal

ROOT = Path(__file__).resolve().parents[1]
RUN = os.environ.get("FYP_MODAL_RUN", "frw-20261018")
APP_NAME = "fyp-unattended-preservation"
RESULTS_NAME = "fyp-finetune-results"
ARCHIVE_NAME = "fyp-finetune-archive"
app = modal.App(APP_NAME, include_source=False)
results = modal.Volume.from_name(RESULTS_NAME, version=2)
archives = modal.Volume.from_name(ARCHIVE_NAME)
image = (modal.Image.debian_slim(python_version="3.11")
         .env({"PYTHONPATH": "/app/source:/app/source/tools", "PYTHONUNBUFFERED": "1"})
         .add_local_dir(ROOT / "rank_suite", "/app/source/rank_suite", copy=True,
                        ignore=["__pycache__", "*.pyc"]))
for name in ("requirements.txt", "README.md", "setup.sh", "run.sh",
             "tools/preservation.py", "tools/modal_unattended.py", "tools/modal_experiments.py",
             "tools/modal_archive.py", "tools/collect_modal_results.py", "tools/analyze_collected_runs.py",
             "tools/audit_requested_results.py", "tools/finalize_collected_results.py",
             "tools/modal_stream.py", "docs/OVERNIGHT.md", "docs/MODAL_EXECUTION.md", "docs/usage.md",
             "docs/PROTOCOL_V3.md"):
    image = image.add_local_file(ROOT / name, "/app/source/" + name, copy=True)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def text_file(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def store_source(destination, provenance):
    source = Path("/app/source")
    shutil.copytree(source, destination / "source", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    snapshot = provenance.pop("actual_main_runner_text", None)
    if snapshot:
        path = destination / "source" / "execution" / "main-runner.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(snapshot)
        provenance["actual_main_runner_sha256"] = hashlib.sha256(snapshot.encode()).hexdigest()
    scientific = hashlib.sha256()
    for path in sorted((source / "rank_suite").glob("*.py")):
        scientific.update(path.name.encode())
        scientific.update(path.read_bytes())
    provenance["scientific_source_sha256"] = scientific.hexdigest()
    save(destination / "source" / "provenance.json", provenance)
    files = []
    for path in sorted((destination / "source").rglob("*")):
        if path.is_file():
            files.append({"path": str(path.relative_to(destination)), "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    save(destination / "source-manifest.json", {"files": files, "captured_utc": utc()})
    return scientific.hexdigest()


@app.function(image=image, volumes={"/results": results,
                                   "/archives": archives.with_mount_options(read_only=True)},
              cpu=(0.125, 0.5), memory=(256, 1024), timeout=28800,
              max_containers=1, single_use_containers=True, retries=0)
def preserve(run: str, interval: int, max_seconds: int, archive_grace: int, provenance: dict):
    from tools.preservation import csv_summary, expected_grid, inventory, read_json
    base, archive_base = Path("/results") / run, Path("/archives") / run
    destination = base / "execution" / "unattended"
    destination.mkdir(parents=True, exist_ok=True)
    begin, finished_at = time.monotonic(), None
    observer_errors, outcomes, calls = [], {}, {}
    previous_journal, last_report = {}, {}
    try:
        source_hash = store_source(destination, dict(provenance))
        defaults = read_json(base / "defaults" / "lr_defaults.json")
        if defaults is None or defaults["context"]["source_sha256"] != source_hash:
            raise ValueError("Preserved scientific source does not match activated calibration")
        expected = expected_grid(defaults)
        save(destination / "expected-grid.json", expected)
        results.commit()
    except Exception:
        save(destination / "observer-exit.json", {"utc": utc(), "phase": "setup",
             "error": traceback.format_exc(), "raw_exports_retained": True})
        results.commit()
        raise
    try:
        while True:
            try:
                results.reload()
                archives.reload()
                journal = read_json(base / "execution" / "main-workers.json", previous_journal)
                previous_journal = journal
                for model, value in journal.get("returned", {}).items():
                    outcomes[model] = value
                for model, call_id in journal.get("calls", {}).items():
                    if model in outcomes:
                        continue
                    call = calls.setdefault(model, modal.FunctionCall.from_id(call_id))
                    try:
                        outcomes[model] = call.get(timeout=0)
                    except (TimeoutError, modal.exception.TimeoutError):
                        pass
                    except Exception as exc:
                        # Fatal worker exits remain identifiable even if its
                        # ordinary training failure log was never flushed.
                        outcomes[model] = {"status": "worker_failed", "error": repr(exc),
                                           "observed_utc": utc(), "call_id": call_id}
                report = inventory(base, archive_base, expected, journal, outcomes)
                done = report["training_finished"]
                if done and finished_at is None:
                    finished_at = time.monotonic()
                archived = report["archived_sequences"] == report["completed_sequences"]
                timed_out = time.monotonic() - begin >= max_seconds
                grace_elapsed = done and time.monotonic() - finished_at >= archive_grace
                terminal = done and (archived or grace_elapsed)
                status = ("complete" if report["completed_sequences"] == len(expected) and archived
                          else "finished_with_issues") if terminal else ("observer_timeout" if timed_out else "running")
                report.update({"status": status, "updated_utc": utc(), "run": run,
                               "results_volume": RESULTS_NAME, "archive_volume": ARCHIVE_NAME,
                               "scientific_source_sha256": source_hash,
                               "elapsed_observer_seconds": time.monotonic() - begin,
                               "cpu_and_final_analysis": "deferred_until_user_returns",
                               "automatic_training_retries": False, "observer_errors": observer_errors,
                               "archive_progress": read_json(base / "execution" / "archive-progress.json", {})})
                save(destination / "status.json", report)
                save(destination / "failures.json", {"updated_utc": report["updated_utc"],
                     "failure_events": report["failure_events"], "worker_errors": report["worker_errors"],
                     "worker_outcomes": outcomes, "read_issues": report["read_issues"],
                     "observer_errors": observer_errors,
                     "artifact_errors": [{"run_id": row["run_id"], "errors": row["artifact_errors"]}
                                         for row in report["runs"] if row["artifact_errors"]]})
                save(destination / "retry-plan.json", {"automatic_retry": False,
                     "training_finished": done, "updated_utc": report["updated_utc"],
                     "candidates": report["retry_candidates"],
                     "instruction": "Review failures and committed checkpoints after returning; "
                                    "rerun only missing/failed identities under the same scientific controls."})
                text_file(destination / "runs.csv", csv_summary(report["runs"]))
                text_file(destination / "README.md", f"""# Remotely preserved LoRA experiments

Status: **{status}**, updated {report['updated_utc']}.
Completed {report['completed_sequences']}/{len(expected)} sequences; lossless archives {report['archived_sequences']}.
CPU spectral analysis and final aggregate analysis are deferred until the user returns.
No automatic training retries are enabled.

Read `status.json`, `runs.csv`, `failures.json`, and `retry-plan.json` first.
Raw exports, predictions, training logs and partial recovery checkpoints remain in
`{RESULTS_NAME}/{run}/main/<model>/`. Calibration evidence and defaults remain in
`{RESULTS_NAME}/{run}/calibration/` and `defaults/`.
Independent lossless archives and their SHA-256 manifests remain in
`{ARCHIVE_NAME}/{run}/archives/<model>/`.
`source/` preserves the frozen scientific code, pinned requirements and recovery tools.
`source-manifest.json` checksums this code snapshot. Scientific source SHA-256: `{source_hash}`.

This preservation report checks identity, scope, export sizes and step-log completeness.
Full numerical, prediction, spectral and final scientific audits remain pending.
The primary and archive volumes are two copies within the same Modal account;
retain them until a verified local backup is complete. Storage charges can continue.
See `source/docs/OVERNIGHT.md` for retrieval and deferred analysis commands.
""")
                results.commit()
                last_report = report
                print(json.dumps({"updated_utc": report["updated_utc"], "status": status,
                                  "counts": report["counts"], "archived": report["archived_sequences"],
                                  "retry_candidates": len(report["retry_candidates"])}), flush=True)
                if terminal or timed_out:
                    return {key: report[key] for key in ("status", "counts", "completed_sequences",
                                                         "archived_sequences", "updated_utc")}
            except Exception as exc:
                observer_errors.append({"utc": utc(), "error": repr(exc), "traceback": traceback.format_exc()})
                save(destination / "observer-errors.json", observer_errors)
                results.commit()
                if time.monotonic() - begin >= max_seconds:
                    raise
            time.sleep(interval)
    except BaseException:
        save(destination / "observer-exit.json", {"utc": utc(), "error": traceback.format_exc(),
             "last_successful_update": last_report.get("updated_utc"),
             "raw_exports_retained": True})
        results.commit()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "status"))
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--root", default="Finetune-result")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--max-seconds", type=int, default=25200)
    parser.add_argument("--archive-grace", type=int, default=600)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in (".", ".."):
        parser.error("run must be a single directory name")
    if args.interval < 15 or not 60 <= args.max_seconds <= 28200 or args.archive_grace < 0:
        parser.error("interval >= 15, 60 <= max-seconds <= 28200, archive-grace >= 0 required")
    local = Path(args.root).resolve() / "execution"
    if args.command == "start":
        provenance = {"started_utc": utc(), "main_calls": {}, "archive_call": {}}
        for key, name in (("main_calls", "main-calls.json"), ("archive_call", "archive-call.json")):
            path = local / name
            if path.exists():
                provenance[key] = json.loads(path.read_text())
        main_hash = provenance["main_calls"].get("cloud_runner_sha256")
        if main_hash:
            path = local / "runner-snapshots" / f"main-{main_hash[:12]}.py"
            if path.exists():
                text = path.read_text()
                if hashlib.sha256(text.encode()).hexdigest() != main_hash:
                    raise ValueError("Actual main runner snapshot checksum mismatch")
                provenance["actual_main_runner_text"] = text
        existing = local / "unattended-call.json"
        if existing.exists() and json.loads(existing.read_text()).get("run") == args.run:
            raise SystemExit("Observer already recorded; inspect status before starting a duplicate")
        call = modal.Function.from_name(APP_NAME, "preserve").spawn(
            args.run, args.interval, args.max_seconds, args.archive_grace, provenance)
        payload = {"run": args.run, "app": APP_NAME, "call_id": call.object_id,
                   "started_utc": utc(), "gpu": None, "automatic_training_retries": False}
        save(existing, payload)
        print(json.dumps(payload, indent=2))
        return
    destination = local / "unattended"
    destination.mkdir(parents=True, exist_ok=True)
    remote = f"/{args.run}/execution/unattended"
    for name in ("status.json", "failures.json", "retry-plan.json", "runs.csv", "README.md"):
        data = b"".join(results.read_file(remote + "/" + name))
        path = destination / name
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
    report = json.loads((destination / "status.json").read_text())
    print(json.dumps({key: report[key] for key in ("status", "updated_utc", "counts", "models",
                     "completed_sequences", "archived_sequences", "training_finished", "budget")}, indent=2))


if __name__ == "__main__":
    main()
