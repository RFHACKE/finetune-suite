"""Budgeted, unattended 512/1024 extension with epoch diagnostics.

Explicitly launch once with Modal's --detach flag. All raw scientific exports,
diagnostic predictions, failure reports and charts persist remotely even if
the local client disconnects. No automatic GPU retries or adaptive tuning.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

import modal

from tools.modal_experiments import (ROOT, cache, copy_delta, durable_training,
                                     hf_secret, image as base_image, outputs, save)

RUN = "frw-20261018-r512-1024"
ORIGIN = "frw-20261018"
LOCAL = ROOT / "Finetune-result" / "large-ranks" / "execution"
RANKS = [512, 1024]
MODELS = ["llama3.2-1b", "qwen2.5-1.5b", "granite3.3-2b", "qwen2.5-3b"]
RATE = 3600 * (.000542 + 2 * .0000131 + 16 * .00000222)
app = modal.App("fyp-large-rank-extension", include_source=False)
vault = modal.Volume.from_name("fyp-finetune-archive")
image = base_image.apt_install("zstd")
for name in ("tools/modal_large_ranks.py", "tools/modal_archive.py", "tools/preservation.py",
             "docs/LARGE_RANK_EXTENSION.md", "docs/calibrated-lora-defaults.json"):
    image = image.add_local_file(ROOT / name, "/app/" + name, copy=True)


def specification(path, model=None, **kwargs):
    from rank_suite.config import ExperimentSpec
    return ExperimentSpec(models=[model] if model else MODELS, track="lora", ranks=RANKS,
                          output_dir=str(path), micro_batch=16, accum=1, eval_batch=32,
                          allow_overcomplete_ranks=True, epoch_diagnostics=True,
                          diagnostic_samples=256, calibrated_defaults={},
                          fail_fast=False, **kwargs)


@app.function(image=image, secrets=[hf_secret()], volumes={"/cache": cache, "/results": outputs},
              gpu="L40S", cpu=(2, 2), memory=(8192, 16384), timeout=14400,
              max_containers=4, single_use_containers=True, retries=0)
def train_model(model, seconds):
    import torch
    from rank_suite.runner import run_all
    from rank_suite.protocol import source_digest
    started = time.time()
    torch.set_num_threads(4)
    local = Path("/work") / RUN / model
    remote = Path("/results") / RUN / "main" / model
    if remote.exists():
        copy_delta(remote, local)
    local.mkdir(parents=True, exist_ok=True)
    flush = durable_training(local, remote)
    try:
        spec = specification(local, model, max_hours=max(60, seconds - 90) / 3600,
                             hourly_rate=RATE)
        result = run_all(spec)
        result.update({"worker_elapsed_s": time.time() - started,
                       "scientific_source_sha256": source_digest()})
        save(local / "worker-summary.json", result)
        return result
    except Exception as exc:
        save(local / "worker-error.json", {"error": repr(exc), "traceback": traceback.format_exc(),
                                          "elapsed_s": time.time() - started})
        raise
    finally:
        flush()


@app.function(image=image, volumes={"/results": outputs, "/vault": vault},
              cpu=(0.125, 2), memory=(256, 6144), timeout=21600,
              single_use_containers=True, retries=0)
def archive_and_plot(recovery_note=None):
    """CPU-only durable archive and chart generation, with a bounded lifetime."""
    from tools.modal_archive import package_run
    from tools.preservation import read_json, inventory
    from rank_suite.overfitting import write_reports
    root, archive_root = Path("/results") / RUN, Path("/vault") / RUN
    packaged, based, errors = set(), set(), []
    begin = time.monotonic()
    if recovery_note:
        save(root / "execution" / "archive-recovery.json", recovery_note)
        source = Path(__file__)
        shutil.copy2(source, root / "execution" / f"archive-runner-{hashlib.sha256(source.read_bytes()).hexdigest()[:12]}.py")
        outputs.commit()
    journal = {}
    for path in (archive_root / "archives").glob("*/*.json"):
        metadata = json.loads(path.read_text())
        packaged.add(path.stem)
        if metadata.get("includes_base_projections"):
            based.add(metadata["model"])
    while time.monotonic() - begin < 21000:
        outputs.reload()
        pending = []
        for model in MODELS:
            model_root = root / "main" / model
            for result_path in sorted((model_root / "runs").glob("*/result.json")):
                run_dir = result_path.parent
                if run_dir.name in packaged:
                    continue
                try:
                    if package_run(model_root, run_dir, archive_root / "archives" / model, model not in based):
                        packaged.add(run_dir.name)
                        based.add(model)
                    else:
                        pending.append(run_dir.name)
                except Exception as exc:
                    errors.append({"run_id": run_dir.name, "error": repr(exc), "unix": time.time()})
                    pending.append(run_dir.name)
        journal_path = root / "execution" / "main-workers.json"
        # A volume reader can transiently see an incomplete metadata update,
        # even when the producing process uses a local atomic rename.
        journal = read_json(journal_path, journal)
        progress = {"packaged_runs": len(packaged), "pending": pending, "errors": errors,
                    "updated_unix": time.time(), "gpu": None}
        save(root / "execution" / "archive-progress.json", progress)
        outputs.commit()
        if len(journal.get("returned", {})) == len(MODELS) and not pending:
            report = write_reports(root)
            save(root / "execution" / "overfitting-report.json", report)
            if recovery_note:
                expected = read_json(root / "execution" / "expected-grid.json", [])
                final = inventory(root, archive_root, expected, journal, journal["returned"])
                final["budget"] = {"approved_additional_usd": 15,
                                   "gpu_stage_cap_usd": journal["stage_budget_usd"],
                                   "conservative_gpu_stage_cost_usd": journal["conservative_resource_cost_usd"]}
                final.update({"updated_unix": time.time(), "archive_result": progress,
                              "archive_recovery": recovery_note, "overfitting_report": report})
                save(root / "execution" / "status.json", final)
                save(root / "execution" / "archive-recovery-returned.json", {**progress, "overfitting_report": report})
            outputs.commit()
            return {**progress, "overfitting_report": report, "status": "complete"}
        time.sleep(30)
    progress["status"] = "archive_timeout_raw_exports_preserved"
    save(root / "execution" / "archive-progress.json", progress)
    outputs.commit()
    return progress


def store_source(base):
    from rank_suite import config
    from rank_suite.protocol import source_digest
    from rank_suite.runner import build_grid
    destination = base / "execution" / "source"
    for name in ("rank_suite", "requirements.txt", "tools/modal_large_ranks.py",
                 "tools/modal_experiments.py", "tools/modal_archive.py", "tools/preservation.py",
                 "docs/LARGE_RANK_EXTENSION.md", "docs/calibrated-lora-defaults.json"):
        source = Path("/app") / name
        target = destination / name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    calibration_source = Path("/results") / ORIGIN / "defaults"
    if not calibration_source.is_dir():
        raise ValueError("Original completed calibration evidence is missing")
    copy_delta(calibration_source, base / "execution" / "calibration-origin")
    original = json.loads((calibration_source / "lr_defaults.json").read_text())
    for model in MODELS:
        selected = original["selected"][model]["lora"]
        if (selected["lr"] != config.DEFAULT_LORA_LR[model]
                or selected["head_lr_multiplier"] != config.DEFAULT_LORA_HEAD_MULTIPLIER[model]):
            raise ValueError("Promoted defaults do not match the selected original calibration")
    expected = [{"run_id": run.run_id, "model": run.model, "rank": run.rank, "seed": run.seed,
                 "lr": run.lr, "head_lr_multiplier": run.head_lr_multiplier}
                for run in build_grid(specification(base))]
    assert len(expected) == 40
    save(base / "execution" / "expected-grid.json", expected)
    files = [{"path": str(path.relative_to(destination)), "bytes": path.stat().st_size,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in sorted(destination.rglob("*")) if path.is_file()]
    save(base / "execution" / "source-manifest.json", {"source_sha256": source_digest(), "files": files})
    save(base / "execution" / "protocol-amendment.json", {
        "origin_run": ORIGIN, "ranks": RANKS, "seeds": list(config.DEFAULT_SEEDS),
        "sequences": 40, "approved_additional_budget_usd": 15,
        "rank_semantics": "factor_width_with_projection_dimension_cap_on_effective_update_rank",
        "epoch_probes": "256_fixed_stratified_train_and_validation_examples_per_task",
        "checkpoint_selection": False, "training_changes": False,
        "lr_policy": "transfer_completed_calibration_without_retesting_or_validation_selection",
        "original_calibration_sha256": original["sha256"], "source_sha256": source_digest(),
        "automatic_training_retries": False})
    outputs.commit()
    return expected


@app.function(image=image, volumes={"/results": outputs, "/vault": vault.with_mount_options(read_only=True)},
              cpu=(0.125, 0.5), memory=(256, 2048), timeout=21600,
              single_use_containers=True, retries=0)
def coordinate(seconds, stage_budget):
    from tools.preservation import inventory, csv_summary
    base = Path("/results") / RUN
    destination = base / "execution"
    if (destination / "main-workers.json").exists():
        raise ValueError("Cohort already launched; inspect status rather than starting duplicate GPUs")
    expected = store_source(base)
    calls, starts = {}, {}
    for model in MODELS:
        starts[model] = time.monotonic()
        calls[model] = train_model.with_options(timeout=seconds).spawn(model, seconds)
    archive_call = archive_and_plot.spawn()
    journal = {"stage": "main", "run": RUN, "gpu": "L40S", "ranks": RANKS,
               "seconds_per_worker": seconds, "created_unix": time.time(),
               "stage_budget_usd": stage_budget, "approved_additional_budget_usd": 15,
               "calls": {model: call.object_id for model, call in calls.items()},
               "archive_call_id": archive_call.object_id}
    save(destination / "main-workers.json", journal)
    outputs.commit()
    returned, finishes, last_inventory = {}, {}, 0
    while len(returned) < len(calls):
        for model, call in calls.items():
            if model in returned:
                continue
            try:
                returned[model] = call.get(timeout=0)
                finishes[model] = time.monotonic()
            except (TimeoutError, modal.exception.TimeoutError):
                pass
            except Exception as exc:
                returned[model] = {"status": "worker_failed", "error": repr(exc)}
                finishes[model] = time.monotonic()
        now = time.monotonic()
        ceiling = sum(finishes.get(model, now) - starts[model] for model in calls) * RATE / 3600 + .05
        if ceiling + .10 >= stage_budget and len(returned) < len(calls):
            journal["stop_reason"] = "aggregate_stage_budget"
            for model, call in calls.items():
                if model not in returned:
                    call.cancel(terminate_containers=True)
                    returned[model] = {"status": "budget_deferred", "checkpoints_preserved": True}
                    finishes[model] = time.monotonic()
        journal.update({"returned": returned, "conservative_resource_cost_usd": ceiling,
                        "updated_unix": time.time()})
        outputs.reload()
        save(destination / "main-workers.json", journal)
        if now - last_inventory >= 30 or len(returned) == len(calls):
            vault.reload()
            report = inventory(base, Path("/vault") / RUN, expected, journal, returned)
            report["budget"] = {"approved_additional_usd": 15, "gpu_stage_cap_usd": stage_budget,
                                "conservative_gpu_stage_cost_usd": ceiling}
            report["updated_unix"] = time.time()
            save(destination / "status.json", report)
            save(destination / "failures.json", {key: report[key] for key in
                 ("failure_events", "worker_errors", "read_issues", "retry_candidates")})
            save(destination / "retry-plan.json", {"automatic_retry": False,
                 "training_finished": report["training_finished"], "candidates": report["retry_candidates"]})
            (destination / "runs.csv").write_text(csv_summary(report["runs"]))
            last_inventory = now
        outputs.commit()
        if len(returned) < len(calls):
            time.sleep(5)
    # Keep the durable parent alive until the CPU child preserves all exports
    # and writes the charts, even if the launching laptop disconnects.
    try:
        archived = archive_call.get(timeout=21000)
    except Exception as exc:
        archived = {"status": "archive_failed", "error": repr(exc), "raw_exports_preserved": True}
    outputs.reload()
    vault.reload()
    report = inventory(base, Path("/vault") / RUN, expected, journal, returned)
    report["budget"] = {"approved_additional_usd": 15, "gpu_stage_cap_usd": stage_budget,
                        "conservative_gpu_stage_cost_usd": journal["conservative_resource_cost_usd"]}
    report.update({"updated_unix": time.time(), "archive_result": archived})
    save(destination / "status.json", report)
    outputs.commit()
    return {"completed": report["completed_sequences"], "archived": report["archived_sequences"],
            "counts": report["counts"], "budget": report["budget"], "archive_result": archived}


@app.local_entrypoint()
def main(seconds: int = 14400, stage_budget: float = 13.5, archive_only: bool = False):
    if not 120 <= seconds <= 18000 or not 0 < stage_budget <= 13.5:
        raise ValueError("Use 120–18000 seconds and a positive GPU stage cap no greater than $13.50")
    LOCAL.mkdir(parents=True, exist_ok=True)
    if archive_only:
        path = LOCAL / "archive-recovery-call.json"
        if path.exists():
            raise ValueError("CPU recovery already recorded; inspect it before starting duplicate work")
        original = json.loads((LOCAL / "main-workers.json").read_text())
        note = {"previous_archive_call_id": original["archive_call_id"],
                "reason": "Transient incomplete status JSON stopped initial CPU archiver; GPU work unaffected",
                "started_unix": time.time(), "gpu": None,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        call = archive_and_plot.spawn(recovery_note=note)
        save(path, {**note, "run": RUN, "call_id": call.object_id, "app_id": app.app_id})
        print(json.dumps({"cpu_archive_recovery_call": call.object_id, "app_id": app.app_id,
                          "gpu_experiments_started": False}), flush=True)
        result = call.get()
        save(LOCAL / "archive-recovery-returned.json", result)
        print(json.dumps(result), flush=True)
        return
    path = LOCAL / "main-calls.json"
    if path.exists():
        raise ValueError("Launch already recorded; inspect the saved call before starting duplicate work")
    call = coordinate.spawn(seconds, stage_budget)
    journal = {"run": RUN, "origin": ORIGIN, "app_id": app.app_id, "call_id": call.object_id,
               "results_volume": "fyp-finetune-results", "archive_volume": "fyp-finetune-archive",
               "ranks": RANKS, "sequences": 40, "stage_budget_usd": stage_budget,
               "approved_additional_usd": 15, "started_unix": time.time(),
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(path, journal)
    print(json.dumps(journal, indent=2), flush=True)
    result = call.get()
    save(LOCAL / "main-returned.json", result)
    print(json.dumps(result, indent=2), flush=True)
