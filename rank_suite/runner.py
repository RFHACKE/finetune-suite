"""Plan once, run independent seed blocks on one GPU per process, resume safely."""
from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path
import time
import traceback
import uuid

from . import config, utils
from .protocol import file_lock, register_hardware, register_invocation

logger = logging.getLogger("rank_suite.runner")


def build_grid(spec):
    runs = []
    def add(model, track, rank, lr, seed, **kw):
        optimizer = "adamw8bit" if spec.optimizer == "standard" and track == "full" else "adamw"
        runs.append(config.RunSpec(model, track, rank, lr, seed,
                                    optimizer=optimizer, **kw))
    for model in spec.models:
        m = config.MODELS[model]
        if spec.mode == "sanity":
            if spec.track in ("lora", "both"):
                for lr in sorted({m.default_lora_lr, config.SANITY_COMPARISON_LR}):
                    add(model, "lora", 16, lr, 42, task_keys=("rte",), mode="sanity")
            if spec.track in ("full", "both"):
                for lr in (m.default_full_lr, 2 * m.default_full_lr):
                    add(model, "full", None, lr, 42, task_keys=("rte",), mode="sanity")
            continue
        # Each seed is a complete rank + baseline block on the same shard.
        seeds = [42] if spec.mode in ("pilot", "smoke") else list(dict.fromkeys(spec.seeds))
        ranks = [16] if spec.mode in ("pilot", "smoke") else list(dict.fromkeys(spec.ranks))
        for seed in seeds:
            if spec.track in ("lora", "both"):
                for rank in ranks:
                    for lr in dict.fromkeys(spec.resolved_lora_lrs(model)):
                        add(model, "lora", rank, lr, seed,
                            mode="smoke" if spec.mode == "smoke" else "main",
                            epochs=1 if spec.mode == "smoke" else spec.epochs,
                            train_samples=64 if spec.mode == "smoke" else spec.train_samples)
            if spec.track in ("full", "both"):
                for lr in dict.fromkeys(spec.resolved_full_lrs(model)):
                    add(model, "full", None, lr, seed,
                        mode="smoke" if spec.mode == "smoke" else "main",
                        epochs=1 if spec.mode == "smoke" else spec.epochs,
                        train_samples=64 if spec.mode == "smoke" else spec.train_samples)
    # Canonical positions keep pilot/main/subset assignments identical.
    blocks = [(m, s) for m in config.DEFAULT_MODEL_KEYS for s in config.DEFAULT_SEEDS]
    blocks += sorted({(r.model, r.seed) for r in runs} - set(blocks))
    assignment = {b: i % spec.num_shards for i, b in enumerate(blocks)}
    return [r for r in runs if assignment[(r.model, r.seed)] == spec.shard_index]


def estimate_workload(grid):
    import math
    counts = {}
    for r in grid:
        key = f"{r.model}/{r.track}"
        counts[key] = counts.get(key, 0) + 1
    return {"total_runs": len(grid), "by_model_track": counts,
            "optimizer_steps": sum(math.ceil((int(0.8 * r.train_samples) if r.mode == "sanity"
                                              else r.train_samples) / config.TRAIN_BATCH)
                                    * r.epochs * len(r.task_keys) for r in grid),
            "note": "Sanity uses an 80% training split; smoke and sanity are excluded from main results."}


def prepare(spec):
    """Download pinned artifacts before training; safe to run on a CPU machine."""
    from huggingface_hub import snapshot_download
    from .data import prepare_task_data
    from .model_utils import load_tokenizer
    root = Path(spec.output_dir)
    for key in spec.models:
        m = config.MODELS[key]
        logger.info("Prefetch pinned checkpoint %s@%s", m.hf_id, m.revision)
        snapshot_download(m.hf_id, revision=m.revision,
                          allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*", "merges.txt", "vocab.*"])
        tok = load_tokenizer(m)
        prepare_task_data(tok, selection_dir=root / "data" / "main")
    return {"models": list(spec.models), "data_revision": config.DATA_REVISION}


def run_all(spec):
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from . import train
    from .data import prepare_task_data
    from .model_utils import load_tokenizer
    from .hardware import check_environment
    from .cost import cost_estimate, scheduling_limit_hours
    check_environment(require_gpu=True, optimizer=("adamw8bit" if spec.optimizer == "standard"
                                                  and spec.track in ("full", "both") else "adamw"))
    # Reduced-data modes use separate roots, never overwrite main selections or controls.
    root = Path(spec.output_dir)
    if spec.mode in ("smoke", "sanity"):
        root = root / spec.mode
    root.mkdir(parents=True, exist_ok=True)
    utils.get_logger("rank_suite", root / "logs" / f"worker-{spec.shard_index}.log")
    grid = build_grid(spec)
    if not grid:
        return {"completed": 0, "skipped": 0, "failed": 0, "grid": estimate_workload(grid)}
    runtime_spec = replace(spec, output_dir=str(root))
    env = utils.capture_environment()
    protocol_sha = register_invocation(root, runtime_spec, env)
    start = time.monotonic()
    limit = scheduling_limit_hours(spec)
    summary = {"completed": 0, "skipped": 0, "failed": 0, "locked": 0,
               "deferred": 0, "grid": estimate_workload(grid), "output": str(root)}
    by_model = {}
    for run in grid:
        by_model.setdefault(run.model, []).append(run)
    stop = False
    for key, runs in by_model.items():
        if stop:
            break
        pending = []
        for run in runs:
            path = root / "runs" / run.run_id / "result.json"
            old = utils.load_json(path, {})
            if old.get("status") == "completed" and not spec.force:
                if old.get("protocol_sha256") != protocol_sha:
                    raise ValueError("Completed result has different experiment controls")
                summary["skipped"] += 1
            else:
                pending.append(run)
        if not pending:
            continue
        hardware = register_hardware(root, key)
        mspec = config.MODELS[key]
        tok = load_tokenizer(mspec)
        tasks = tuple(dict.fromkeys(t for r in pending for t in r.task_keys))
        data = prepare_task_data(tok, task_keys=tasks, n_train=pending[0].train_samples,
                                 eval_limit=32 if spec.mode == "smoke" else None,
                                 selection_dir=root / "data" / ("calibration" if spec.mode == "sanity" else "main"),
                                 calibration=spec.mode == "sanity")
        for run in pending:
            elapsed_hours = (time.monotonic() - start) / 3600
            if limit is not None:
                estimate = cost_estimate([run], root, force=True).get("known_seconds", 0) / 3600
                # The first unmeasured run is an admitted pilot. Never truncate its tasks.
                if elapsed_hours >= limit or (estimate and elapsed_hours + 1.25 * estimate > limit):
                    stop = True
                    summary["stop_reason"] = "scheduling_limit"
                    break
            run_dir = root / "runs" / run.run_id
            with file_lock(run_dir / "run.lock", blocking=False) as acquired:
                if not acquired:
                    summary["locked"] += 1
                    continue
                # Another worker may have finished between initial scan and lock acquisition.
                old = utils.load_json(run_dir / "result.json", {})
                if old.get("status") == "completed" and not spec.force:
                    summary["skipped"] += 1
                    continue
                if spec.force:
                    (run_dir / "result.json").unlink(missing_ok=True)
                    (run_dir / "resume.pt").unlink(missing_ok=True)
                try:
                    result = train.run_single(run, run_dir, data, tok, mspec,
                                              experiment=runtime_spec,
                                              protocol_sha=protocol_sha, hardware=hardware)
                    summary["completed"] += 1
                    logger.info("Finished %s | BWT %s | A33 %.4f | %.0fs", run.run_id,
                                result["metrics"]["bwt"], result["metrics"]["final_task_acc"],
                                result["train_time_s"])
                except Exception as exc:
                    summary["failed"] += 1
                    utils.append_jsonl(root / f"failures-worker-{spec.shard_index}.jsonl", {
                        "run_id": run.run_id, "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    })
                    logger.error("Failed %s: %s", run.run_id, exc)
                    if spec.fail_fast:
                        stop = True
                        summary["stop_reason"] = "failure"
                        break
            train._cleanup()
    summary["elapsed_hours"] = (time.monotonic() - start) / 3600
    summary["estimated_rental_usd"] = (summary["elapsed_hours"] * spec.hourly_rate
                                        if spec.hourly_rate else None)
    summary["deferred"] = len(grid) - sum(summary[k] for k in ("completed", "skipped", "failed", "locked"))
    if spec.mode == "sanity":
        summary["sanity_checks"] = sanity_checks(root, spec)
    utils.save_json(root / "batches" / f"{uuid.uuid4().hex}.json", summary)
    # Plotting/SVD are explicit CPU commands; do not keep paid GPUs idle here.
    return summary


def sanity_checks(root, spec):
    from .metrics import load_results
    results = load_results(root, modes=("sanity",))
    checks = []
    for key in spec.models:
        m = config.MODELS[key]
        for track, expected in (("lora", {m.default_lora_lr, config.SANITY_COMPARISON_LR}),
                                ("full", {m.default_full_lr, 2 * m.default_full_lr})):
            if spec.track not in ("both", track):
                continue
            pair = [r for r in results if r["model"] == key and r["track"] == track]
            complete = {r["lr"] for r in pair} == expected
            healthy = complete and all(not r["flags"]["grad_spike_steps"] for r in pair)
            accs = [r["metrics"]["final_task_acc"] for r in pair]
            diff = (max(accs) - min(accs)) * 100 if complete and len(expected) > 1 else None
            checks.append({"model": key, "track": track, "unique_rates": len(expected),
                           "complete": complete, "healthy": healthy, "accuracy_diff_pp": diff,
                           "within_2pp": diff <= 2 if diff is not None else None,
                           "note": "Sensitivity diagnostic on training holdout; does not establish an LR plateau."})
    utils.save_json(Path(root) / "sanity_checks.json", checks)
    return checks
