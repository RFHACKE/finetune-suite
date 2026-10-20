"""Read saved experiment evidence without loading tensors or changing training."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

from rank_suite import config
from rank_suite.checkpoints import checkpoint_inventory_errors

RANKS = (4, 8, 16, 32, 64, 128, 256)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        # An older running controller writes its journal in place.
        return default


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def expected_grid(defaults):
    rows = []
    for model in config.MODELS:
        policy = defaults["selected"][model]["lora"]
        for seed in config.DEFAULT_SEEDS:
            for rank in RANKS:
                run = config.RunSpec(model, "lora", rank, policy["lr"], seed,
                                     head_lr_multiplier=policy["head_lr_multiplier"])
                rows.append({"run_id": run.run_id, "model": model, "rank": rank, "seed": seed,
                             "lr": run.lr, "head_lr_multiplier": run.head_lr_multiplier})
    return rows


def failure_events(model_root):
    events, issues = [], []
    for path in sorted(model_root.glob("failures*.jsonl")):
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                events.append({**json.loads(line), "log": path.name, "line": line_number})
            except json.JSONDecodeError:
                issues.append(f"{path.name}:{line_number}: incomplete failure-log record")
    return events, issues


def export_errors(run_dir, result, expected, protocol):
    """Cheap preservation checks; full numerical/checksum audit remains deferred."""
    errors = []
    for key in ("run_id", "model", "rank", "seed", "lr", "head_lr_multiplier"):
        if result.get(key) != expected[key]:
            errors.append("Identity mismatch: " + key)
    if result.get("protocol_sha256") != canonical(protocol):
        errors.append("Protocol mismatch")
    if result.get("hf_revision") != config.MODELS[expected["model"]].revision:
        errors.append("Model revision mismatch")
    if result.get("track") != "lora" or result.get("mode") != "main":
        errors.append("Unexpected training track/mode")
    if result.get("epochs") != 3 or result.get("train_samples") != 2490:
        errors.append("Unexpected training scope")
    sizes = result.get("checkpoint_sizes", {})
    if not sizes:
        errors.append("Missing checkpoint size inventory")
    errors.extend(checkpoint_inventory_errors(run_dir, result))
    boundaries = result.get("boundaries", [])
    if len(boundaries) != 3:
        errors.append("Missing task boundary")
    for i, boundary in enumerate(boundaries):
        checkpoint = run_dir / boundary["path"]
        required = ["adapter_model.safetensors", "adapter_config.json", "heads.pt"]
        required += [f"predictions_{task}.npz" for task in config.TASK_KEYS[:i + 1]]
        for name in required:
            if not (checkpoint / name).is_file():
                errors.append("Missing analysis export: " + str((checkpoint / name).relative_to(run_dir)))
    log = run_dir / "train_log.jsonl"
    try:
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        if [row.get("step") for row in rows] != list(range(1, 1405)):
            errors.append("Incomplete optimizer step log")
    except (FileNotFoundError, json.JSONDecodeError):
        errors.append("Missing/incomplete optimizer step log")
    return errors


def inventory(base, archive_base, expected, journal, outcomes):
    """Classify each requested run, preserving failure and partial-checkpoint evidence."""
    rows, events, worker_errors, read_issues = [], [], {}, []
    protocol_hashes, base_projections = {}, {}
    by_model = {}
    for model in config.MODELS:
        root = base / "main" / model
        failures, issues = failure_events(root)
        events.extend({"model": model, **event} for event in failures)
        read_issues.extend({"model": model, "error": issue} for issue in issues)
        by_model[model] = failures
        error = read_json(root / "worker-error.json")
        if error:
            worker_errors[model] = error
        protocol = read_json(root / "protocol.json")
        if protocol:
            protocol_hashes[model] = canonical(protocol)
        meta = read_json(root / "cache" / "projections" / model / config.MODELS[model].revision / "base.json")
        if meta:
            base_projections[model] = meta
    for expected_row in expected:
        row = dict(expected_row)
        model, run_id = row["model"], row["run_id"]
        root = base / "main" / model
        run_dir = root / "runs" / run_id
        result = read_json(run_dir / "result.json", {})
        attempts = [event for event in by_model[model] if event.get("run_id") == run_id]
        terminal = model in outcomes
        task_dirs = sorted(p.name for p in (run_dir / "checkpoints").glob("task*") if p.is_dir())
        log = run_dir / "train_log.jsonl"
        last_step = None
        if log.is_file():
            # The last record can be incomplete during a checkpoint copy.
            for line in reversed(log.read_text().splitlines()[-3:]):
                try:
                    last_step = json.loads(line).get("step")
                    break
                except json.JSONDecodeError:
                    pass
        errors = []
        if result.get("status") == "completed":
            errors = export_errors(run_dir, result, row, read_json(root / "protocol.json", {}))
            state = ("artifact_invalid" if terminal else "export_pending") if errors else "completed"
        elif attempts:
            state = "failed"
        elif terminal:
            state = "incomplete" if log.is_file() or task_dirs or (run_dir / "resume.pt").is_file() else "not_started"
        else:
            state = "in_progress" if log.is_file() or task_dirs else "pending"
        manifest = read_json(archive_base / "archives" / model / f"{run_id}.json", {})
        archive = archive_base / "archives" / model / manifest.get("archive", "missing")
        archived = (state == "completed" and manifest.get("protocol_sha256") == result.get("protocol_sha256")
                    and bool(manifest.get("archive_sha256")) and archive.is_file()
                    and archive.stat().st_size == manifest.get("archive_bytes"))
        row.update({"state": state, "worker_finished": terminal, "archive_available": archived,
                    "remote_directory": f"{base.name}/main/{model}/runs/{run_id}",
                    "saved_task_directories": task_dirs, "last_saved_optimizer_step": last_step,
                    "resume_checkpoint_available": (run_dir / "resume.pt").is_file(),
                    "failure_attempts": len(attempts), "artifact_errors": errors,
                    "failure_reason": attempts[-1].get("error") if attempts else
                        (outcomes.get(model, {}).get("error") or outcomes.get(model, {}).get("status")
                         if state != "completed" and terminal else None),
                    "archive_sha256": manifest.get("archive_sha256") if archived else None})
        rows.append(row)
    finished = all(model in outcomes for model in config.MODELS)
    retry = [row for row in rows if row["state"] in ("failed", "artifact_invalid")
             or (row["worker_finished"] and row["state"] != "completed")]
    counts = dict(Counter(row["state"] for row in rows))
    return {"expected_sequences": len(expected), "counts": counts, "training_finished": finished,
            "completed_sequences": counts.get("completed", 0),
            "archived_sequences": sum(row["archive_available"] for row in rows),
            "models": {model: dict(Counter(row["state"] for row in rows if row["model"] == model))
                       for model in config.MODELS},
            "protocol_hashes": protocol_hashes, "base_projection_metadata": base_projections,
            "worker_outcomes": outcomes, "worker_errors": worker_errors,
            "budget": {"approved_total_usd": 35, "main_stage_cap_usd": journal.get("stage_budget_usd"),
                       "main_conservative_cost_usd": journal.get("conservative_resource_cost_usd"),
                       "main_stop_reason": journal.get("stop_reason")},
            "runs": rows, "failure_events": events, "read_issues": read_issues,
            "retry_candidates": retry,
            "validation_scope": "Identity, scope, presence, export sizes and optimizer-log steps. "
                                "Packager verifies projection SHA-256; collector verifies archives on download. "
                                "Full prediction/numerical/spectral audit is deferred."}


def csv_summary(rows):
    output = io.StringIO()
    fields = ("model", "rank", "seed", "lr", "head_lr_multiplier", "run_id", "state",
              "archive_available", "last_saved_optimizer_step", "resume_checkpoint_available",
              "failure_attempts", "failure_reason", "remote_directory")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
