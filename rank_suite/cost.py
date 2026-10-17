"""Measured estimates and whole-run admission limits; never reduce experimental data."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import statistics

from . import utils


def scheduling_limit_hours(spec):
    limits = [v for v in [spec.max_hours] if v is not None]
    if spec.budget is not None:
        limits.append((spec.budget - spec.reserve) / spec.hourly_rate)
    return min(limits) if limits else None


def cost_estimate(grid, root, hourly_rate=None, force=False):
    root = Path(root)
    timings = defaultdict(list)
    for path in root.glob("runs/*/result.json"):
        r = utils.load_json(path, {})
        if (r.get("status") == "completed" and r.get("mode") == "main"
                and r.get("train_time_s", 0) > 0):
            timings[(r["model"], r["track"], r.get("optimizer"))].append(r["train_time_s"])
    known_seconds = 0.0
    unknown = defaultdict(int)
    complete = remaining = 0
    for run in grid:
        r = utils.load_json(root / "runs" / run.run_id / "result.json", {})
        if r.get("status") == "completed" and not force:
            complete += 1
            continue
        remaining += 1
        values = timings.get((run.model, run.track, run.optimizer), [])
        if values and run.mode == "main":
            known_seconds += statistics.median(values)
        else:
            unknown[f"{run.model}/{run.track}"] += 1
    hours = known_seconds / 3600
    return {
        "completed_runs": complete, "remaining_runs": remaining,
        "unmeasured_runs": sum(unknown.values()), "unmeasured_groups": dict(unknown),
        "known_seconds": known_seconds, "measured_estimate_hours": hours,
        "planning_range_hours": [1.25 * hours, 2.0 * hours] if not unknown else None,
        "planning_range_usd": ([1.25 * hours * hourly_rate, 2.0 * hours * hourly_rate]
                               if hourly_rate and not unknown else None),
        "timing_samples": {f"{m}/{t}/{o}": len(v) for (m, t, o), v in timings.items()},
        "note": "Pilot timings include load, full evaluation and checkpoint I/O. The 1.25–2× "
                "range is a planning allowance, not a confidence interval or price guarantee. "
                "Setup/download, CPU analysis, storage, network and idle rental are additional.",
    }
