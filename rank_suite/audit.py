"""Completeness is a scientific outcome, independent of whether commands succeeded."""
from . import config


def completeness_audit(results, mode="main", default_policy=None):
    present = {(r["model"], r["track"], r.get("rank"), r["lr"], r["seed"])
               for r in results if r.get("mode") == "main"}
    missing = []
    for key, m in config.MODELS.items():
        def rate(track):
            declared = {r[f"declared_{track}_lr"] for r in results
                        if r["model"] == key and r.get(f"declared_{track}_lr") is not None}
            if len(declared) > 1:
                raise ValueError("Conflicting declared LR anchors")
            return (next(iter(declared)) if declared else
                    (default_policy or {}).get(key, {}).get(track, {}).get("lr", getattr(m, f"default_{track}_lr")))
        for seed in config.DEFAULT_SEEDS:
            for track, rank, lr in [("lora", r, rate("lora")) for r in config.METHODOLOGY_RANKS] + [
                    ("full", None, rate("full"))]:
                if (key, track, rank, lr, seed) not in present:
                    missing.append({"model": key, "track": track, "rank": rank, "lr": lr, "seed": seed})
    pending = sum(r.get("intruder_status") != "completed" for r in results)
    flagged = [r["run_id"] for r in results if r.get("flags", {}).get("grad_spike_steps")]
    return {"status": ("complete" if not missing and not pending else "incomplete")
                       if mode == "main" else "diagnostic_only",
            "behavioral_design_complete": mode == "main" and not missing,
            "spectral_design_complete": mode == "main" and not missing and not pending,
            "expected_main_runs": 120, "loaded_runs": len(results),
            "missing": missing, "pending_spectra": pending, "flagged_runs": flagged}
