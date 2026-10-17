"""Audit the requested 140-sequence LoRA cohort separately from full-FT design.

Run from the repository root: python -m tools.audit_requested_results
This reads saved evidence only and starts no cloud computation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from rank_suite import config
from rank_suite.metrics import compute_run_metrics
from rank_suite.protocol import digest

RANKS = (4, 8, 16, 32, 64, 128, 256)
EVAL_COUNTS = {"rte": 277, "mnli": 9815, "sst2": 872}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(root):
    protocol = json.loads((root / "protocol.json").read_text())
    policy = json.loads((root / "lr_defaults.json").read_text())["selected"]
    expected = {(m, r, s) for m in config.MODELS for r in RANKS for s in config.DEFAULT_SEEDS}
    found, errors, flagged = [], [], []
    labels, data_hashes, base_checks = {}, {}, {}
    spectra = 0
    checkpoint_bytes = 0
    for path in sorted((root / "runs").glob("*/result.json")):
        result = json.loads(path.read_text())
        if result.get("mode") != "main":
            continue
        try:
            def require(condition, message):
                if not condition:
                    raise ValueError(message)

            model = result["model"]
            identity = (model, result["rank"], result["seed"])
            require(identity in expected, "Unexpected model/rank/seed")
            require(result["status"] == "completed", "Training incomplete")
            require(result["track"] == "lora", "Unexpected training track")
            require(result["protocol_sha256"] == digest(protocol), "Protocol mismatch")
            require(result["hf_revision"] == config.MODELS[model].revision, "Model revision mismatch")
            require(result["task_order"] == list(EVAL_COUNTS), "Task order mismatch")
            require(result["epochs"] == 3 and result["train_samples"] == 2490, "Training scope mismatch")
            require(result["lr"] == policy[model]["lora"]["lr"], "Selected LR mismatch")
            require(result["head_lr_multiplier"] == policy[model]["lora"]["head_lr_multiplier"], "Head LR mismatch")
            require(result["batch"] == {"effective": 16, "micro": 16, "accum": 1, "eval": 32}, "Batch mismatch")
            require("L40S" in result["hardware"]["name"], "GPU type mismatch")
            compute_run_metrics(result["A"], result["task_order"])
            for task, count in EVAL_COUNTS.items():
                data = result["data"][task]
                require(data["n_train"] == 2490 and data["n_eval"] == count, "Dataset cardinality mismatch")
                require(not data["calibration"] and data["revision"] == config.DATA_REVISION, "Dataset policy mismatch")
                key = (model, task)
                fingerprint = (data["training_indices_sha256"], data["train_fingerprint"], data["eval_fingerprint"])
                require(data_hashes.setdefault(key, fingerprint) == fingerprint, "Within-model data selection mismatch")
                require(result["train_summary"][task]["steps"] == 3 * math.ceil(2490 / 16), "Optimizer step count mismatch")
            for name, size in result["checkpoint_sizes"].items():
                require((path.parent / name).stat().st_size == size, "Checkpoint missing or wrong size: " + name)
            checkpoint_bytes += result["checkpoint_total_bytes"]
            for i, boundary in enumerate(result["boundaries"]):
                checkpoint = path.parent / boundary["path"]
                require(sha256(checkpoint / "projections.pt") == boundary["projections_sha256"], "Projection checksum mismatch")
                for name in ("adapter_model.safetensors", "adapter_config.json", "heads.pt"):
                    require((checkpoint / name).is_file(), "Missing final export: " + name)
                for j, task in enumerate(result["task_order"][:i + 1]):
                    with np.load(checkpoint / f"predictions_{task}.npz", allow_pickle=False) as prediction:
                        pred, gold = prediction["predictions"], prediction["labels"]
                        require(pred.shape == gold.shape == (EVAL_COUNTS[task],), "Prediction count mismatch")
                        require(abs(float(np.mean(pred == gold)) - result["A"][i][j]) < 1e-12, "Saved predictions disagree with accuracy")
                        label_sha = hashlib.sha256(gold.tobytes()).hexdigest()
                        require(labels.setdefault((model, task), label_sha) == label_sha, "Evaluation labels/order mismatch")
            log = [json.loads(line) for line in (path.parent / "train_log.jsonl").read_text().splitlines()]
            require([row["step"] for row in log] == list(range(1, 1405)), "Incomplete optimizer log")
            require(all(math.isfinite(row[k]) for row in log for k in ("loss", "lr", "head_lr", "grad_norm")), "Nonfinite optimizer log")
            base = root / "cache" / "projections" / model / result["hf_revision"] / "base.pt"
            if model not in base_checks:
                metadata = json.loads(base.with_suffix(".json").read_text())
                require(sha256(base) == metadata["sha256"], "Pretrained projection checksum mismatch")
                base_checks[model] = metadata["sha256"]
            if result.get("intruder_status") == "completed":
                require(len(result["intruder"]) == 3, "Missing spectral task boundary")
                require(all(x["n_matrices"] == 2 * config.MODELS[model].layers for x in result["intruder"].values()), "Spectral matrix count mismatch")
                spectra += 1
            if result.get("flags", {}).get("grad_spike_steps"):
                flagged.append(result["run_id"])
            found.append(identity)
        except (ValueError, KeyError, OSError) as exc:
            errors.append({"run_id": path.parent.name, "error": str(exc)})
    duplicates = [list(key) for key, n in Counter(found).items() if n > 1]
    missing = sorted(expected - set(found))
    return {"status": "complete" if not missing and not duplicates and not errors and spectra == 140 else "partial",
            "scope": "Requested LoRA-only cohort; no full-FT baseline certification.",
            "expected_sequences": 140, "verified_sequences": len(found), "completed_spectral_sequences": spectra,
            "ranks": list(RANKS), "seeds": list(config.DEFAULT_SEEDS), "missing": missing,
            "duplicates": duplicates, "errors": errors, "gradient_spike_flagged_runs": flagged,
            "protocol_sha256": digest(protocol), "scientific_source_sha256": protocol["source_sha256"],
            "checkpoint_bytes": checkpoint_bytes, "verified_pretrained_projection_sha256": base_checks,
            "selected_policies": policy}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="Finetune-result")
    args = parser.parse_args()
    root = Path(args.root)
    report = audit(root)
    (root / "requested-completeness.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in
                      ("missing", "selected_policies", "gradient_spike_flagged_runs")}, indent=2))
    print(f"Missing sequences: {len(report['missing'])}; gradient-spike flagged runs: "
          f"{len(report['gradient_spike_flagged_runs'])}. Full evidence: {root / 'requested-completeness.json'}")


if __name__ == "__main__":
    main()
