"""Observation-only epoch probes and reproducible overfitting warning charts.

Probe metrics never control the optimizer, schedule, epoch count or checkpoint
selection. A warning is descriptive evidence, not a statistical diagnosis.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import config, utils
from .data import make_loader
from .protocol import digest, immutable_json


def stratified_indices(labels, count, seed=config.DATA_SEED):
    labels = np.asarray(labels)
    count = min(count, len(labels))
    if count <= 0:
        raise ValueError("Cannot diagnose an empty dataset")
    values, counts = np.unique(labels, return_counts=True)
    exact = counts * count / len(labels)
    quota = np.floor(exact).astype(int)
    order = np.argsort(-(exact - quota), kind="stable")
    quota[order[:count - int(quota.sum())]] += 1
    rng = np.random.RandomState(seed)
    selected = []
    for label, take in zip(values, quota):
        selected.extend(rng.permutation(np.flatnonzero(labels == label))[:take].tolist())
    return sorted(selected)


class EpochDiagnostics:
    def __init__(self, transformer, head, task, tokenizer, run_dir, device, samples, batch):
        self.transformer, self.head = transformer, head
        self.task, self.device = task.spec.key, device
        self.root = Path(run_dir)
        self.loaders = {}
        selection = {"task": self.task, "data_seed": config.DATA_SEED,
                     "samples_requested": samples, "policy": "stratified_observation_only"}
        for split, dataset in (("train", task.train_dataset), ("validation", task.eval_dataset)):
            indices = stratified_indices(dataset["labels"], samples)
            selected = dataset.select(indices)
            self.loaders[split] = make_loader(selected, batch, False, config.DATA_SEED, tokenizer)
            selection[split] = {"positions": indices, "count": len(indices),
                                "indices_sha256": digest(indices),
                                "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
                                "label_counts": {str(v): int(sum(int(y) == v for y in selected["labels"]))
                                                 for v in sorted(set(selected["labels"]))}}
        immutable_json(self.root / "diagnostics" / f"selection_{self.task}.json", selection)

    def __call__(self, epoch, step, online_train_loss):
        from .train import evaluate
        begin = time.monotonic()
        devices = [torch.cuda.current_device()] if str(self.device).startswith("cuda") else []
        record = {"task": self.task, "epoch": epoch, "step": step,
                  "online_train_loss": online_train_loss, "model_selection": False}
        # Unshuffled DataLoaders still consume the global CPU RNG for base_seed.
        # Restore both CPU and the active CUDA generator after observing probes.
        with torch.random.fork_rng(devices=devices):
            for split, loader in self.loaders.items():
                path = self.root / "diagnostics" / self.task / f"epoch{epoch}_{split}.npz"
                record[split] = {**evaluate(self.transformer, self.head, loader, self.device,
                                           path, return_details=True, save_scores=True),
                                 "count": len(loader.dataset),
                                 "predictions": str(path.relative_to(self.root))}
        record["nll_gap"] = record["validation"]["nll"] - record["train"]["nll"]
        record["accuracy_gap"] = record["train"]["accuracy"] - record["validation"]["accuracy"]
        record["diagnostic_elapsed_s"] = time.monotonic() - begin
        utils.append_jsonl(self.root / "epoch_diagnostics.jsonl", record)


def summarize_run(rows):
    tasks = {}
    for key in dict.fromkeys(row["task"] for row in rows):
        ordered = sorted((row for row in rows if row["task"] == key), key=lambda row: row["epoch"])
        warnings = []
        for prior, current in zip(ordered, ordered[1:]):
            rise = current["validation"]["nll"] - prior["validation"]["nll"]
            fall = prior["train"]["nll"] - current["train"]["nll"]
            if rise > .02 and fall > .02:
                warnings.append({"epoch": current["epoch"], "validation_nll_rise": rise,
                                 "train_nll_fall": fall,
                                 "validation_accuracy_change": current["validation"]["accuracy"]
                                     - prior["validation"]["accuracy"]})
        best = min(ordered, key=lambda row: (row["validation"]["nll"], row["epoch"]))
        tasks[key] = {"possible_overfitting": bool(warnings), "warnings": warnings,
                      "observed_epochs": [r["epoch"] for r in ordered],
                      "lowest_probe_nll_epoch": best["epoch"],
                      "last_epoch_nll_gap": ordered[-1]["nll_gap"]}
    return {"policy": "observation_only_fixed_epochs_no_selection",
            "possible_overfitting": any(t["possible_overfitting"] for t in tasks.values()),
            "warning_rule": {"validation_nll_rise_gt": .02, "train_nll_fall_gt": .02},
            "caveat": "Small fixed probes provide descriptive warnings; distribution differences "
                      "and sampling uncertainty can also produce gaps. Official full validation "
                      "remains the task-boundary measurement. Lowest-probe epoch is not selected.",
            "tasks": tasks}


def write_reports(root):
    """Make CPU-only JSON/CSV/charts from completed runs; no training launches."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    destination = root / "overfitting"
    destination.mkdir(parents=True, exist_ok=True)
    records, summaries = [], []
    result_paths = list((root / "runs").glob("*/result.json"))
    result_paths += list((root / "main").glob("*/runs/*/result.json"))
    for path in sorted(result_paths):
        result = json.loads(path.read_text())
        log = path.parent / "epoch_diagnostics.jsonl"
        if result.get("status") != "completed" or not log.exists():
            continue
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        summary = summarize_run(rows)
        identity = {k: result[k] for k in ("run_id", "model", "rank", "seed", "lr", "head_lr_multiplier")}
        summaries.append({**identity, **summary})
        for row in rows:
            records.append({**identity, "task": row["task"], "epoch": row["epoch"],
                            "step": row["step"], "online_train_loss": row["online_train_loss"],
                            **{f"{s}_{m}": row[s][m] for s in ("train", "validation")
                               for m in ("nll", "accuracy", "count")},
                            "nll_gap": row["nll_gap"], "accuracy_gap": row["accuracy_gap"],
                            "diagnostic_elapsed_s": row["diagnostic_elapsed_s"]})
    utils.save_json(destination / "summary.json", {"completed_sequences": len(summaries),
                    "possible_overfitting_sequences": sum(r["possible_overfitting"] for r in summaries),
                    "checkpoint_selection": False, "runs": summaries})
    if not records:
        return {"completed_sequences": 0, "charts": []}
    with (destination / "epoch-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    charts = []
    groups = sorted({(r["model"], r["rank"]) for r in records})
    for model, rank in groups:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
        selected = [r for r in records if (r["model"], r["rank"]) == (model, rank)]
        for column, task in enumerate(config.TASK_KEYS):
            for split, color in (("train", "tab:blue"), ("validation", "tab:orange")):
                for i, metric in enumerate(("nll", "accuracy")):
                    epochs = sorted({r["epoch"] for r in selected if r["task"] == task})
                    values = [np.asarray([r[f"{split}_{metric}"] for r in selected
                                          if r["task"] == task and r["epoch"] == epoch]) for epoch in epochs]
                    mean = np.asarray([v.mean() for v in values])
                    std = np.asarray([v.std(ddof=1) if len(v) > 1 else 0 for v in values])
                    axes[i, column].plot(epochs, mean, "o-", label=split, color=color)
                    axes[i, column].fill_between(epochs, mean - std, mean + std, alpha=.15, color=color)
                    axes[i, column].set(xlabel="Epoch", ylabel=metric.upper() if metric == "nll" else "Accuracy",
                                        title=task.upper(), xticks=epochs)
                    axes[i, column].grid(alpha=.2)
            axes[0, column].legend()
        fig.suptitle(f"{model}, factor width {rank}: fixed probes, mean ± seed SD\n"
                     "Observation only; no checkpoint selection")
        fig.tight_layout()
        for suffix in ("png", "svg"):
            path = destination / f"{model}_r{rank}.{suffix}"
            fig.savefig(path, dpi=160)
            charts.append(str(path.relative_to(root)))
        plt.close(fig)
    return {"completed_sequences": len(summaries), "possible_overfitting_sequences":
            sum(r["possible_overfitting"] for r in summaries), "charts": charts}
