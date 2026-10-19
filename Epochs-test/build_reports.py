"""Validate independent epoch-budget evidence; generate CPU charts, Word and PPTX."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from epoch_study import EPOCH_BUDGETS, MODELS, SEEDS, STEPS_PER_EPOCH, expected_grid
from rank_suite import config, utils
from rank_suite.protocol import digest

HERE = Path(__file__).resolve().parent
REFERENCES = [
    (
        "Hu et al. LoRA: Low-Rank Adaptation of Large Language Models (2021/2022)",
        "https://arxiv.org/abs/2106.09685",
    ),
    ("Wang et al. GLUE (2018/2019)", "https://arxiv.org/abs/1804.07461"),
    (
        "Dodge et al. Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping (2020)",
        "https://arxiv.org/abs/2002.06305",
    ),
    ("Prechelt. Early Stopping—But When? (1998)", "https://doi.org/10.1007/3-540-49430-8_3"),
    (
        "Hugging Face. Cosine schedule with warmup",
        "https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.get_cosine_schedule_with_warmup",
    ),
]
METRICS = (
    "train_nll",
    "validation_nll",
    "train_accuracy",
    "validation_accuracy",
    "nll_gap",
    "accuracy_gap",
    "validation_brier",
    "validation_wrong_confidence",
    "mean_adapter_lr",
)


def mean_sd(values):
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "n": len(a),
        "values": a.tolist(),
    }


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def scheduled_lr(index, budget, peak):
    """Actual LR used at zero-based optimizer step, before scheduler.step()."""
    total = STEPS_PER_EPOCH * budget
    warmup = math.ceil(config.WARMUP_FRAC * total)
    factor = (
        index / warmup
        if index < warmup
        else 0.5 * (1 + math.cos(math.pi * (index - warmup) / (total - warmup)))
    )
    return peak * factor


def prediction_metrics(path, expected, count):
    with np.load(path, allow_pickle=False) as f:
        labels, pred, prob, nll = (f[k] for k in ("labels", "predictions", "probabilities", "example_nll"))
    if (
        labels.shape != (count,)
        or pred.shape != (count,)
        or nll.shape != (count,)
        or prob.shape != (count, 2)
        or not np.isin(labels, [0, 1]).all()
        or not np.isfinite(prob).all()
        or not np.isfinite(nll).all()
        or (prob < 0).any()
        or (prob > 1).any()
        or (nll < 0).any()
        or not np.allclose(prob.sum(axis=1), 1, atol=1e-5)
        or not np.array_equal(pred, prob.argmax(axis=1))
    ):
        raise ValueError(f"Invalid prediction arrays: {path}")
    correct_prob = prob[np.arange(count), labels]
    # FP32 softmax can underflow for a confident wrong answer; CE stays finite.
    stable = correct_prob > np.finfo(np.float32).tiny
    if (
        abs(float(nll.mean()) - expected["nll"]) > 2e-5
        or abs(float((labels == pred).mean()) - expected["accuracy"]) > 1e-12
        or not np.allclose(-np.log(correct_prob[stable]), nll[stable], atol=2e-5)
    ):
        raise ValueError(f"Prediction/metric mismatch: {path}")
    wrong = labels != pred
    return {
        "brier": float(np.square(prob - np.eye(2)[labels]).sum(axis=1).mean()),
        "wrong_confidence": float(prob.max(axis=1)[wrong].mean()) if wrong.any() else 0.0,
        "wrong_count": int(wrong.sum()),
        "labels_sha256": hashlib.sha256(labels.astype(np.int64).tobytes()).hexdigest(),
    }


def analyze(root):
    root = Path(root)
    control = json.loads((root / "protocol.json").read_text())
    grid = json.loads((root / "expected_grid.json").read_text())
    if control.get("schema") != "independent-epoch-budget-study-v1" or grid != expected_grid("sst2"):
        raise ValueError("Expected the separate, prespecified 48-run independent-budget cohort")
    sha = digest(control)
    records, times, missing, failures, rejected = [], [], [], [], []
    paired_initializations, label_checksums = {}, {}
    for identity in grid:
        run_root = root / "models" / identity["model"] / "runs" / identity["run_id"]
        path = run_root / "result.json"
        if not path.exists():
            missing.append(identity["run_id"])
            continue
        result = json.loads(path.read_text())
        if result.get("status") != "completed":
            failures.append(
                {"run_id": identity["run_id"], "status": result.get("status"), "error": result.get("error")}
            )
            continue
        try:
            if result.get("protocol_sha256") != sha or any(result.get(k) != v for k, v in identity.items()):
                raise ValueError("Identity/protocol mismatch")
            budget = identity["epoch_budget"]
            observations = read_rows(run_root / result["epoch_metrics"])
            if [r["epoch"] for r in observations] != list(range(1, budget + 1)):
                raise ValueError("Missing, duplicate or out-of-order within-fit epoch observations")
            initial = result["initialization"]
            if result["final_adapter_sha256"] == initial["adapter_sha256"]:
                raise ValueError("Adapter parameters never changed")
            pair_key = (identity["model"], identity["seed"])
            if pair_key in paired_initializations and initial != paired_initializations[pair_key]:
                raise ValueError("Budgets did not start with matching adapter/head initialization")
            paired_initializations[pair_key] = initial
            steps = read_rows(run_root / result["attempt"] / "train_steps.jsonl")
            total = STEPS_PER_EPOCH * budget
            if len(steps) != total or [s["step"] for s in steps] != list(range(1, total + 1)):
                raise ValueError("Training did not record all planned optimizer steps")
            for index, s in enumerate(steps):
                if (
                    s["epoch"] != index // STEPS_PER_EPOCH + 1
                    or s["task"] != "sst2"
                    or s["examples"] != (10 if index % STEPS_PER_EPOCH == 155 else 16)
                    or not np.isfinite([s["loss"], s["grad_norm"], s["lr"], s["head_lr"]]).all()
                    or not math.isclose(
                        s["lr"], scheduled_lr(index, budget, identity["lr"]), rel_tol=1e-8, abs_tol=1e-14
                    )
                    or not math.isclose(
                        s["head_lr"], s["lr"] * identity["head_lr_multiplier"], rel_tol=1e-8, abs_tol=1e-14
                    )
                ):
                    raise ValueError("Invalid optimizer steps or budget-matched schedule")
            local = []
            for row in observations:
                if (
                    any(row.get(k) != v for k, v in identity.items())
                    or row["task"] != "sst2"
                    or row["step"] != row["epoch"] * STEPS_PER_EPOCH
                ):
                    raise ValueError("Observation identity/task/step mismatch")
                flat = {
                    **identity,
                    "epoch": row["epoch"],
                    "is_endpoint": row["epoch"] == budget,
                    "online_train_loss": row["online_train_loss"],
                    "nll_gap": row["nll_gap"],
                    "accuracy_gap": row["accuracy_gap"],
                }
                for split, count in (("train", 2490), ("validation", 872)):
                    if row[split]["count"] != count:
                        raise ValueError("Unexpected full-set evaluation count")
                    extra = prediction_metrics(run_root / row[split]["predictions"], row[split], count)
                    labels_sha = extra.pop("labels_sha256")
                    if split in label_checksums and labels_sha != label_checksums[split]:
                        raise ValueError("Evaluation example labels/order differ across fits")
                    label_checksums[split] = labels_sha
                    flat.update({f"{split}_{k}": row[split][k] for k in ("nll", "accuracy", "count")})
                    flat.update({f"{split}_{k}": v for k, v in extra.items()})
                if not math.isclose(
                    flat["nll_gap"], flat["validation_nll"] - flat["train_nll"], abs_tol=1e-12
                ) or not math.isclose(
                    flat["accuracy_gap"], flat["train_accuracy"] - flat["validation_accuracy"], abs_tol=1e-12
                ):
                    raise ValueError("Invalid reported generalization gap")
                epoch_steps = steps[(row["epoch"] - 1) * STEPS_PER_EPOCH : row["epoch"] * STEPS_PER_EPOCH]
                flat["mean_adapter_lr"] = float(np.mean([s["lr"] for s in epoch_steps]))
                flat["last_adapter_lr"] = epoch_steps[-1]["lr"]
                local.append(flat)
            records.extend(local)
            times.append(
                {
                    **identity,
                    "elapsed_s": result["elapsed_s"],
                    "peak_vram_gb": result["peak_vram_gb"],
                    "optimizer_steps": total,
                }
            )
        except (ValueError, KeyError, OSError, TypeError) as exc:
            rejected.append({"run_id": identity["run_id"], "reason": str(exc)})
    endpoints = [r for r in records if r["is_endpoint"]]
    models = {}
    for model in MODELS:
        selected = [r for r in endpoints if r["model"] == model]
        budgets = {}
        for budget in EPOCH_BUDGETS:
            rows = sorted([r for r in selected if r["epoch_budget"] == budget], key=lambda r: r["seed"])
            if rows:
                budgets[str(budget)] = {
                    "seeds": [r["seed"] for r in rows],
                    **{k: mean_sd([r[k] for r in rows]) for k in METRICS},
                }
        complete = len(selected) == len(SEEDS) * len(EPOCH_BUDGETS)
        reference = {r["seed"]: r for r in selected if r["epoch_budget"] == 3}
        comparisons = []
        for budget in (4, 5, 6):
            paired = [
                {
                    "seed": r["seed"],
                    "train_nll_change": r["train_nll"] - reference[r["seed"]]["train_nll"],
                    "validation_nll_change": r["validation_nll"] - reference[r["seed"]]["validation_nll"],
                    "accuracy_change_pp": 100
                    * (r["validation_accuracy"] - reference[r["seed"]]["validation_accuracy"]),
                }
                for r in selected
                if r["epoch_budget"] == budget and r["seed"] in reference
            ]
            if paired:
                changes = {
                    k: mean_sd([r[k] for r in paired])
                    for k in ("train_nll_change", "validation_nll_change", "accuracy_change_pp")
                }
                count = sum(r["train_nll_change"] < 0 and r["validation_nll_change"] > 0 for r in paired)
                comparisons.append(
                    {
                        "epoch_budget": budget,
                        "reference": 3,
                        "n_pairs": len(paired),
                        **changes,
                        "paired_values": paired,
                        "overfitting_direction_seeds": count,
                        "mean_overfitting_pattern": changes["train_nll_change"]["mean"] < 0
                        and changes["validation_nll_change"]["mean"] > 0,
                        "consistent_descriptive_pattern": len(paired) == len(SEEDS)
                        and count == len(SEEDS)
                        and changes["validation_nll_change"]["mean"] > 0.02,
                    }
                )
        best_nll = (
            min(budgets, key=lambda b: (budgets[b]["validation_nll"]["mean"], int(b))) if complete else None
        )
        best_acc = (
            max(budgets, key=lambda b: (budgets[b]["validation_accuracy"]["mean"], -int(b)))
            if complete
            else None
        )
        models[model] = {
            "complete": complete,
            "budgets": budgets,
            "best_observed_nll_budget": int(best_nll) if best_nll else None,
            "best_observed_accuracy_budget": int(best_acc) if best_acc else None,
            "budget3_lowest_mean_nll": int(best_nll) == 3 if best_nll else None,
            "comparisons_to_budget3": comparisons,
        }
    complete = len(times) == len(grid) and not rejected and not failures
    supported = (
        all(
            m["budget3_lowest_mean_nll"]
            and len(m["comparisons_to_budget3"]) == 3
            and all(c["consistent_descriptive_pattern"] for c in m["comparisons_to_budget3"])
            for m in models.values()
        )
        if complete
        else None
    )
    summary = {
        "schema": "independent-epoch-budget-analysis-v1",
        "analysis_revision": 1,
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "complete": complete,
        "expected_runs": len(grid),
        "completed_runs": len(times),
        "epoch_observations": len(records),
        "endpoint_observations": len(endpoints),
        "optimizer_steps": sum(r["optimizer_steps"] for r in times),
        "scope": control["scope"],
        "protocol_sha256": sha,
        "models": models,
        "missing": missing,
        "failures": failures,
        "rejected": rejected,
        "epoch3_hypothesis_supported_descriptively": supported,
        "independent_epoch_budget_comparison": True,
        "seeds": list(SEEDS),
        "paired_initializations_verified": len(paired_initializations),
        "labels_sha256": label_checksums,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    return summary, records, times, control


def verdict(summary):
    if not summary["complete"]:
        return f"Study incomplete: {summary['completed_runs']}/{summary['expected_runs']} fits passed validation. No overall epoch-budget conclusion is declared."
    count = sum(m["budget3_lowest_mean_nll"] for m in summary["models"].values())
    if summary["epoch3_hypothesis_supported_descriptively"]:
        return "Budget 3 has the lowest observed mean validation NLL in all four models; budgets 4–6 meet the predeclared descriptive overfitting rule versus budget 3. This supports the hypothesis in this two-seed SST-2 study, without proving a universal optimum."
    return f"Budget 3 has the lowest observed mean validation NLL in {count}/4 models. The full claim that three epochs are best for every model and all larger budgets consistently overfit is not supported."


def model_statement(model, data):
    if not data["complete"]:
        return f"{config.MODELS[model].label}: incomplete; no preferred budget declared."
    later = [
        str(c["epoch_budget"]) for c in data["comparisons_to_budget3"] if c["consistent_descriptive_pattern"]
    ]
    return f"{config.MODELS[model].label}: lowest mean NLL at {data['best_observed_nll_budget']} epochs; highest mean accuracy at {data['best_observed_accuracy_budget']}. Consistent loss-overfitting versus budget 3 at: {', '.join(later) or 'none'}."


def fmt(stat, scale=1, digits=3):
    return f"{stat['mean'] * scale:.{digits}f} ± {stat['sd'] * scale:.{digits}f}"


def figures(root, summary, records):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
        }
    )
    out = root / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    endpoint = [r for r in records if r["is_endpoint"]]
    blue, orange = "#176B87", "#D76C35"

    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        paths[name] = out / f"{name}.png"

    def curve(ax, rows, key, label, color, scale=1, seed_points=True):
        budgets = sorted({r["epoch_budget"] for r in rows})
        groups = [[r[key] * scale for r in rows if r["epoch_budget"] == b] for b in budgets]
        if groups:
            ax.errorbar(
                budgets,
                [np.mean(g) for g in groups],
                yerr=[np.std(g, ddof=1) if len(g) > 1 else 0 for g in groups],
                label=label,
                color=color,
                lw=1.8,
                marker="o",
                capsize=3,
            )
        if seed_points:
            for seed, marker, offset in zip(SEEDS, ("o", "D"), (-0.055, 0.055)):
                selected = sorted([r for r in rows if r["seed"] == seed], key=lambda r: r["epoch_budget"])
                ax.plot(
                    [r["epoch_budget"] + offset for r in selected],
                    [r[key] * scale for r in selected],
                    color=color,
                    alpha=0.35,
                    marker=marker,
                    ls=":",
                    lw=0.8,
                    ms=4,
                )
        ax.set_xticks(EPOCH_BUDGETS)
        ax.set_xlabel("Planned epoch budget (fresh fit)")
        ax.axvline(3, color="#89949A", ls=":", lw=1)
        ax.grid(alpha=0.18)

    for key, name, title, scale in (
        ("validation_nll", "validation-overview", "Endpoint validation NLL (lower is better)", 1),
        ("validation_accuracy", "accuracy-overview", "Endpoint validation accuracy (%)", 100),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.8), layout="constrained")
        for ax, model in zip(axes.flat, MODELS):
            rows = [r for r in endpoint if r["model"] == model]
            curve(ax, rows, key, "Mean ± SD", blue, scale)
            ax.set_title(config.MODELS[model].label)
            ax.set_ylabel("NLL" if scale == 1 else "Accuracy (%)")
        fig.suptitle(title + "\nTwo seeds; faint dotted lines show individual paired seeds", fontsize=13)
        save(fig, name)

    for model in MODELS:
        rows = [r for r in endpoint if r["model"] == model]
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), layout="constrained")
        ax = axes[0, 0]
        curve(ax, rows, "train_nll", "Train NLL", blue)
        curve(ax, rows, "validation_nll", "Validation NLL", orange)
        ax.set_ylabel("Full-set NLL")
        ax.legend(fontsize=8)
        ax = axes[0, 1]
        curve(ax, rows, "train_accuracy", "Train", blue, 100)
        curve(ax, rows, "validation_accuracy", "Validation", orange, 100)
        ax.set_ylabel("Accuracy (%)")
        ax.legend(fontsize=8)
        ax = axes[1, 0]
        curve(ax, rows, "nll_gap", "Validation − train", orange)
        ax.set_ylabel("NLL generalization gap")
        ax.axhline(0, color="grey", lw=0.7)
        ax = axes[1, 1]
        comps = summary["models"][model]["comparisons_to_budget3"]
        for key, label, color, offset in (
            ("train_nll_change", "Train", blue, -0.07),
            ("validation_nll_change", "Validation", orange, 0.07),
        ):
            ax.errorbar(
                [c["epoch_budget"] + offset for c in comps],
                [c[key]["mean"] for c in comps],
                yerr=[c[key]["sd"] for c in comps],
                color=color,
                label=label,
                marker="o",
                capsize=3,
            )
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks([4, 5, 6])
        ax.set_xlabel("Epoch budget, compared with budget 3")
        ax.set_ylabel("Paired endpoint NLL change")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.18)
        fig.suptitle(
            config.MODELS[model].label + " | independent budget comparison; mean ± sample SD", fontsize=13
        )
        save(fig, model + "-budgets")

        fig, axes = plt.subplots(2, 3, figsize=(11.2, 7), layout="constrained")
        for ax, budget in zip(axes.flat, EPOCH_BUDGETS):
            selected = [r for r in records if r["model"] == model and r["epoch_budget"] == budget]
            for split, label, color in (("train", "Train", blue), ("validation", "Validation", orange)):
                epochs = sorted({r["epoch"] for r in selected})
                groups = [[r[f"{split}_nll"] for r in selected if r["epoch"] == e] for e in epochs]
                if groups:
                    means = np.array([np.mean(g) for g in groups])
                    sd = np.array([np.std(g, ddof=1) if len(g) > 1 else 0 for g in groups])
                    ax.plot(epochs, means, color=color, marker="o", ms=3, label=label)
                    ax.fill_between(epochs, np.maximum(0, means - sd), means + sd, color=color, alpha=0.12)
            ax.set_title(f"Planned budget: {budget} epoch{'s' if budget > 1 else ''}")
            ax.set_xticks(range(1, budget + 1))
            ax.set_xlim(0.8, max(1.2, budget + 0.2))
            ax.set_xlabel("Observed epoch within this fit")
            ax.set_ylabel("Full-set NLL")
            ax.grid(alpha=0.18)
            if budget == 1:
                ax.legend(fontsize=8)
        fig.suptitle(
            config.MODELS[model].label
            + " | within-fit loss curves\nEach panel has a fresh, budget-matched schedule; shading = SD across two seeds",
            fontsize=13,
        )
        save(fig, model + "-learning-curves")

    matrix = np.full((4, 3), np.nan)
    for index, model in enumerate(MODELS):
        for comp in summary["models"][model]["comparisons_to_budget3"]:
            matrix[index, comp["epoch_budget"] - 4] = comp["validation_nll_change"]["mean"]
    fig, ax = plt.subplots(figsize=(9.5, 4.5), layout="constrained")
    limit = max(0.02, float(np.nanmax(np.abs(matrix)))) if np.isfinite(matrix).any() else 0.02
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(3), ["4 epochs", "5 epochs", "6 epochs"])
    ax.set_yticks(range(4), [config.MODELS[m].label for m in MODELS])
    for i in range(4):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:+.3f}" if np.isfinite(matrix[i, j]) else "missing",
                ha="center",
                va="center",
                fontsize=12,
                color="white" if abs(matrix[i, j]) > limit * 0.65 else "black",
            )
    fig.colorbar(im, ax=ax, label="Mean validation NLL change versus budget 3")
    ax.set_title(
        "Endpoint validation loss change relative to 3 epochs\nNegative/blue = lower NLL; positive/red = higher NLL"
    )
    save(fig, "comparison-to-budget3")

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.8), layout="constrained")
    for ax, model in zip(axes.flat, MODELS):
        curve(ax, [r for r in endpoint if r["model"] == model], "validation_brier", "Brier", blue)
        ax.set_title(config.MODELS[model].label)
        ax.set_ylabel("Brier score (0–2; lower is better)")
    fig.suptitle("Endpoint probability quality | Brier score; mean ± sample SD", fontsize=13)
    save(fig, "brier-overview")

    fig, ax = plt.subplots(figsize=(10.5, 4.7), layout="constrained")
    for budget in EPOCH_BUDGETS:
        steps = np.arange(STEPS_PER_EPOCH * budget)
        ax.plot(
            (steps + 1) / STEPS_PER_EPOCH,
            [scheduled_lr(int(s), budget, 1) for s in steps],
            label=f"{budget} epochs",
        )
    ax.set_xlabel("Training progress (epochs)")
    ax.set_ylabel("Adapter LR / selected peak LR")
    ax.set_title("Each independent budget has its own warmup and cosine horizon")
    ax.legend(ncol=3)
    ax.grid(alpha=0.2)
    save(fig, "budget-matched-schedules")
    return paths


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Shading Accent 1"
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = str(value)
    for row in rows:
        for cell, value in zip(table.add_row().cells, row):
            cell.text = str(value)
    return table


def model_table(doc, data):
    add_table(
        doc,
        ("Epoch budget", "Train NLL", "Validation NLL", "Validation acc. (%)", "NLL gap"),
        [
            (
                b,
                fmt(g["train_nll"]),
                fmt(g["validation_nll"]),
                fmt(g["validation_accuracy"], 100, 2),
                fmt(g["nll_gap"]),
            )
            for b, g in data["budgets"].items()
        ],
    )
    doc.add_paragraph(
        "Final measurements of separate fresh fits; mean ± sample SD over seeds 42 and 123. Lower NLL and higher accuracy are better.",
        "Caption",
    )


def word_report(root, summary, control, paths):
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor

    doc = Document()
    section = doc.sections[0]
    section.top_margin = section.bottom_margin = Inches(0.65)
    section.left_margin = section.right_margin = Inches(0.7)
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)
    for style in ("Title", "Heading 1", "Heading 2"):
        doc.styles[style].font.color.rgb = RGBColor.from_string("15384D")
    doc.add_heading("Independent epoch budgets and overfitting", 0)
    doc.add_paragraph("SST-2 · LoRA rank 16 · four fixed models · two seeds · budgets 1–6", "Subtitle")
    doc.add_paragraph(
        f"Generated {summary['created_utc']}\n{summary['completed_runs']}/48 completed and validated fits; {summary['endpoint_observations']} endpoints; {summary['epoch_observations']} within-fit epoch observations.\nProtocol SHA-256: {summary['protocol_sha256']}"
    )
    doc.add_heading("Findings", 1)
    doc.add_paragraph(verdict(summary))
    for model, result in summary["models"].items():
        doc.add_paragraph(model_statement(model, result))
    doc.add_heading("Research question and independent-run design", 1)
    doc.add_paragraph(
        "We test the hypothesis that three epochs provide the best validation performance and larger budgets cause overfitting. The outcome is not predetermined. Four pinned pretrained models [1] are each trained for six independently planned budgets (1, 2, 3, 4, 5, 6 epochs), with seeds 42 and 123: 48 fresh fits. Every fit starts from the pinned base weights with a fresh LoRA adapter, classification heads, AdamW optimizer, cosine scheduler and shuffle generator. Initialization checksums match across budgets within each model/seed. There is no warm-starting from an earlier fit, early stopping, adaptive tuning, seed removal, or best-checkpoint selection."
    )
    doc.add_paragraph(
        "Each fit uses the same 2,490 fixed SST-2 training examples and 872 official validation examples [2]. After every epoch we evaluate the entire training and validation sets with dropout disabled and save probabilities, labels and per-example NLL. Observations preserve Torch CPU/CUDA RNG state. Only the final epoch of each fit enters the budget comparison: 48 endpoint measurements. The 168 within-fit epoch observations are learning-curve measurements, not 168 independent replications. Two shared seeds are paired blocks, not enough replications for strong significance claims [3]."
    )
    add_table(
        doc,
        ("Model", "Adapter peak LR", "Head multiplier", "Head peak LR"),
        [
            (
                config.MODELS[m].label,
                f"{control['learning_rates'][m]:.3g}",
                f"{control['head_lr_multipliers'][m]:g}×",
                f"{control['learning_rates'][m] * control['head_lr_multipliers'][m]:.3g}",
            )
            for m in MODELS
        ],
    )
    doc.add_paragraph(
        "Selected learning rates are the best among the project's earlier tested candidates; they are held fixed here, not claimed to be globally optimal. Other controls: q/v LoRA rank 16, alpha 32, BF16 compute with FP32 trainable parameters and AdamW moments; effective/micro batch 16, eval batch 32, length 256, weight decay 0, dropout 0, gradient clipping 1. Dataset/model revisions, data indices, optimizer defaults and environment are retained in protocol.json and model selection/environment files."
    )
    doc.add_heading("Why the schedule is matched to each budget", 2)
    doc.add_paragraph(
        "Each epoch has 156 optimizer steps. A budget of E epochs therefore has 156E steps, ceil(0.03 × 156E) warmup steps and cosine decay toward zero over its own horizon [5]. The six budgets use 5, 10, 15, 19, 24 and 29 warmup steps respectively. This compares practical duration-plus-schedule policies: it does not isolate duration while holding the entire learning-rate path constant. It addresses the main limitation of interpreting epoch 3 from a single six-epoch schedule as a completed three-epoch fit. Intermediate observations from different horizons are never pooled."
    )
    doc.add_picture(str(paths["budget-matched-schedules"]), width=Inches(6.7))
    doc.add_heading("Metrics and the predeclared overfitting rule", 2)
    doc.add_paragraph(
        "NLL (negative log-likelihood) is the mean −log probability assigned to the correct label; lower is better. Training NLL is measured on the full training set after updates, rather than using the online batch loss as a generalization comparison. A fall in training NLL with a rise in validation NLL is consistent with loss-based overfitting [4]. A larger train–validation gap alone is insufficient, and optimization instability can also worsen validation loss. Accuracy measures correctness; NLL additionally penalizes confident mistakes, so the two can prefer different budgets. Brier score is the mean sum of squared probability errors over both classes (range 0–2)."
    )
    doc.add_paragraph(
        "The primary endpoint is mean validation NLL; accuracy, training NLL, gaps, Brier score and wrong-answer confidence are secondary. Budgets 4–6 are paired with budget 3 using the same seed. The descriptive consistency screen requires lower endpoint training NLL and higher validation NLL in BOTH paired seeds, with a mean validation NLL rise above 0.02. This 0.02 threshold is a predeclared practical screening choice, not a literature-derived significance threshold. All seed values and mean ± sample SD are reported; SD is not a confidence interval. Two seeds and reused validation data do not prove an optimal epoch or exact overfitting onset."
    )
    for key, caption in (
        (
            "validation-overview",
            "Final full-validation NLL of independent fits: means, sample SD and individual paired seeds.",
        ),
        (
            "accuracy-overview",
            "Final validation accuracy, shown separately because probability loss and correctness can disagree.",
        ),
        (
            "comparison-to-budget3",
            "Mean endpoint validation NLL change versus the independently completed three-epoch fits.",
        ),
    ):
        doc.add_picture(str(paths[key]), width=Inches(6.7))
        doc.add_paragraph(caption, "Caption")
    for model, result in summary["models"].items():
        doc.add_page_break()
        doc.add_heading(config.MODELS[model].label, 1)
        doc.add_paragraph(model_statement(model, result))
        model_table(doc, result)
        doc.add_picture(str(paths[model + "-budgets"]), width=Inches(6.7))
        doc.add_heading("Budgets 4–6 compared with budget 3", 2)
        for comp in result["comparisons_to_budget3"]:
            doc.add_paragraph(
                f"{comp['epoch_budget']} versus 3 epochs: train NLL change {fmt(comp['train_nll_change'])}; validation NLL change {fmt(comp['validation_nll_change'])}; accuracy change {fmt(comp['accuracy_change_pp'], digits=2)} percentage points. Loss-overfitting direction in {comp['overfitting_direction_seeds']}/{comp['n_pairs']} paired seeds; descriptive screen: {'met' if comp['consistent_descriptive_pattern'] else 'not met'}."
            )
        doc.add_picture(str(paths[model + "-learning-curves"]), width=Inches(6.7))
        doc.add_paragraph(
            "Each panel is a separate planned budget. Points show full-set NLL after each epoch, averaged over two fresh fits; shaded bands show sample SD. These curves describe training progress within a schedule and are not additional endpoint replications.",
            "Caption",
        )
    doc.add_heading("Probability quality", 1)
    doc.add_picture(str(paths["brier-overview"]), width=Inches(6.7))
    doc.add_paragraph(
        "The seed-level endpoint CSV also records wrong-prediction counts and mean confidence on wrong predictions. That confidence metric describes a changing set of mistakes and should not by itself be treated as a matched-example causal explanation. All per-example probabilities are retained for further CPU analysis."
    )
    doc.add_heading("Conclusion, uncertainty and applicability", 1)
    doc.add_paragraph(verdict(summary))
    doc.add_paragraph(
        "These are exploratory comparisons on a small, fixed SST-2 subset, rank 16 and two seeds. The official validation set was inspected in earlier LR/epoch studies and is reused; it is not an untouched test set. The minimum among six means is a descriptive choice subject to selection bias, and small differences should not be overinterpreted. Training a different seed, rank, task or full RTE → MNLI → SST-2 sequence can change the outcome. This study does not establish a universal best duration, retention performance or a causal memorization mechanism. The original suite's three-epoch default and rank-sweep experiments are unchanged."
    )
    if not summary["complete"]:
        doc.add_paragraph(
            f"Incomplete evidence: {len(summary['missing'])} missing, {len(summary['failures'])} failed/incomplete and {len(summary['rejected'])} rejected fits. Exact identifiers and reasons are recorded in summary.json; missing values are not imputed."
        )
    doc.add_heading("Reproducibility and retained evidence", 1)
    doc.add_paragraph(
        "The remote Modal volume and local results retain protocol/grid/source hashes, selected example indices, initialization checksums, per-step losses/learning rates/gradient norms, full train/validation per-epoch predictions, timing/VRAM and failures. CSV/JSON, PNG/PDF figures, this Word report and the PowerPoint are generated from the same validated evidence. No base/adapter model weights, optimizer states or projection tensors are exported. Failed fits require an explicit new training request; CPU report regeneration does not launch GPUs."
    )
    doc.add_heading("References", 1)
    for index, (title, url) in enumerate(REFERENCES, 1):
        doc.add_paragraph(f"[{index}] {title}\n{url}")
    section.footer.paragraphs[
        0
    ].text = f"Independent SST-2 epoch budgets | protocol {summary['protocol_sha256'][:12]}"
    doc.save(root / "reports" / "epoch-study.docx")
    tables = Document()
    tables.add_heading("Independent epoch-budget comparison", 0)
    tables.add_paragraph(verdict(summary))
    for model, result in summary["models"].items():
        tables.add_heading(config.MODELS[model].label, 1)
        model_table(tables, result)
        tables.add_paragraph(model_statement(model, result))
    tables.add_paragraph(
        "48 fresh fits, seeds 42/123, SST-2, rank16. NLL is the primary endpoint. Observed best budgets are descriptive; previously inspected validation data and two seeds limit generalization. See epoch-study.docx for complete methods and figures."
    )
    tables.save(root / "reports" / "epoch-comparison.docx")


def presentation(root, summary, control, paths):
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Inches, Pt
    from PIL import Image

    deck = Presentation()
    deck.slide_width, deck.slide_height = Inches(13.333), Inches(7.5)

    def text(item, value, x, y, w, h, size=22, bold=False):
        frame = item.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
        frame.word_wrap = True
        for i, line in enumerate(value.split("\n")):
            p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            p.text, p.font.size, p.font.bold = line, Pt(size), bold
            p.font.color.rgb = RGBColor(21, 56, 77)
            p.space_after = Pt(10)

    def slide(title, body=None, chart=None, note="", size=22):
        item = deck.slides.add_slide(deck.slide_layouts[6])
        text(item, title, 0.55, 0.23, 12.2, 0.7, 28, True)
        if body:
            text(item, body, 0.75, 1.2, 11.8, 5.3, size)
        if chart:
            with Image.open(chart) as picture:
                w, h = picture.size
            scale = min(12.1 / w, 5.55 / h)
            w, h = w * scale, h * scale
            item.shapes.add_picture(
                str(chart), Inches((13.333 - w) / 2), Inches(1.0), width=Inches(w), height=Inches(h)
            )
        text(item, note, 0.65, 6.65, 12, 0.55, 11)
        text(item, f"Epochs-test | independent budgets | SST-2 | {len(deck.slides)}", 0.65, 7.2, 12, 0.2, 9)

    slide(
        "Independent epoch budgets and overfitting",
        f"Four fixed models • LoRA rank 16 • seeds 42 and 123\n48 fresh fits across budgets 1–6\n\n{verdict(summary)}",
        note=f"Protocol SHA-256: {summary['protocol_sha256'][:20]}",
    )
    slide(
        "Design: fresh training for every planned duration",
        "• Four models × six budgets × two seeds = 48 fits.\n• Fresh pinned base, adapter, heads, optimizer and scheduler every time.\n• Same initialization and shuffle seed paired across budgets.\n• Fixed 2,490 SST-2 training and 872 validation examples.\n• Full train/validation evaluation after every epoch; no early stopping.\n• Only final observations compare budgets; 168 observations are correlated.",
        note="All seeds and failures are retained. Intermediate epochs are learning curves, not extra replications.",
    )
    slide(
        "Fixed learning rates and training controls",
        "\n".join(
            f"• {config.MODELS[m].label}: adapter LR {control['learning_rates'][m]:.3g}; head ×{control['head_lr_multipliers'][m]:g}"
            for m in MODELS
        )
        + "\n\nRank 16; alpha 32; q/v adapters; AdamW; BF16 compute, FP32 trainables/moments; batch 16; length 256; weight decay/dropout 0; clipping 1.",
        note="Rates were selected by earlier project calibration; no new LR tuning occurs here.",
    )
    slide(
        "Each fit uses its own cosine schedule",
        chart=paths["budget-matched-schedules"],
        note="156 steps/epoch; ceil(3% warmup); compare duration plus its matched schedule, not duration at a fixed LR path.",
    )
    slide(
        "How do we identify loss-based overfitting?",
        "• NLL = mean −log probability of the correct label; lower is better.\n• Falling full-training NLL + rising validation NLL suggests overfitting.\n• Compare budgets 4–6 with independently completed budget 3.\n• Descriptive screen: this direction in both seeds AND mean NLL rise >0.02.\n• Accuracy and Brier score add context; a larger gap alone is insufficient.\n• Two seeds give descriptive evidence, not a significance test.",
        note="The hypothesis is tested, not assumed; results can contradict epoch 3 being best.",
    )
    slide(
        "Endpoint validation NLL",
        chart=paths["validation-overview"],
        note="Lower is better. Mean ± sample SD across two seeds; faint dotted lines = individual paired seeds.",
    )
    slide(
        "Endpoint validation accuracy",
        chart=paths["accuracy-overview"],
        note="Higher is better. Different metrics can prefer different planned budgets.",
    )
    slide(
        "How much do larger budgets differ from 3 epochs?",
        chart=paths["comparison-to-budget3"],
        note="Validation loss change alone does not prove overfitting; inspect training loss and both paired seeds.",
    )
    for model, result in summary["models"].items():
        slide(
            config.MODELS[model].label + ": endpoint budgets",
            chart=paths[model + "-budgets"],
            note=model_statement(model, result),
        )
        slide(
            config.MODELS[model].label + ": loss curves",
            chart=paths[model + "-learning-curves"],
            note="Each panel is a different fresh planned budget. Full-set train/validation NLL; shaded bands = two-seed SD.",
        )
    slide(
        "Probability quality: Brier score",
        chart=paths["brier-overview"],
        note="Lower is better. NLL can rise if remaining mistakes become more confident even while accuracy improves.",
    )
    slide(
        "Observed preferences by model",
        "\n\n".join(model_statement(m, result) for m, result in summary["models"].items()),
        note="Preferences are descriptive minima/maxima among six tested budget means; they are not universal optima.",
        size=20,
    )
    slide(
        "Conclusion and limits",
        verdict(summary)
        + "\n\n• Two seeds; one small SST-2 subset; one rank.\n• Reused validation data, not an untouched test set.\n• Duration and matched schedule vary together.\n• No conclusion about other tasks, ranks or continual retention.\n• Original rank-sweep data and three-epoch defaults are unchanged.",
        size=21,
    )
    slide(
        "Evidence retained for CPU reanalysis",
        f"• {summary['completed_runs']}/48 fits validated; {summary['epoch_observations']} epoch observations.\n• Word report, table document, this presentation, JSON/CSV, PNG/PDF.\n• Per-example full train/validation probabilities, labels and NLL.\n• Step losses, actual learning rates, gradients, timing and VRAM.\n• Frozen source, pinned revisions, data indices and initialization checksums.\n• Remote persistence and failure logs; no large model-weight exports.",
    )
    slide(
        "References",
        "\n\n".join(f"[{i}] {title}\n{url}" for i, (title, url) in enumerate(REFERENCES[:3], 1)),
        size=17,
    )
    slide(
        "References (continued)",
        "\n\n".join(f"[{i}] {title}\n{url}" for i, (title, url) in enumerate(REFERENCES[3:], 4)),
        size=18,
    )
    deck.save(root / "reports" / "epoch-study.pptx")


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def build(root):
    root = Path(root)
    out = root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    summary, records, times, control = analyze(root)
    utils.save_json(out / "summary.json", summary)
    write_csv(out / "all-epoch-observations.csv", records)
    write_csv(out / "endpoints-by-seed.csv", [r for r in records if r["is_endpoint"]])
    write_csv(out / "run-timing.csv", times)
    write_csv(
        out / "endpoint-aggregates.csv",
        [
            {
                "model": m,
                "epoch_budget": int(b),
                "seeds": ";".join(map(str, g["seeds"])),
                **{f"{k}_{stat}": g[k][stat] for k in METRICS for stat in ("mean", "sd", "n")},
            }
            for m, data in summary["models"].items()
            for b, g in data["budgets"].items()
        ],
    )
    write_csv(
        out / "paired-changes-versus-budget3.csv",
        [
            {"model": m, "epoch_budget": c["epoch_budget"], "reference": 3, **r}
            for m, data in summary["models"].items()
            for c in data["comparisons_to_budget3"]
            for r in c["paired_values"]
        ],
    )
    paths = figures(root, summary, records)
    word_report(root, summary, control, paths)
    presentation(root, summary, control, paths)
    manifest = [
        {
            "path": str(p.relative_to(root)),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "artifact-manifest.json"
    ]
    utils.save_json(
        out / "artifact-manifest.json", {"protocol_sha256": summary["protocol_sha256"], "files": manifest}
    )
    return {
        "complete": summary["complete"],
        "completed_runs": summary["completed_runs"],
        "expected_runs": summary["expected_runs"],
        "epoch_observations": summary["epoch_observations"],
        "endpoint_observations": summary["endpoint_observations"],
        "verdict": verdict(summary),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE / "results")
    args = parser.parse_args()
    print(json.dumps(build(args.root), indent=2))
