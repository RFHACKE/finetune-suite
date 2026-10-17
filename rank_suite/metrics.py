"""Behavioral metrics, seed-aware summaries, and guarded feasible-rank decisions."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from . import config

PP = 100.0
CONFIG_KEYS = ["model", "track", "rank", "lr"]


def compute_run_metrics(A, task_keys):
    T = len(task_keys)
    if T == 0 or len(A) != T or any(len(row) != T for row in A):
        raise ValueError("Accuracy matrix must match the task sequence")
    for i in range(T):
        for j in range(i + 1):
            v = A[i][j]
            if v is None or not np.isfinite(v) or not 0 <= v <= 1:
                raise ValueError("Incomplete/nonfinite accuracy matrix cannot enter analysis")
    diag = [A[i][i] for i in range(T)]
    components = {task_keys[j]: A[-1][j] - A[j][j] for j in range(T - 1)}
    bwt = float(np.mean(list(components.values()))) if components else None
    # Peak forgetting differs from -BWT: an earlier task can improve after learning
    # an intermediate task, so its diagonal is not necessarily its peak.
    peaks = {task_keys[j]: max(A[i][j] for i in range(j, T)) - A[-1][j]
             for j in range(T - 1)}
    return {"bwt": bwt, "forgetting_magnitude": -bwt if bwt is not None else None,
            "final_task_acc": A[-1][-1], "avg_diagonal": float(np.mean(diag)),
            "avg_final_row": float(np.mean(A[-1])),
            "avg_matrix": float(np.mean([A[i][j] for i in range(T) for j in range(i + 1)])),
            "diagonal_accuracies": dict(zip(task_keys, diag)),
            "transition_changes": {f"{task_keys[i - 1]}->{task_keys[i]}:{task_keys[j]}": A[i][j] - A[i - 1][j]
                                   for i in range(1, T) for j in range(i)},
            "worst_task_final_change": min(components.values()) if components else None,
            "per_task_forgetting": components, "per_task_peak_forgetting": peaks,
            "peak_forgetting": float(np.mean(list(peaks.values()))) if peaks else None}


def load_results(results_dir, modes=("main",), validate_spectra=True):
    results = []
    for path in sorted(Path(results_dir).glob("runs/*/result.json")):
        with path.open() as f:
            r = json.load(f)
        if r.get("status") != "completed" or r.get("mode") not in modes:
            continue
        if r.get("run_identity", {}).get("protocol") != config.PROTOCOL_VERSION:
            continue  # v1 results have different metrics/precision; never pool them.
        r["metrics"] = compute_run_metrics(r["A"], r["task_order"])
        r["_dir"] = str(path.parent)
        results.append(r)
    fingerprints = {r["protocol_sha256"] for r in results}
    if len(fingerprints) > 1:
        raise ValueError("Results contain multiple protocol fingerprints")
    signatures = {json.dumps({"solver": payload.get("solver"),
                              "version": payload.get("analysis_version"),
                              "stack": payload.get("numerical_stack")}, sort_keys=True)
                  for r in results for payload in r.get("intruder", {}).values()}
    if validate_spectra and len(signatures) > 1:
        raise ValueError("Mixed spectral solvers/numerical stacks: finish one complete intruders pass before analysis")
    return results


def results_dataframe(results):
    import pandas as pd
    rows = []
    for r in results:
        m = r["metrics"]
        boundary = f"after_{r['task_order'][-1]}"
        intr = r.get("intruder", {}).get(boundary, {})
        total = intr.get("total", {})
        primary = total.get("0.9")
        denom = intr.get("n_vectors", intr.get("n_matrices", 0) * intr.get("k", config.INTRUDER_K))
        row = {k: r.get(k) for k in ("run_id", "model", "hf_id", "track", "rank", "alpha", "lr", "seed",
                                    "status", "trainable_params", "train_time_s", "peak_vram_gb", "d_model", "wv_out",
                                    "head_lr_multiplier", "declared_lora_lr", "declared_full_lr")}
        row.update({k: m.get(k) for k in ("bwt", "forgetting_magnitude", "final_task_acc", "avg_final_row",
                                          "avg_diagonal", "avg_matrix", "peak_forgetting", "worst_task_final_change")})
        row.update({"intruder_total_final": primary, "intruder_final_boundary": boundary,
                    "intruder_fraction_final": primary / denom if primary is not None and denom else None,
                    "intruder_status": r.get("intruder_status", "pending_cpu"),
                    "protocol_sha256": r["protocol_sha256"]})
        for e in ("0.6", "0.7", "0.8", "0.9"):
            row[f"intruder_eps_{e}"] = total.get(e)
        for i, ti in enumerate(r["task_order"]):
            row[f"diagonal_{ti}"] = r["A"][i][i]
            row[f"majority_{ti}"] = (r.get("data", {}).get(ti) or {}).get("majority_eval_accuracy")
            for j, tj in enumerate(r["task_order"][:i + 1]):
                row[f"A_{ti}_{tj}"] = r["A"][i][j]
                if j < i:
                    row[f"transition_{ti}_{tj}"] = r["A"][i][j] - r["A"][i - 1][j]
        for key, value in m.get("per_task_forgetting", {}).items():
            row[f"forget_{key}"] = value
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and df.duplicated(CONFIG_KEYS + ["seed"]).any():
        raise ValueError("Duplicate configuration/seed results; refusing to count a seed twice")
    return df


def config_summary(df):
    if df.empty:
        return df
    cols = [c for c in ("bwt", "final_task_acc", "avg_final_row", "avg_diagonal", "avg_matrix",
                        "peak_forgetting", "intruder_total_final", "intruder_fraction_final") if c in df]
    cols += [c for c in df if c.startswith(("A_", "diagonal_", "transition_", "forget_"))]
    out = df.groupby(CONFIG_KEYS, dropna=False).agg({c: ["mean", "std", "count"] for c in cols})
    out.columns = ["_".join(c) for c in out.columns]
    return out.reset_index()


def _intervals(ranks, passing):
    intervals, current = [], []
    for r in ranks:
        if r in passing:
            current.append(r)
        elif current:
            intervals.append(current)
            current = []
    if current:
        intervals.append(current)
    return intervals


def _five_seeds(df):
    return len(df) == len(config.DEFAULT_SEEDS) and set(df["seed"]) == set(config.DEFAULT_SEEDS)


def declared_lr(runs_df, model, track):
    """Read the frozen calibration anchor, never choose a baseline from outcomes."""
    name = f"declared_{track}_lr"
    if name in runs_df:
        values = runs_df.loc[runs_df["model"] == model, name].dropna().unique()
        if len(values) > 1:
            raise ValueError("Conflicting declared LR anchors in one model cohort")
        if len(values):
            return float(values[0])
    return getattr(config.MODELS[model], f"default_{track}_lr")


def feasible_windows(cfg_df, runs_df):
    """Only certify the full five-rank, five-seed design with a fixed baseline.

    Retain the methodology's max(2pp, 2*LoRA sample SD) definition and also
    report a fixed-2pp sensitivity result, because a variance-dependent
    margin rewards noisy configurations. No outcome-selected baseline LR.
    """
    if cfg_df.empty:
        return []
    windows = []
    lora = cfg_df[cfg_df["track"] == "lora"]
    for model in sorted(lora["model"].unique()):
        m = config.MODELS[model]
        full_lr = declared_lr(runs_df, model, "full")
        full = runs_df[(runs_df["model"] == model) & (runs_df["track"] == "full")
                       & np.isclose(runs_df["lr"], full_lr, rtol=1e-10, atol=0)]
        full_ok = _five_seeds(full)
        full_ref = float(full["final_task_acc"].mean()) if full_ok else None
        for lr in sorted(lora[lora["model"] == model]["lr"].unique()):
            sub = lora[(lora["model"] == model) & (lora["lr"] == lr)].sort_values("rank")
            details, passing, fixed = [], [], []
            available = set()
            for _, row in sub.iterrows():
                rank = int(row["rank"])
                individual = runs_df[(runs_df["model"] == model) & (runs_df["track"] == "lora")
                                     & (runs_df["lr"] == lr) & (runs_df["rank"] == rank)]
                complete = _five_seeds(individual)
                if complete:
                    available.add(rank)
                a33, bwt = float(row["final_task_acc_mean"]), float(row["bwt_mean"])
                sa = float(row["final_task_acc_std"]) if complete else None
                sb = float(row["bwt_std"]) if complete else None
                margin_acc = max(config.WINDOW_MARGIN_PP, 2 * sa * PP) if complete else None
                margin_bwt = max(config.WINDOW_MARGIN_PP, 2 * sb * PP) if complete else None
                plastic = bool(complete and full_ok and (a33 - full_ref) * PP >= -margin_acc)
                stable = bool(complete and bwt * PP >= -margin_bwt)
                ok = plastic and stable
                fixed_ok = bool(complete and full_ok and (a33 - full_ref) * PP >= -2 and bwt * PP >= -2)
                if ok:
                    passing.append(rank)
                if fixed_ok:
                    fixed.append(rank)
                details.append({"rank": rank, "seed_complete": complete,
                                "a33_mean": a33, "a33_std_pp": sa * PP if sa is not None else None,
                                "bwt_mean": bwt, "bwt_std_pp": sb * PP if sb is not None else None,
                                "margin_acc_pp": margin_acc, "margin_bwt_pp": margin_bwt,
                                "plasticity_ok": plastic, "stability_ok": stable,
                                "in_window": ok, "fixed_2pp_ok": fixed_ok})
            certified = full_ok and set(config.METHODOLOGY_RANKS).issubset(available)
            # Restrict certified decisions to the preregistered grid. Other ranks
            # can be explored but cannot silently redefine the main window.
            passing = [r for r in config.METHODOLOGY_RANKS if r in passing] if certified else []
            intervals = _intervals(config.METHODOLOGY_RANKS, passing)
            windows.append({"model": model, "lr": lr,
                            "status": "complete" if certified else "incomplete",
                            "full_ft_reference_a33": full_ref, "full_ft_reference_lr": full_lr,
                            "full_ft_seeds_complete": full_ok,
                            "window_ranks": passing, "window_intervals": intervals,
                            "window_ranks_fixed_2pp": [r for r in config.METHODOLOGY_RANKS if r in fixed] if certified else [],
                            "window_r_over_d": [r / m.d_model for r in passing],
                            "window_r_over_wv": [r / m.wv_out for r in passing],
                            "n_ranks_evaluated": len(sub), "details": details})
    return windows


def paired_rank_comparisons(runs_df):
    """Seed-paired descriptive differences with Student-t intervals; no p-value hunt."""
    from scipy.stats import t
    rows = []
    lora = runs_df[runs_df["track"] == "lora"]
    for (model, lr), group in lora.groupby(["model", "lr"]):
        for ra, rb in itertools.combinations(sorted(group["rank"].unique()), 2):
            a = group[group["rank"] == ra].set_index("seed")
            b = group[group["rank"] == rb].set_index("seed")
            shared = sorted(set(a.index) & set(b.index))
            for metric in ("bwt", "final_task_acc", "avg_diagonal"):
                diff = (b.loc[shared, metric] - a.loc[shared, metric]).to_numpy(dtype=float) * PP
                n = len(diff)
                if not n:
                    continue
                mean = float(diff.mean())
                half = float(t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)) if n > 1 else None
                rows.append({"model": model, "lr": lr, "rank_a": int(ra), "rank_b": int(rb),
                             "metric": metric, "n_pairs": n, "seeds": shared,
                             "delta_b_minus_a_pp": mean,
                             "ci95_low_pp": mean - half if half is not None else None,
                             "ci95_high_pp": mean + half if half is not None else None,
                             "complete": set(shared) == set(config.DEFAULT_SEEDS),
                             "interpretation": "exploratory; t interval assumes independent approximately normal seed differences"})
    return rows


def intruder_correlations(runs_df):
    from scipy.stats import spearmanr
    if runs_df.empty:
        return []
    out = []
    lora = runs_df[(runs_df["track"] == "lora") & runs_df["intruder_total_final"].notna()
                   & runs_df["bwt"].notna()]
    def add(sub, label, x="intruder_total_final"):
        if len(sub) < 3 or sub[x].nunique() < 2 or sub["bwt"].nunique() < 2:
            return
        rho, p = spearmanr(sub[x], -sub["bwt"])
        out.append({"scope": label, "n_runs": len(sub), "spearman_rho": float(rho),
                    "p_value": float(p), "x": x,
                    "interpretation": "exploratory; repeated ranks/seeds are not independent causal evidence"})
    if not lora.empty:
        primary = lora[lora.apply(lambda row: np.isclose(row["lr"], declared_lr(runs_df, row["model"], "lora"),
                                                        rtol=1e-10, atol=0), axis=1)]
        add(primary.dropna(subset=["intruder_fraction_final"]), "pooled_normalized_exploratory", "intruder_fraction_final")
        for (model, lr), group in lora.groupby(["model", "lr"]):
            label = (model if np.isclose(lr, declared_lr(runs_df, model, "lora"), rtol=1e-10, atol=0)
                     else f"{model}@lr={lr:g}")
            add(group, label)
            means = group.groupby("rank", as_index=False)[["intruder_total_final", "bwt"]].mean()
            add(means, label + "_rank_means")
    full = runs_df[(runs_df["track"] == "full") & runs_df["intruder_total_final"].notna()]
    if not full.empty:
        out.append({"scope": "full_ft_comparison", "n_runs": len(full),
                    "mean_intruders_eps09": float(full["intruder_total_final"].mean()),
                    "max_intruders_eps09": int(full["intruder_total_final"].max())})
    return out


def all_task_feasible_sets(runs_df, margins=(1.0, 2.0, 5.0)):
    """Supplementary all-task gates; the original window remains a separate output.

    Simultaneous t intervals cover 25 constraints within each model/LR family:
    five prescribed ranks times three learning and two retention comparisons.
    The normal independent-seed-difference assumption is explicit; five seeds
    do not establish generalization over datasets or task orders.
    """
    from scipy.stats import t
    if runs_df.empty:
        return []
    output = []
    tasks = config.TASK_KEYS
    for model in sorted(runs_df.loc[runs_df.track == 'lora', 'model'].unique()):
        baseline_lr = declared_lr(runs_df, model, 'full')
        full = runs_df[(runs_df.model == model) & (runs_df.track == 'full')
                       & np.isclose(runs_df.lr, baseline_lr, rtol=1e-10, atol=0)]
        baseline_complete = _five_seeds(full)
        needed = [f'A_{task}_{task}' for task in tasks]
        needed += [f'A_{tasks[-1]}_{task}' for task in tasks[:-1]]
        if any(col not in runs_df for col in needed):
            continue
        health = {}
        for task in tasks:
            majority = f'majority_{task}'
            known = baseline_complete and majority in full and full[majority].notna().all()
            health[task] = (bool(full[f'A_{task}_{task}'].mean() > full[majority].mean())
                            if known else None)
        healthy = all(v is True for v in health.values())
        lora = runs_df[(runs_df.model == model) & (runs_df.track == 'lora')]
        for lr, group in lora.groupby('lr'):
            details, complete_ranks = [], set()
            for rank in config.METHODOLOGY_RANKS:
                part = group[group['rank'] == rank]
                complete = _five_seeds(part)
                if complete:
                    complete_ranks.add(rank)
                comparisons = []
                if complete and baseline_complete:
                    a, b = part.set_index('seed'), full.set_index('seed')
                    seeds = list(config.DEFAULT_SEEDS)
                    for task in tasks:
                        col = f'A_{task}_{task}'
                        comparisons.append(('learning', task, (a.loc[seeds, col] - b.loc[seeds, col]).to_numpy()))
                    for task in tasks[:-1]:
                        comparisons.append(('retention', task,
                            (a.loc[seeds, f'A_{tasks[-1]}_{task}'] - a.loc[seeds, f'A_{task}_{task}']).to_numpy()))
                constraints = []
                for kind, task, differences in comparisons:
                    mean = float(np.mean(differences) * PP)
                    half = float(t.ppf(1 - .05 / (2 * 25), 4) * np.std(differences, ddof=1) / np.sqrt(5) * PP)
                    constraints.append({'kind': kind, 'task': task, 'mean_change_pp': mean,
                                        'simultaneous_ci_low_pp': mean - half,
                                        'simultaneous_ci_high_pp': mean + half})
                details.append({'rank': rank, 'seed_complete': complete, 'constraints': constraints})
            certified = baseline_complete and len(complete_ranks) == len(config.METHODOLOGY_RANKS) and healthy
            for margin in margins:
                passing, supported, rejected = [], [], []
                rank_details = []
                for item in details:
                    constraints = item['constraints']
                    point = len(constraints) == 5 and all(c['mean_change_pp'] >= -margin - 1e-10 for c in constraints)
                    support = point and all(c['simultaneous_ci_low_pp'] >= -margin - 1e-10 for c in constraints)
                    reject = any(c['simultaneous_ci_high_pp'] < -margin - 1e-10 for c in constraints)
                    if certified and point:
                        passing.append(item['rank'])
                    if certified and support:
                        supported.append(item['rank'])
                    if certified and reject:
                        rejected.append(item['rank'])
                    rank_details.append({**item, 'point_pass': point,
                        'interval_decision': 'supported' if support else 'fails' if reject else 'inconclusive'})
                output.append({'model': model, 'lr': float(lr), 'tolerance_pp': margin,
                    'status': 'complete' if certified else 'baseline_underlearning' if False in health.values() else 'incomplete',
                    'baseline_lr': baseline_lr, 'baseline_seeds_complete': baseline_complete,
                    'baseline_learning_above_majority': health, 'window_ranks': passing,
                    'window_intervals': _intervals(config.METHODOLOGY_RANKS, passing),
                    'interval_supported_ranks': supported, 'interval_rejected_ranks': rejected,
                    'details': rank_details, 'interval_family': '25 constraints within model/LR; Bonferroni 95% t intervals',
                    'interpretation': 'supplementary operational gates; intervals assume independent normal paired seed differences'})
    return output
