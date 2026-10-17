"""Analysis, charts and report generation (methodology Sec. 12 deliverables).

Reads every run's result.json, aggregates across seeds, and writes:
  results/aggregates/*.csv     - flat tables (runs, configs, windows, correlations)
  results/analysis/*.png       - all figures
  results/analysis/report.md   - human-readable report with tables + figures

The figures cover the methodology's expected outputs: rank-forgetting curves
(BWT vs r with seed bands), final-task accuracy vs r, the feasible rank
window (raw r, r/d_model, r/Wv), window drift across the four models,
intruder-dimension analysis with epsilon sensitivity and the Spearman
correlation against BWT, the LoRA vs full-FT comparison, per-task accuracy
heatmaps over every rank x learning-rate configuration, seed variance, and
training-loss curves.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import config, figures as figx, metrics

logger = logging.getLogger("rank_suite.analysis")

PP = 100.0


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130,
        "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
        "figure.autolayout": False,
    })
    return plt


def _save(fig, out_dir: Path, name: str) -> str:
    name = name.replace("/", "--")   # HF ids contain '/'; keep filenames flat
    path = Path(out_dir) / name
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001 - decoration overflow is not fatal
            pass
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return name


def _fmt_pp(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * PP:.2f}"


def _fmt_pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * PP:.2f}%"


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def run_analysis(results_dir: Path, mode: str = "main") -> dict:
    results_dir = Path(results_dir)
    ana_dir = results_dir / "analysis"
    agg_dir = results_dir / "aggregates"
    ana_dir.mkdir(parents=True, exist_ok=True)
    agg_dir.mkdir(parents=True, exist_ok=True)

    results = metrics.load_results(results_dir, modes=(mode,))
    sanity = metrics.load_results(results_dir / "sanity", modes=("sanity",))
    from .utils import load_json, save_json
    latest = load_json(results_dir / "lr_test" / "latest.json", {})
    calibration_report = None
    if latest:
        from .calibration_plots import write_report
        report_path = results_dir / "lr_test" / latest["report"]
        calibration_report = load_json(report_path)
        if calibration_report:
            calibration_report["charts"] = write_report(calibration_report, report_path.parent, results_dir / "lr_test")
            save_json(report_path, calibration_report)
    if not results:
        logger.warning("No completed runs found under %s", results_dir / "runs")
        return {"runs": 0, "lr_calibration": latest or None}

    df = metrics.results_dataframe(results)
    cfg = metrics.config_summary(df)
    windows = metrics.feasible_windows(cfg, df)
    all_task_sets = metrics.all_task_feasible_sets(df)
    corrs = metrics.intruder_correlations(df)

    _write_table(df, agg_dir / "runs")
    _write_table(cfg, agg_dir / "config_summary")
    _write_long_tables(results, agg_dir)
    save_json(agg_dir / "all_task_feasible_sets.json", all_task_sets)
    _write_table(pd_rows([{k: v for k, v in item.items() if k != "details"}
                         for item in all_task_sets]), agg_dir / "all_task_feasible_sets_summary")
    _write_windows(windows, agg_dir / "feasible_windows")
    _write_corrs(corrs, agg_dir / "intruder_correlations")
    _write_table(pd_rows(metrics.paired_rank_comparisons(df)), agg_dir / "paired_rank_comparisons")
    from .audit import completeness_audit
    policy = load_json(results_dir / "protocol.json", {}).get("learning_rate_defaults")
    audit = completeness_audit(results, mode, policy)
    import json
    failure_attempts = []
    for path in results_dir.glob("failures*.jsonl"):
        for line in path.read_text().splitlines():
            try:
                failure_attempts.append(json.loads(line)["run_id"])
            except (ValueError, KeyError):
                pass
    audit["failure_attempts"] = failure_attempts
    save_json(agg_dir / "completeness.json", audit)

    figures = _make_figures(results, df, cfg, windows, corrs, ana_dir)
    from .measurement_plots import measurement_figures
    figures += measurement_figures(df, results, ana_dir)
    guidance = _write_rank_guidance(df, windows, agg_dir)
    _write_report(results_dir, df, cfg, windows, corrs, figures, sanity,
                  guidance, audit=audit, all_task_sets=all_task_sets,
                  calibration_report=calibration_report)

    logger.info("Analysis complete: %d runs, %d figures -> %s",
                len(results), len(figures), ana_dir)
    return {"runs": len(results), "figures": figures, "completeness": audit}


# --------------------------------------------------------------------------
# CSV + JSON helpers (every table is available in both formats so any chart
# can be rebuilt later without the suite)
# --------------------------------------------------------------------------
def _write_table(df, base):
    base = Path(base)
    df.to_csv(base.with_suffix(".csv"), index=False)
    df.to_json(base.with_suffix(".json"), orient="records", indent=2)


def _write_long_tables(results, agg_dir: Path):
    rows = []
    for r in results:
        for i, ti in enumerate(r.get("task_order", [])):
            for j, tj in enumerate(r.get("task_order", [])):
                if j <= i:
                    rows.append({
                        "run_id": r["run_id"], "model": r["model"],
                        "track": r["track"], "rank": r.get("rank"),
                        "lr": r["lr"], "seed": r["seed"],
                        "after_task": ti, "eval_task": tj,
                        "accuracy": r["A"][i][j],
                    })
    _write_table(pd_rows(rows), Path(agg_dir) / "accuracy_matrix_long")

    # intruder counts per run / boundary / epsilon
    irows = []
    for r in results:
        for boundary, payload in (r.get("intruder") or {}).items():
            for eps_key, n in (payload.get("total") or {}).items():
                irows.append({
                    "run_id": r["run_id"], "model": r["model"],
                    "track": r["track"], "rank": r.get("rank"),
                    "lr": r["lr"], "seed": r["seed"],
                    "boundary": boundary, "eps": eps_key,
                    "intruder_count": n,
                    "mean_max_cosine": payload.get("mean_max_cosine"),
                })
    _write_table(pd_rows(irows), Path(agg_dir) / "intruder_counts_long")
    task_rows, transition_rows, spectral_rows = [], [], []
    import re
    for r in results:
        identity = {k: r.get(k) for k in ("run_id", "model", "track", "rank", "lr", "seed", "head_lr_multiplier")}
        for i, task in enumerate(r["task_order"]):
            task_rows.append({**identity, "task": task, "diagonal_accuracy": r["A"][i][i],
                "final_accuracy": r["A"][-1][i], "retention_change_pp": (r["A"][-1][i] - r["A"][i][i]) * 100})
            for j in range(i):
                transition_rows.append({**identity, "after_task": task, "eval_task": r["task_order"][j],
                    "change_pp": (r["A"][i][j] - r["A"][i - 1][j]) * 100})
        for boundary, payload in r.get("intruder", {}).items():
            for name, diagnostic in payload.get("diagnostics_by_matrix", {}).items():
                match = re.search(r"layers\.(\d+)", name)
                layer = int(match[1]) if match else None
                row = {**identity, "boundary": boundary, "matrix": name,
                       "matrix_type": name.rsplit(".", 1)[-1], "layer": layer,
                       "normalized_depth": layer / max(config.MODELS[r["model"]].layers - 1, 1) if layer is not None else None,
                       **diagnostic["update"]}
                for target, subspace in diagnostic["subspace"].items():
                    row[f"subspace_distance_k{target}"] = subspace["distance"]
                    row[f"subspace_k_actual_{target}"] = subspace["k_actual"]
                    row[f"max_angle_radians_k{target}"] = max(subspace["principal_angles_radians"])
                    for side in ("reference", "tuned"):
                        row[f"{side}_cutoff_gap_relative_k{target}"] = subspace[f"{side}_cutoff_gap"]["relative"]
                        row[f"{side}_cutoff_near_tie_k{target}"] = subspace[f"{side}_cutoff_gap"]["near_tie"]
                spectral_rows.append(row)
    _write_table(pd_rows(task_rows), agg_dir / "per_task_learning_retention")
    _write_table(pd_rows(transition_rows), agg_dir / "task_transition_changes")
    _write_table(pd_rows(spectral_rows), agg_dir / "spectral_diagnostics_long")


def pd_rows(rows):
    import pandas as pd

    return pd.DataFrame(rows)


def _write_windows(windows, base: Path):
    rows = []
    for w in windows:
        rows.append({
            "model": w["model"], "lr": w["lr"],
            "window_ranks": ";".join(str(r) for r in w["window_ranks"]),
            "window_r_over_d": ";".join(f"{v:.4f}" for v in w["window_r_over_d"]),
            "window_r_over_wv": ";".join(f"{v:.4f}" for v in w["window_r_over_wv"]),
            "full_ft_reference_a33": w["full_ft_reference_a33"],
            "full_ft_reference_lr": w["full_ft_reference_lr"],
            "n_ranks_evaluated": w["n_ranks_evaluated"],
            "status": w["status"],
            "window_intervals": str(w["window_intervals"]),
            "window_ranks_fixed_2pp": str(w["window_ranks_fixed_2pp"]),
        })
    _write_table(pd_rows(rows), base)
    # full-fidelity JSON (numeric lists, per-rank criteria details)
    from .utils import save_json

    save_json(base.parent / f"{base.name}_full.json", windows)


def _write_corrs(corrs, base: Path):
    _write_table(pd_rows(corrs), base)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _make_figures(results, df, cfg, windows, corrs, ana_dir: Path) -> list:
    plt = _plt()
    figures = []
    lora_runs = df[df["track"] == "lora"] if not df.empty else df
    full_runs = df[df["track"] == "full"] if not df.empty else df

    for model in sorted(df["model"].unique()):
        label = config.MODELS[model].label if model in config.MODELS else model
        model_rows = df[df["model"] == model]
        if (model_rows.groupby(["track", "rank", "lr"], dropna=False)["seed"].nunique() < 5).any():
            label += " (partial seed coverage)"
        sub_l = lora_runs[lora_runs["model"] == model] if not lora_runs.empty else lora_runs

        if not sub_l.empty:
            figures.append(_fig_bwt_vs_rank(plt, sub_l, ana_dir, model, label))
            figures.append(_fig_final_acc_vs_rank(plt, sub_l, full_runs, ana_dir, model, label))
            figures.append(_fig_learn_forget_scatter(plt, sub_l, windows, ana_dir, model, label))
            figures.append(_fig_forgetting_components(plt, sub_l, ana_dir, model, label))
            figures.append(_fig_heatmaps(plt, sub_l, ana_dir, model, label))
            figures.append(_fig_task_acc_heatmap(plt, sub_l, ana_dir, model, label))
            figures.append(_fig_seed_variance(plt, sub_l, ana_dir, model, label))
            figures.append(_fig_intruder_scatter(plt, sub_l, corrs, ana_dir, model, label))
            figures.append(_fig_intruder_eps(plt, results, sub_l, ana_dir, model, label))
            figures.append(_fig_loss_curves(plt, sub_l, results, ana_dir, model, label))
            figures.append(_fig_lr_effect(plt, sub_l, ana_dir, model, label))
            # extended set: every hyperparameter variant of the model in one chart
            figures.append(figx.fig_metric_bars(
                plt, sub_l, ana_dir, model, label, "bwt",
                "BWT per configuration", "BWT (pp)", name="bwt_bars"))
            figures.append(figx.fig_metric_bars(
                plt, sub_l, ana_dir, model, label, "final_task_acc",
                "Final-task accuracy per configuration", "A33 (%)",
                ref=figx.full_ft_ref(df, model), ref_label="full-FT A33",
                name="final_acc_bars"))
            figures.append(figx.fig_metric_bars(
                plt, sub_l, ana_dir, model, label, "avg_final_row",
                "Average accuracy (final row) per configuration", "avg acc (%)",
                name="avg_acc_bars"))
            figures.append(figx.fig_metric_bars(
                plt, sub_l, ana_dir, model, label, "forgetting_magnitude",
                "Signed forgetting (−BWT) per configuration",
                "−BWT (pp)", name="forgetting_magnitude_bars"))
            figures.append(figx.fig_task_trajectories(plt, sub_l, ana_dir, model, label))
            figures.append(figx.fig_accuracy_matrices(plt, sub_l, ana_dir, model, label))
            figures.append(figx.fig_stability_plasticity(plt, df, ana_dir, model, label, windows))
            figures.append(figx.fig_plasticity_gap(plt, df, ana_dir, model, label))
            figures.append(figx.fig_seed_spaghetti(plt, sub_l, ana_dir, model, label))
            figures.append(figx.fig_seed_boxplot(plt, sub_l, ana_dir, model, label))
            figures.append(figx.fig_intruder_vs_rank(plt, sub_l, ana_dir, model, label))
            figures.append(figx.fig_intruder_accumulation(plt, results, ana_dir, model, label))
            figures.append(figx.fig_intruder_cosine_trajectory(plt, results, ana_dir, model, label))
            figures.append(figx.fig_grad_norm_curves(plt, results, ana_dir, model, label))
            figures.append(figx.fig_radar(plt, sub_l, ana_dir, model, label))
            figures.append(figx.fig_config_table(plt, sub_l, ana_dir, model, label, windows))
            figures.append(figx.fig_dashboard(plt, df, ana_dir, model, label, windows))

    figures.append(_fig_window_drift(plt, windows, ana_dir))
    figures.append(_fig_track_comparison(plt, df, ana_dir))
    # secondary cross-model views (normalized units / anchors only)
    figures.append(figx.fig_window_drift_raw(plt, windows, ana_dir))
    figures.append(figx.fig_cross_model_normalized(plt, df, ana_dir))
    figures.append(figx.fig_full_ft_anchors(plt, df, ana_dir))
    return [f for f in figures if f]


def _group(sub, by):
    """mean/std per group for the main metrics."""
    g = sub.groupby(by, dropna=False)
    out = g.agg(
        bwt_mean=("bwt", "mean"), bwt_std=("bwt", "std"),
        a33_mean=("final_task_acc", "mean"), a33_std=("final_task_acc", "std"),
        avg_mean=("avg_final_row", "mean"), n=("seed", "count"),
    ).reset_index()
    return out.sort_values(by)


def _fig_bwt_vs_rank(plt, sub, ana_dir, model, label):
    g = _group(sub, ["lr", "rank"])
    if g.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for lr, gl in g.groupby("lr"):
        gl = gl.sort_values("rank")
        x = [str(int(r)) for r in gl["rank"]]
        y = gl["bwt_mean"] * PP
        yerr = gl["bwt_std"].fillna(0) * PP
        ax.plot(x, y, marker="o", label=f"lr={utils_fmt(lr)}")
        ax.fill_between(range(len(x)), y - yerr, y + yerr, alpha=0.15)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("BWT (percentage points)")
    ax.set_title(f"Rank-forgetting curve: {label}\nBWT vs rank (mean ± seed band)")
    ax.legend(fontsize=7)
    return _save(fig, ana_dir, f"bwt_vs_rank_{model}.png")


def _fig_final_acc_vs_rank(plt, sub, full_runs, ana_dir, model, label):
    g = _group(sub, ["lr", "rank"])
    if g.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for lr, gl in g.groupby("lr"):
        gl = gl.sort_values("rank")
        x = [str(int(r)) for r in gl["rank"]]
        y = gl["a33_mean"] * PP
        yerr = gl["a33_std"].fillna(0) * PP
        ax.plot(x, y, marker="s", label=f"lr={utils_fmt(lr)}")
        ax.fill_between(range(len(x)), y - yerr, y + yerr, alpha=0.15)
    ref = figx.full_ft_ref(full_runs, model) if not full_runs.empty else None
    if ref is not None:
        ax.axhline(ref * PP, color="red", ls="--", lw=1,
                   label=f"fixed full-FT mean A33 = {ref * PP:.2f}%")
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("Final-task accuracy (SST-2, %)")
    ax.set_title(f"Plasticity vs rank: {label}\nfinal-task accuracy (mean ± seed band)")
    ax.legend(fontsize=7)
    return _save(fig, ana_dir, f"final_acc_vs_rank_{model}.png")


def _fig_learn_forget_scatter(plt, sub, windows, ana_dir, model, label):
    if sub["bwt"].isna().all():
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    lrs = sorted(sub["lr"].unique())
    cmap = plt.get_cmap("viridis")
    for i, lr in enumerate(lrs):
        win_ranks = next((w["window_ranks"] for w in windows
                          if w["model"] == model and w["lr"] == lr), [])
        gl = sub[sub["lr"] == lr]
        x = gl["bwt"] * PP
        y = gl["final_task_acc"] * PP
        ax.scatter(x, y, color=cmap(i / max(len(lrs) - 1, 1)),
                   label=f"lr={utils_fmt(lr)}", s=40, alpha=0.85)
        for (_, row), xi, yi in zip(gl.iterrows(), x, y):
            mark = "*" if int(row["rank"]) in win_ranks else ""
            ax.annotate(f"r={int(row['rank'])}{mark}", (xi, yi),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("BWT (percentage points; left = more forgetting)")
    ax.set_ylabel("Final-task accuracy (%)")
    ax.set_title(f"Learning-forgetting trade-off: {label}\n"
                 "(★ = inside feasible rank window)")
    ax.legend(fontsize=7)
    return _save(fig, ana_dir, f"learn_forget_scatter_{model}.png")


def _fig_forgetting_components(plt, sub, ana_dir, model, label):
    comp_cols = [c for c in sub.columns if c.startswith("forget_")]
    if not comp_cols:
        return None
    g = sub.groupby(["rank", "lr"], dropna=False)[comp_cols].mean().reset_index()
    if g.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ranks = sorted(g["rank"].unique())
    lrs = sorted(g["lr"].unique())
    width = 0.8 / (len(comp_cols) * max(len(lrs), 1))
    x = np.arange(len(ranks))
    for li, lr in enumerate(lrs):
        gl = g[g["lr"] == lr].set_index("rank")
        for ci, col in enumerate(comp_cols):
            vals = [gl.loc[r, col] * PP if r in gl.index else 0 for r in ranks]
            off = (li * len(comp_cols) + ci) * width - 0.4 + width / 2
            ax.bar(x + off, vals, width,
                   label=f"{col.replace('forget_', '')} @ lr={utils_fmt(lr)}")
    ax.set_xticks(x, [str(int(r)) for r in ranks])
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("Accuracy change (pp)")
    ax.set_title(f"Per-task forgetting components: {label}\n"
                 "A_{T,j} − A_{j,j} per earlier task (0 = no forgetting)")
    ax.legend(fontsize=6, ncol=2)
    return _save(fig, ana_dir, f"forgetting_components_{model}.png")


def _fig_heatmaps(plt, sub, ana_dir, model, label):
    metrics_map = [("bwt", "BWT (pp)", PP), ("final_task_acc", "Final-task acc (%)", PP),
                   ("avg_final_row", "Avg accuracy, final row (%)", PP)]
    ranks = sorted(sub["rank"].dropna().unique())
    lrs = sorted(sub["lr"].dropna().unique())
    if not ranks or not lrs:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, (col, title, scale) in zip(axes, metrics_map):
        pivot = sub.pivot_table(index="rank", columns="lr", values=col, aggfunc="mean")
        pivot = pivot.reindex(index=ranks, columns=lrs)
        mat = pivot.values * scale
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm")
        ax.set_xticks(range(len(lrs)), [utils_fmt(rate) for rate in lrs],
                      rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(ranks)), [str(int(r)) for r in ranks], fontsize=7)
        ax.set_xlabel("learning rate")
        ax.set_ylabel("rank r")
        ax.set_title(title, fontsize=9)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                            fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Rank × learning-rate performance: {label}", fontsize=11)
    return _save(fig, ana_dir, f"heatmaps_{model}.png")


def _fig_task_acc_heatmap(plt, sub, ana_dir, model, label):
    acc_cols = [c for c in sub.columns if c.startswith("A_") and c.count("_") == 2]
    # final row only: cells A_<last>_<task>
    tasks = list(config.TASK_KEYS)
    final = f"A_{tasks[-1]}_"
    final_cols = [c for c in acc_cols if c.startswith(final)]
    if not final_cols:
        return None
    g = sub.groupby(["rank", "lr"], dropna=False)[final_cols].mean().reset_index()
    g = g.sort_values(["lr", "rank"])
    mat = g[final_cols].values * PP
    fig, ax = plt.subplots(figsize=(6.8, 0.5 * len(g) + 2.2))
    im = ax.imshow(mat, aspect="auto", cmap="Greens")
    ax.set_xticks(range(len(final_cols)),
                  [c.replace(final, "").upper() for c in final_cols])
    ax.set_yticks(range(len(g)),
                  [f"r={int(r)}, lr={utils_fmt(rate)}" for r, rate in zip(g["rank"], g["lr"])],
                  fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"Final-row task accuracies after the full sequence: {label}")
    return _save(fig, ana_dir, f"task_acc_heatmap_{model}.png")


def _fig_seed_variance(plt, sub, ana_dir, model, label):
    g = _group(sub, ["lr", "rank"])
    if g.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for ax, col, title in ((axes[0], "bwt_std", "BWT across-seed std (pp)"),
                           (axes[1], "a33_std", "Final-acc across-seed std (pp)")):
        lrs = sorted(g["lr"].unique())
        ranks = sorted(g["rank"].unique())
        x = np.arange(len(ranks))
        width = 0.8 / max(len(lrs), 1)
        for li, lr in enumerate(lrs):
            gl = g[g["lr"] == lr].set_index("rank")
            vals = [(gl.loc[r, col] * PP if r in gl.index and not np.isnan(gl.loc[r, col])
                     else 0.0) for r in ranks]
            ax.bar(x + li * width - 0.4 + width / 2, vals, width,
                   label=f"lr={utils_fmt(lr)}")
        ax.set_xticks(x, [str(int(r)) for r in ranks])
        ax.set_xlabel("LoRA rank r")
        ax.set_ylabel(title)
        ax.legend(fontsize=7)
    fig.suptitle(f"Seed sensitivity per configuration: {label}", fontsize=11)
    return _save(fig, ana_dir, f"seed_variance_{model}.png")


def _fig_intruder_scatter(plt, sub, corrs, ana_dir, model, label):
    s = sub.dropna(subset=["intruder_total_final", "bwt"])
    if s.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    cmap = plt.get_cmap("plasma")
    ranks = sorted(s["rank"].dropna().unique())
    for i, r in enumerate(ranks):
        gl = s[s["rank"] == r]
        ax.scatter(gl["intruder_total_final"], -gl["bwt"] * PP,
                   color=cmap(i / max(len(ranks) - 1, 1)), s=42,
                   label=f"r={int(r)}")
    rho_txt = ""
    for c in corrs:
        if c["scope"] == model and "spearman_rho" in c:
            rho_txt = (f"Spearman ρ = {c['spearman_rho']:.3f} "
                       f"(p = {c['p_value']:.2e}, n = {c['n_runs']})")
    ax.set_xlabel(f"Intruder dimensions (k=10, ε={config.INTRUDER_EPS_PRIMARY}, final boundary)")
    ax.set_ylabel("Forgetting (−BWT, pp)")
    ax.set_title(f"Intruder dimensions vs forgetting: {label}\n{rho_txt}")
    ax.legend(fontsize=7)
    return _save(fig, ana_dir, f"intruder_vs_bwt_{model}.png")


def _fig_intruder_eps(plt, results, sub, ana_dir, model, label):
    # per-epsilon intruder totals at the final boundary, from raw results
    rows = []
    for r in results:
        if r.get("model") != model or r.get("track") != "lora":
            continue
        intr = r.get("intruder") or {}
        if not intr:
            continue
        payload = intr[list(intr.keys())[-1]]
        for eps_key, n in (payload.get("total") or {}).items():
            rows.append({"rank": r.get("rank"), "lr": r["lr"], "eps": float(eps_key),
                         "intruder_count": n})
    if not rows:
        return None
    import pandas as pd

    data = pd.DataFrame(rows)
    g = data.groupby(["rank", "lr", "eps"], dropna=False)["intruder_count"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for (r, lr), gl in g.groupby(["rank", "lr"]):
        gl = gl.sort_values("eps")
        ax.plot(gl["eps"], gl["intruder_count"], marker="o", label=f"r={int(r)}, lr={utils_fmt(lr)}")
    ax.set_xlabel("cosine threshold ε")
    ax.set_ylabel("mean intruder count (all matrices)")
    ax.set_title(f"ε-sensitivity of intruder counts: {label}")
    ax.legend(fontsize=7)
    return _save(fig, ana_dir, f"intruder_eps_sensitivity_{model}.png")


def _fig_loss_curves(plt, sub, results, ana_dir, model, label):
    by_key = {(r["model"], r["run_id"]): r for r in results}
    fig, axes = plt.subplots(1, len(config.TASK_KEYS),
                             figsize=(4.0 * len(config.TASK_KEYS), 3.8),
                             sharey=False)
    any_plotted = False
    for ax, task in zip(np.atleast_1d(axes), config.TASK_KEYS):
        for _, row in sub.iterrows():
            r = by_key.get((model, row["run_id"]))
            if not r:
                continue
            ts = r.get("train_summary", {}).get(task)
            if not ts:
                continue
            losses = ts.get("epoch_losses") or []
            if not losses:
                continue
            ax.plot(range(1, len(losses) + 1), losses, marker=".",
                    label=f"r={int(row['rank'])}, lr={utils_fmt(row['lr'])}",
                    alpha=0.8)
            any_plotted = True
        ax.set_xlabel("epoch")
        ax.set_ylabel("mean train loss")
        ax.set_title(f"task {task}")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6)
    if not any_plotted:
        plt.close(fig)
        return None
    fig.suptitle(f"Training loss per task (per-seed runs): {label}", fontsize=11)
    return _save(fig, ana_dir, f"loss_curves_{model}.png")


def _fig_lr_effect(plt, sub, ana_dir, model, label):
    g = _group(sub, ["lr", "rank"])
    if g["lr"].nunique() < 2:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for r, gl in g.groupby("rank"):
        gl = gl.sort_values("lr")
        ax.plot(gl["lr"], gl["bwt_mean"] * PP, marker="o", label=f"r={int(r)}")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("BWT (pp)")
    ax.set_title(f"Learning-rate effect on forgetting: {label}")
    ax.legend(fontsize=7)
    return _save(fig, ana_dir, f"lr_effect_{model}.png")


def _fig_window_drift(plt, windows, ana_dir):
    usable = [w for w in windows if w["window_r_over_d"]]
    if not usable:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 1.2 * len(usable) + 1.6))
    for i, w in enumerate(usable):
        xs = w["window_r_over_d"]
        ax.scatter(xs, [i] * len(xs), marker="s", s=100)
        for x, r in zip(xs, w["window_ranks"]):
            ax.text(x, i, f" r={r}", va="center", fontsize=7)
    ax.set_yticks(range(len(usable)),
                  [f"{w['model']} @ lr={utils_fmt(w['lr'])}" for w in usable],
                  fontsize=8)
    ax.set_xlabel("rank / d_model")
    ax.set_title("Feasible rank window per model (raw r / d_model units)")
    return _save(fig, ana_dir, "window_drift.png")


def _fig_track_comparison(plt, df, ana_dir):
    if df.empty:
        return None
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(len(models), 2, figsize=(11, 3.2 * len(models)), squeeze=False)
    for i, model in enumerate(models):
        m = config.MODELS[model]
        for ax, col, title in ((axes[i, 0], "bwt", "BWT (pp)"),
                               (axes[i, 1], "final_task_acc", "Final-task accuracy (%)")):
            means, errors, labels = [], [], []
            for rank in [*config.METHODOLOGY_RANKS, None]:
                track = "full" if rank is None else "lora"
                lr = metrics.declared_lr(df, model, track)
                rows = df[(df["model"] == model) & (df["track"] == track)
                          & np.isclose(df["lr"], lr, rtol=1e-10, atol=0)]
                if rank is not None:
                    rows = rows[rows["rank"] == rank]
                values = rows[col].dropna()
                means.append(float(values.mean()) * PP if len(values) else np.nan)
                errors.append(float(values.std(ddof=1)) * PP if len(values) > 1 else 0)
                labels.append((f"r={rank}" if rank is not None else "full FT") + f"\nn={len(values)}")
            ax.bar(range(6), means, yerr=errors, capsize=3,
                   color=["tab:blue"] * 5 + ["tab:red"])
            ax.set_xticks(range(6), labels, fontsize=7)
            ax.set_ylabel(title)
            ax.set_title(m.label)
    fig.suptitle("Rank-specific LoRA vs non-embedding full FT: nominal LRs\n"
                 "mean ± sample SD; n<2 has no SD estimate", fontsize=11)
    return _save(fig, ana_dir, "trackA_vs_trackB.png")


def utils_fmt(lr) -> str:
    from .utils import fmt_lr

    return fmt_lr(float(lr))


def _write_rank_guidance(runs_df, windows, agg_dir):
    """RQ4 deliverable: per model, which (rank, LR) configurations are
    feasible and which are recommended (best plasticity / most stable)."""
    rows, summary = [], {}
    if runs_df.empty:
        return summary
    win_map = {(w["model"], float(w["lr"])): set(w["window_ranks"])
               for w in windows}
    lora = runs_df[runs_df["track"] == "lora"]
    for model in sorted(lora["model"].unique()):
        sub_m = lora[lora["model"] == model]
        model_rows = []
        for (rank, lr), g in sub_m.groupby(["rank", "lr"]):
            in_win = int(rank) in win_map.get((model, float(lr)), set())
            row = {
                "model": model, "lr": float(lr), "rank": int(rank),
                "n_seeds": int(len(g)),
                "a33_mean_pct": float(g["final_task_acc"].mean()) * PP,
                "bwt_mean_pp": float(g["bwt"].mean()) * PP,
                "forgetting_magnitude_pp": float(-g["bwt"].mean()) * PP,
                "intruders_eps09": (
                    float(g["intruder_total_final"].mean())
                    if g["intruder_total_final"].notna().any() else None),
                "in_feasible_window": bool(in_win),
                "recommendation": "",
            }
            rows.append(row)
            model_rows.append(row)

        certified_lrs = {w["lr"] for w in windows
                         if w["model"] == model and w["status"] == "complete"}
        if not certified_lrs:
            continue
        eligible = [r for r in model_rows if r["lr"] in certified_lrs]
        feasible = [r for r in eligible if r["in_feasible_window"]] or eligible
        best_plast = max(feasible, key=lambda r: r["a33_mean_pct"])
        best_stab = max(feasible, key=lambda r: r["bwt_mean_pp"])
        best_plast["recommendation"] = "best plasticity"
        best_stab["recommendation"] = (
            (best_stab["recommendation"] + " | " if best_stab["recommendation"] else "")
            + "most stable")
        summary[model] = {
            "window_ranks_by_lr": {
                str(lr): sorted(rs)
                for (m, lr), rs in win_map.items() if m == model
            },
            "best_plasticity": best_plast,
            "most_stable": best_stab,
        }

    if rows:
        agg_dir = Path(agg_dir)
        agg_dir.mkdir(parents=True, exist_ok=True)
        import pandas as pd

        pd.DataFrame(rows).to_csv(agg_dir / "rank_guidance.csv", index=False)
        from .utils import save_json

        save_json(agg_dir / "rank_guidance.json", rows)
    return summary


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def _write_report(results_dir, df, cfg, windows, corrs, figures, sanity,
                  guidance=None, audit=None, all_task_sets=None, calibration_report=None):
    import pandas as pd

    lines = []
    add = lines.append
    add("# Feasible Rank Window — Experiment Report")
    add("")
    if audit:
        add(f"**Study status: {audit['status']}.** "
            f"Missing main configurations/seeds: {len(audit['missing'])}; "
            f"pending spectral analyses: {audit['pending_spectra']}.")
        add("Partial results are descriptive; a feasible window requires all five ranks, "
            "all five prescribed seeds and the fixed full-FT baseline. Smoke/sanity runs "
            "never contribute to main-study conclusions.")
        add("")
    add(f"Results directory: `{results_dir}`  ")
    if calibration_report:
        add(f"LR calibration: **{calibration_report['status']}**; active defaults: "
            f"**{calibration_report['activated']}**. See `lr_test/latest.json` and its linked JSON/Markdown report.")
        add("Calibration-only runs never enter the main accuracy/rank analysis. Baseline references use the declared, frozen rates.")
        add("")
    add(f"Completed runs: **{len(df)}** "
        f"(LoRA: {int((df['track'] == 'lora').sum())}, "
        f"full FT: {int((df['track'] == 'full').sum())})")
    add("")

    add("## 1. Per-configuration results (mean ± std over seeds)")
    add("")
    for model in sorted(df["model"].unique()):
        label = config.MODELS[model].label if model in config.MODELS else model
        add(f"### {label}")
        add("")
        add("| track | rank | alpha | lr | seeds | BWT (pp) | final acc (%) | "
            "avg acc final row (%) | intruders (ε=0.9) |")
        add("|---|---|---|---|---|---|---|---|---|")
        sub = cfg[cfg["model"] == model]
        for _, r in sub.iterrows():
            rank = int(r["rank"]) if pd.notna(r["rank"]) else "—"
            alpha = (2 * int(r["rank"])) if pd.notna(r["rank"]) else "—"
            bwt = (_mean_sd(r.get("bwt_mean"), r.get("bwt_std"))
                   if pd.notna(r.get("bwt_mean")) else "n/a")
            acc = (_mean_sd(r.get("final_task_acc_mean"), r.get("final_task_acc_std"))
                   if pd.notna(r.get("final_task_acc_mean")) else "n/a")
            avg = (_fmt_pct(r.get("avg_final_row_mean")))
            intr = (f"{r['intruder_total_final_mean']:.1f}"
                    if pd.notna(r.get("intruder_total_final_mean")) else "n/a")
            add(f"| {r['track']} | {rank} | {alpha} | {utils_fmt(r['lr'])} | "
                f"{int(r.get('final_task_acc_count', 0))} | {bwt} | {acc} | {avg} | {intr} |")
        add("")

    add("## 2. Feasible rank window (methodology Sec. 6.4)")
    add("")
    add("Criteria: plasticity `A33 >= A33_full-FT − max(2, 2σ)` pp and "
        "stability `BWT >= −max(2, 2σ)` pp. σ is the LoRA configuration's "
        "sample SD across the five seeds. All contiguous intervals on the tested "
        "grid are retained; no interpolation to untested ranks is implied. "
        "The fixed-2pp sensitivity result is in feasible_windows_full.json.")
    add("")
    add("| model | lr | window (raw r) | r/d_model | r/Wv-width | full-FT A33 ref |")
    add("|---|---|---|---|---|---|")
    for w in windows:
        win = ("; ".join(", ".join(map(str, interval)) for interval in w["window_intervals"])
               if w["status"] == "complete" else "incomplete") or "none observed"
        rd = ", ".join(f"{v:.3f}" for v in w["window_r_over_d"]) or "—"
        rw = ", ".join(f"{v:.3f}" for v in w["window_r_over_wv"]) or "—"
        ref = (_fmt_pct(w["full_ft_reference_a33"])
               if w["full_ft_reference_a33"] is not None
               else "n/a (needs Track B)")
        add(f"| {w['model']} | {utils_fmt(w['lr'])} | {win} | {rd} | {rw} | {ref} |")
    add("")

    add("### Supplementary all-task learning and retention gates")
    add("")
    add("Every task diagonal must be within the fixed tolerance of its matched full-FT diagonal, "
        "and each earlier task must retain its own post-learning accuracy within that tolerance. "
        "The full-FT reference must itself beat each validation majority baseline. "
        "The primary supplementary tolerance is 2pp; 1pp/5pp are sensitivity analyses. "
        "These operational gates do not replace the original window. JSON also reports Bonferroni "
        "95% t intervals over 25 constraints within each model/LR family; five seeds give limited power.")
    add("")
    add("| Model | LR | 2pp set | Interval-supported set | Status |")
    add("|---|---|---|---|---|")
    for item in all_task_sets or []:
        if item["tolerance_pp"] == 2:
            add(f"| {item['model']} | {item['lr']:.4g} | {item['window_ranks']} | "
                f"{item['interval_supported_ranks']} | {item['status']} |")
    add("")
    add("All diagonals, final accuracies and transition changes are in `aggregates/per_task_learning_retention.*` "
        "and `aggregates/task_transition_changes.*`. Per-matrix displacement, effective rank and "
        "leading-subspace distances at k=10/20 are in `aggregates/spectral_diagnostics_long.*`. "
        "Near-tied cutoff singular values are flagged; individual-vector intruder counts retain their original definition.")
    add("")
    add("## 3. Rank selection guidance (RQ4)")
    add("")
    if guidance:
        for model, g in guidance.items():
            label = config.MODELS[model].label if model in config.MODELS else model
            add(f"### {label}")
            add("")
            for lr, ranks in g["window_ranks_by_lr"].items():
                add(f"- Feasible window @ lr={utils_fmt(lr)}: "
                    f"r = {', '.join(str(r) for r in ranks) or '—'}")
            bp, bs = g["best_plasticity"], g["most_stable"]
            add(f"- Best plasticity: r={bp['rank']} @ lr={utils_fmt(bp['lr'])} "
                f"(A33 {bp['a33_mean_pct']:.2f}%, BWT {bp['bwt_mean_pp']:.2f} pp"
                f"{', in window' if bp['in_feasible_window'] else ''})")
            add(f"- Most stable: r={bs['rank']} @ lr={utils_fmt(bs['lr'])} "
                f"(BWT {bs['bwt_mean_pp']:.2f} pp, A33 {bs['a33_mean_pct']:.2f}%)")
            add("")
    else:
        add("_No complete five-rank, five-seed study available for guidance yet._")
        add("")

    add("## 4. Intruder dimensions vs forgetting (RQ3)")
    add("")
    add("Positive ρ means more intruders accompany more forgetting (−BWT). "
        "Within-model comparisons are primary. Pooled analysis normalizes by the "
        "number of inspected vectors and remains architecture-confounded. "
        "Run-level p-values are exploratory: ranks sharing seeds are dependent. "
        "Full-FT counts are a comparison, not a required zero-valued control.")
    add("")
    if corrs:
        add("| scope | n runs | Spearman ρ | p |")
        add("|---|---|---|---|")
        for c in corrs:
            if "spearman_rho" in c:
                add(f"| {c['scope']} | {c['n_runs']} | {c['spearman_rho']:.3f} | "
                    f"{c['p_value']:.2e} |")
            else:
                add(f"| {c['scope']} | {c['n_runs']} | "
                    f"mean intruders = {c.get('mean_intruders_eps09', 0):.1f} | "
                    f"max = {c.get('max_intruders_eps09', 0)} |")
        add("")
    else:
        add("_No intruder data available._")
        add("")

    add("## 5. Sanity / verification runs")
    add("")
    if sanity:
        for r in sanity:
            acc = r["metrics"].get("final_task_acc")
            add(f"- `{r['run_id']}`: RTE accuracy {_fmt_pct(acc)}, "
                f"NaN steps: {r['flags']['nan_loss_tasks'] or 'none'}, "
                f"grad spikes: {len(r['flags']['grad_spike_steps'])}")
        add("")
    else:
        add("_No sanity runs recorded (run `--mode sanity` to produce them)._")
        add("")

    add("## 6. Training flags")
    add("")
    add("Failed or incomplete runs do not enter these aggregates.")
    if audit:
        add(f"Recorded failure attempts: {len(audit.get('failure_attempts', []))}; "
            f"completed runs with gradient-spike flags: {len(audit['flagged_runs'])}.")
        for run_id in audit["flagged_runs"]:
            add(f"- `{run_id}`: inspect train_log.jsonl before drawing conclusions.")
    add("")

    add("## 7. Figures (grouped per model; cross-model views last)")
    add("")
    groups = {}
    for f in figures:
        owner = "cross-model & overview"
        for model in sorted(df["model"].unique()):
            san = model.replace("/", "--")
            if f"_{san}." in f:
                owner = (config.MODELS[model].label if model in config.MODELS
                         else model)
                break
        groups.setdefault(owner, []).append(f)
    for owner, names in groups.items():
        add(f"### {owner}")
        add("")
        for f in names:
            add(f"- ![{f}]({f})")
        add("")

    report_path = Path(results_dir) / "analysis" / "report.md"
    report_path.write_text("\n".join(lines))
    logger.info("Report written to %s", report_path)


def _mean_sd(mean, sd) -> str:
    if sd is None or np.isnan(sd):
        return f"{mean * PP:.2f} (SD unavailable)"
    return f"{mean * PP:.2f} ± {sd * PP:.2f}"
