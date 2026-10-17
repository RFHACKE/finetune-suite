"""Extended figure set for the feasible-rank-window research.

Every per-model figure contains ALL hyperparameter variants (rank x LR
configurations) of that model in one chart, so configurations can be compared
directly. Cross-model figures are limited to cases where the comparison is
normalized (r/d_model units) or anchor-only, since raw cross-model comparison
is confounded by parameter count.

Chart vocabulary taken from the continual-learning / rank-ablation
literature: accuracy matrices and per-task retention trajectories
(Lopez-Paz & Ranzato 2017; de Lange et al. 2022), the stability-plasticity
plane (Grossberg 1982), plasticity gap / intransigence-style measures
(Rodriguez-Lomonaco & Maltoni 2018), seed-level distributions (Narula et al.
2026), rank x LR heatmaps and error-band rank curves (Biderman et al. 2024;
Rathore et al. 2025), epsilon-sensitivity curves (Shuttleworth et al. 2025).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config

PP = 100.0
TASKS = list(config.TASK_KEYS)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130,
        "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    })
    return plt


def save(fig, out_dir: Path, name: str) -> str:
    name = name.replace("/", "--")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001 - decoration overflow is not fatal
            pass
    fig.savefig(Path(out_dir) / name, bbox_inches="tight")
    import matplotlib.pyplot as mplt

    mplt.close(fig)
    return name


def fmt_lr(lr) -> str:
    from .utils import fmt_lr as _f

    return _f(float(lr))


def cfg_label(rank, lr) -> str:
    return f"r={int(rank)}, lr={fmt_lr(lr)}"


def lora_configs(runs_df):
    """Sorted (rank, lr) pairs present in the data."""
    sub = runs_df[(runs_df["track"] == "lora") & runs_df["rank"].notna()]
    pairs = {(int(r["rank"]), float(r["lr"])) for _, r in sub.iterrows()}
    return sorted(pairs, key=lambda p: (p[1], p[0]))


def full_ft_ref(runs_df, model):
    """The declared baseline, only after all five seeds; never select its LR by outcome."""
    full = runs_df[(runs_df["track"] == "full") & (runs_df["model"] == model)]
    full = full[np.isclose(full["lr"], config.MODELS[model].default_full_lr, rtol=1e-10, atol=0)]
    if len(full) != 5 or set(full["seed"]) != set(config.DEFAULT_SEEDS):
        return None
    return float(full["final_task_acc"].mean())


def style_for(index, total):
    cmap = None
    import matplotlib

    cmap = matplotlib.colormaps["tab10"]
    color = cmap(index % 10)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    return color, markers[index % len(markers)]


def _cfg_rows(runs_df, model, rank, lr):
    return runs_df[(runs_df["model"] == model)
                   & (runs_df["track"] == "lora")
                   & (runs_df["rank"] == rank)
                   & (runs_df["lr"] == lr)]


# --------------------------------------------------------------------------
# Grouped bar charts (all variants of one model in a single chart)
# --------------------------------------------------------------------------
def fig_metric_bars(plt, runs_df, out_dir, model, label, metric, title, ylabel,
                    ref=None, ref_label=None, name="bars"):
    cfgs = lora_configs(runs_df[runs_df["model"] == model])
    if not cfgs:
        return None
    lrs = sorted({lr for _, lr in cfgs})
    ranks = sorted({r for r, _ in cfgs})
    fig, ax = plt.subplots(figsize=(max(7.0, 1.3 * len(ranks) * len(lrs)), 4.4))
    width = 0.8 / len(lrs)
    x = np.arange(len(ranks))
    for li, lr in enumerate(lrs):
        means, stds = [], []
        for r in ranks:
            rows = _cfg_rows(runs_df, model, r, lr)
            if rows.empty:
                means.append(0.0)
                stds.append(0.0)
                continue
            vals = rows[metric].dropna()
            means.append(float(vals.mean()) if len(vals) else 0.0)
            stds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
        ax.bar(x + li * width - 0.4 + width / 2, np.array(means) * PP,
               width, yerr=np.array(stds) * PP, capsize=3,
               label=f"lr={fmt_lr(lr)}")
    if ref is not None:
        ax.axhline(ref * PP, color="red", ls="--", lw=1.2,
                   label=ref_label or f"full-FT = {ref * PP:.2f}")
    ax.set_xticks(x, [str(r) for r in ranks])
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}: {label}\nall rank × LR variants (mean ± seed std)")
    ax.legend(fontsize=7)
    return save(fig, out_dir, f"{name}_{model}.png")


# --------------------------------------------------------------------------
# Per-task retention trajectories
# --------------------------------------------------------------------------
def fig_task_trajectories(plt, runs_df, out_dir, model, label):
    """Accuracy on each task at every task boundary, all configs in one chart
    (the standard continual-learning trajectory plot)."""
    cfgs = lora_configs(runs_df[runs_df["model"] == model])
    if not cfgs:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    boundaries = ["after 1st task", "after 2nd task", "after 3rd task"]
    for ax, task in zip(axes, TASKS):
        for ci, (rank, lr) in enumerate(cfgs):
            rows = _cfg_rows(runs_df, model, rank, lr)
            traj = []
            for after in TASKS:
                col = f"A_{after}_{task}"
                traj.append(float(rows[col].mean() * PP)
                            if col in rows.columns and rows[col].notna().any()
                            else np.nan)
            # a task is only measured from its own training onward
            ti = TASKS.index(task)
            xs = list(range(ti, 3))
            color, marker = style_for(ci, len(cfgs))
            ax.plot(xs, traj[ti:], marker=marker, color=color,
                    label=cfg_label(rank, lr), alpha=0.85)
        ax.set_xticks(range(3), boundaries, rotation=20, fontsize=7)
        ax.set_title(f"task: {task}")
        ax.set_ylabel("accuracy (%)")
    axes[0].legend(fontsize=6)
    fig.suptitle(f"Per-task accuracy through the sequence: {label}", fontsize=11)
    return save(fig, out_dir, f"task_trajectories_{model}.png")


# --------------------------------------------------------------------------
# Accuracy-matrix small multiples: one heatmap per configuration
# --------------------------------------------------------------------------
def fig_accuracy_matrices(plt, runs_df, out_dir, model, label):
    cfgs = lora_configs(runs_df[runs_df["model"] == model])
    if not cfgs:
        return None
    n = len(cfgs)
    cols = min(4, n)
    rows_n = -(-n // cols)
    fig, axes = plt.subplots(rows_n, cols,
                             figsize=(3.2 * cols, 2.9 * rows_n),
                             squeeze=False)
    for idx, (rank, lr) in enumerate(cfgs):
        ax = axes[idx // cols][idx % cols]
        sub = _cfg_rows(runs_df, model, rank, lr)
        mat = np.full((3, 3), np.nan)
        for i, ti in enumerate(TASKS):
            for j, tj in enumerate(TASKS):
                if j <= i:
                    col = f"A_{ti}_{tj}"
                    if col in sub.columns and sub[col].notna().any():
                        mat[i, j] = float(sub[col].mean()) * PP
        im = ax.imshow(np.ma.masked_invalid(mat), cmap="Greens",
                       vmin=0, vmax=100, aspect="auto")
        for i in range(3):
            for j in range(3):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                            fontsize=7)
        ax.set_xticks(range(3), TASKS, fontsize=7)
        ax.set_yticks(range(3), [f"after {t}" for t in TASKS], fontsize=7)
        ax.set_title(cfg_label(rank, lr), fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for idx in range(n, rows_n * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle(f"Accuracy matrices A[i][j] per configuration: {label}",
                 fontsize=11)
    return save(fig, out_dir, f"accuracy_matrices_{model}.png")


# --------------------------------------------------------------------------
# Stability-plasticity plane
# --------------------------------------------------------------------------
def fig_stability_plasticity(plt, runs_df, out_dir, model, label, windows):
    sub = runs_df[(runs_df["model"] == model) & (runs_df["track"] == "lora")]
    if sub.empty or sub["bwt"].isna().all():
        return None
    ref = full_ft_ref(runs_df, model)
    win_ranks = set()
    for w in windows:
        if w["model"] == model:
            win_ranks |= set(w["window_ranks"])
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    cfgs = lora_configs(runs_df[runs_df["model"] == model])
    for ci, (rank, lr) in enumerate(cfgs):
        rows = _cfg_rows(runs_df, model, rank, lr)
        x = float(rows["bwt"].mean()) * PP
        y = float(rows["final_task_acc"].mean()) * PP
        color, marker = style_for(ci, len(cfgs))
        star = rank in win_ranks
        ax.scatter(x, y, s=110 if star else 55, marker="*" if star else marker,
                   color=color, edgecolors="black", linewidths=0.5, zorder=3)
        ax.annotate(cfg_label(rank, lr), (x, y), fontsize=6.5,
                    textcoords="offset points", xytext=(5, 4))
    ax.axvline(0, color="grey", lw=0.8)
    ax.axvline(-config.WINDOW_MARGIN_PP, color="grey", ls=":", lw=1)
    if ref is not None:
        ax.axhline(ref * PP, color="red", ls="--", lw=1,
                   label=f"full-FT A33 = {ref * PP:.2f}%")
        ax.axhline(ref * PP - config.WINDOW_MARGIN_PP, color="red", ls=":",
                   lw=1, label="plasticity margin (2 pp)")
    ax.set_xlabel("BWT (pp)   [left = more forgetting]")
    ax.set_ylabel("Final-task accuracy (%)")
    ax.set_title(f"Stability–plasticity plane: {label}\n"
                 "★ = inside feasible rank window; upper-left is the target zone")
    if ref is not None:
        ax.legend(fontsize=7)
    return save(fig, out_dir, f"stability_plasticity_{model}.png")


# --------------------------------------------------------------------------
# Plasticity gap (intransigence-style)
# --------------------------------------------------------------------------
def fig_plasticity_gap(plt, runs_df, out_dir, model, label):
    sub = runs_df[(runs_df["model"] == model) & (runs_df["track"] == "lora")]
    ref = full_ft_ref(runs_df, model)
    if sub.empty or ref is None:
        return None
    cfgs = lora_configs(runs_df[runs_df["model"] == model])
    fig, ax = plt.subplots(figsize=(max(7.0, 1.1 * len(cfgs)), 4.2))
    labels, gaps = [], []
    for rank, lr in cfgs:
        rows = _cfg_rows(runs_df, model, rank, lr)
        gaps.append((ref - float(rows["final_task_acc"].mean())) * PP)
        labels.append(f"r={rank}\nlr={fmt_lr(lr)}")
    ax.bar(range(len(cfgs)), gaps, color="tab:orange")
    ax.axhline(config.WINDOW_MARGIN_PP, color="red", ls="--", lw=1,
               label="plasticity margin (2 pp)")
    ax.set_xticks(range(len(cfgs)), labels, fontsize=7)
    ax.set_ylabel("plasticity gap vs full FT (pp)")
    ax.set_title(f"Plasticity gap (full-FT A33 − config A33): {label}\n"
                 "smaller = closer to the measured full-FT reference")
    ax.legend(fontsize=7)
    return save(fig, out_dir, f"plasticity_gap_{model}.png")


# --------------------------------------------------------------------------
# Seed-level distributions (spaghetti + box)
# --------------------------------------------------------------------------
def fig_seed_spaghetti(plt, runs_df, out_dir, model, label):
    sub = runs_df[(runs_df["model"] == model) & (runs_df["track"] == "lora")]
    if sub.empty:
        return None
    cfgs = lora_configs(runs_df[runs_df["model"] == model])
    fig, axes = plt.subplots(1, 2, figsize=(max(11.0, 1.4 * len(cfgs)), 4.4))
    for ax, metric, title in ((axes[0], "bwt", "BWT per seed (pp)"),
                              (axes[1], "final_task_acc",
                               "Final-task accuracy per seed (%)")):
        for ci, (rank, lr) in enumerate(cfgs):
            rows = _cfg_rows(runs_df, model, rank, lr)
            vals = rows[metric].dropna().values * PP
            color, _ = style_for(ci, len(cfgs))
            xs = np.full(len(vals), ci) + np.linspace(-0.15, 0.15, len(vals))
            ax.scatter(xs, vals, color=color, s=22, alpha=0.8)
            if len(vals):
                ax.hlines(vals.mean(), ci - 0.28, ci + 0.28,
                          color="black", lw=1.4)
        ax.set_xticks(range(len(cfgs)),
                      [f"r={r}\n{fmt_lr(rate)}" for r, rate in cfgs],
                      fontsize=6.5)
        ax.set_title(title)
    fig.suptitle(f"Seed-level spread per configuration: {label}\n"
                 "(black bar = mean; wide spread = unreliable conclusion)",
                 fontsize=11)
    return save(fig, out_dir, f"seed_spaghetti_{model}.png")


def fig_seed_boxplot(plt, runs_df, out_dir, model, label):
    sub = runs_df[(runs_df["model"] == model) & (runs_df["track"] == "lora")]
    if sub.empty:
        return None
    cfgs = lora_configs(runs_df[runs_df["model"] == model])
    fig, ax = plt.subplots(figsize=(max(8.0, 1.2 * len(cfgs)), 4.4))
    data = []
    for rank, lr in cfgs:
        rows = _cfg_rows(runs_df, model, rank, lr)
        data.append((rows["bwt"].dropna() * PP).tolist())
    ax.boxplot(data, tick_labels=[f"r={r}\n{fmt_lr(rate)}" for r, rate in cfgs])
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("BWT (pp)")
    ax.set_title(f"BWT distribution across seeds per configuration: {label}")
    ax.tick_params(axis="x", labelsize=6.5)
    return save(fig, out_dir, f"seed_boxplot_{model}.png")


# --------------------------------------------------------------------------
# Intruder-dimension figures
# --------------------------------------------------------------------------
def fig_intruder_vs_rank(plt, runs_df, out_dir, model, label):
    sub = runs_df[(runs_df["model"] == model) & (runs_df["track"] == "lora")]
    if sub.empty or sub["intruder_total_final"].isna().all():
        return None
    cfgs = lora_configs(runs_df[runs_df["model"] == model])
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    lrs = sorted({lr for _, lr in cfgs})
    for li, lr in enumerate(lrs):
        ranks = sorted({r for r, rate in cfgs if rate == lr})
        ys, es = [], []
        for r in ranks:
            rows = _cfg_rows(runs_df, model, r, lr)
            v = rows["intruder_total_final"].dropna()
            ys.append(float(v.mean()) if len(v) else np.nan)
            es.append(float(v.std(ddof=1)) if len(v) > 1 else 0.0)
        color, marker = style_for(li, len(lrs))
        ax.errorbar(ranks, ys, yerr=es, marker=marker, color=color,
                    label=f"lr={fmt_lr(lr)}", capsize=3)
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel(f"intruder dimensions (k=10, ε={config.INTRUDER_EPS_PRIMARY})")
    ax.set_title(f"Intruder dimensions vs rank: {label}")
    ax.legend(fontsize=7)
    return save(fig, out_dir, f"intruder_vs_rank_{model}.png")


def fig_intruder_accumulation(plt, results, out_dir, model, label):
    """Intruder counts at each task boundary — accumulation within one run."""
    rows = {}
    for r in results:
        if r.get("model") != model or r.get("track") != "lora":
            continue
        key = (int(r["rank"]), float(r["lr"]))
        for boundary, payload in (r.get("intruder") or {}).items():
            n = (payload.get("total") or {}).get(
                f"{config.INTRUDER_EPS_PRIMARY:.1f}")
            if n is None:
                continue
            rows.setdefault(key, {}).setdefault(boundary, []).append(n)
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    cfgs = sorted(rows, key=lambda p: (p[1], p[0]))
    boundaries = [f"after_{t}" for t in TASKS]
    for ci, key in enumerate(cfgs):
        ys = [float(np.mean(rows[key].get(b, [np.nan]))) for b in boundaries]
        color, marker = style_for(ci, len(cfgs))
        ax.plot(range(len(boundaries)), ys, marker=marker, color=color,
                label=cfg_label(*key))
    ax.set_xticks(range(len(boundaries)),
                  [f"after {t}" for t in TASKS], fontsize=8)
    ax.set_ylabel(f"intruder count (ε={config.INTRUDER_EPS_PRIMARY})")
    ax.set_title(f"Intruder accumulation across task boundaries: {label}")
    ax.legend(fontsize=6)
    return save(fig, out_dir, f"intruder_accumulation_{model}.png")


def fig_intruder_cosine_trajectory(plt, results, out_dir, model, label):
    rows = {}
    for r in results:
        if r.get("model") != model or r.get("track") != "lora":
            continue
        key = (int(r["rank"]), float(r["lr"]))
        for boundary, payload in (r.get("intruder") or {}).items():
            v = payload.get("mean_max_cosine")
            if v is not None:
                rows.setdefault(key, {}).setdefault(boundary, []).append(v)
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    cfgs = sorted(rows, key=lambda p: (p[1], p[0]))
    boundaries = [f"after_{t}" for t in TASKS]
    for ci, key in enumerate(cfgs):
        ys = [float(np.mean(rows[key].get(b, [np.nan]))) for b in boundaries]
        color, marker = style_for(ci, len(cfgs))
        ax.plot(range(len(boundaries)), ys, marker=marker, color=color,
                label=cfg_label(*key))
    ax.set_xticks(range(len(boundaries)), [f"after {t}" for t in TASKS],
                  fontsize=8)
    ax.set_ylabel("mean max cosine (top-10 singular vectors)")
    ax.set_title(f"Spectral drift of adapted matrices: {label}\n"
                 "(1.0 = pretrained directions preserved)")
    ax.legend(fontsize=6)
    return save(fig, out_dir, f"intruder_cosine_trajectory_{model}.png")


# --------------------------------------------------------------------------
# Gradient-norm curves (training health)
# --------------------------------------------------------------------------
def fig_grad_norm_curves(plt, results, out_dir, model, label):
    import pandas as pd

    per_cfg = {}
    for r in results:
        if r.get("model") != model or r.get("track") != "lora":
            continue
        log_path = Path(r["_dir"]) / "train_log.jsonl"
        if not log_path.exists():
            continue
        key = (int(r["rank"]), float(r["lr"]))
        steps, norms = [], []
        with open(log_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                steps.append(rec.get("step"))
                norms.append(rec.get("grad_norm"))
        per_cfg.setdefault(key, []).append(pd.DataFrame(
            {"step": steps, "grad_norm": norms}))
    if not per_cfg:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    cfgs = sorted(per_cfg, key=lambda p: (p[1], p[0]))
    for ci, key in enumerate(cfgs):
        merged = pd.concat(per_cfg[key])
        g = merged.groupby("step")["grad_norm"].mean()
        g = g.iloc[:: max(1, len(g) // 400)]   # downsample for readability
        color, _ = style_for(ci, len(cfgs))
        ax.plot(g.index, g.values, color=color, lw=1,
                label=cfg_label(*key))
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step (all tasks concatenated)")
    ax.set_ylabel("grad norm (log)")
    ax.set_title(f"Gradient-norm curves per configuration: {label}")
    ax.legend(fontsize=6, ncol=2)
    return save(fig, out_dir, f"grad_norm_curves_{model}.png")


# --------------------------------------------------------------------------
# Radar summary per configuration
# --------------------------------------------------------------------------
def fig_radar(plt, runs_df, out_dir, model, label):
    sub = runs_df[(runs_df["model"] == model) & (runs_df["track"] == "lora")]
    cfgs = lora_configs(runs_df[runs_df["model"] == model])
    if sub.empty or len(cfgs) < 2:
        return None
    metrics = {
        "final acc": "final_task_acc",
        "avg acc": "avg_final_row",
        "stability (BWT)": "bwt",
        "retention RTE": f"A_{TASKS[-1]}_{TASKS[0]}",
        "retention MNLI": f"A_{TASKS[-1]}_{TASKS[1]}",
        "few intruders": "intruder_total_final",
    }
    metrics = {k: v for k, v in metrics.items()
               if v in sub.columns and sub[v].notna().any()}
    if len(metrics) < 3:
        return None

    raw = {}
    for rank, lr in cfgs:
        rows = _cfg_rows(runs_df, model, rank, lr)
        vals = {}
        for name, col in metrics.items():
            v = rows[col].dropna()
            val = float(v.mean()) if len(v) else np.nan
            if name == "few intruders":
                val = -val          # higher is better on every radar axis
            vals[name] = val
        raw[(rank, lr)] = vals

    # normalize each axis to [0.05, 1] across configs
    normed = {}
    for name in metrics:
        vs = np.array([raw[c][name] for c in cfgs], dtype=float)
        lo, hi = np.nanmin(vs), np.nanmax(vs)
        rng = (hi - lo) or 1.0
        for i, c in enumerate(cfgs):
            normed.setdefault(c, {})[name] = 0.05 + 0.95 * (
                (raw[c][name] - lo) / rng)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, polar=True)
    for ci, c in enumerate(cfgs):
        vals = [normed[c][m] for m in metrics] + [normed[c][list(metrics)[0]]]
        color, _ = style_for(ci, len(cfgs))
        ax.plot(angles, vals, color=color, label=cfg_label(*c))
        ax.fill(angles, vals, color=color, alpha=0.08)
    ax.set_xticks(angles[:-1], list(metrics), fontsize=8)
    ax.set_title(f"Multi-metric configuration profile (normalized): {label}",
                 pad=22)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.35, 1.1))
    return save(fig, out_dir, f"radar_{model}.png")


# --------------------------------------------------------------------------
# Rendered configuration table
# --------------------------------------------------------------------------
def fig_config_table(plt, runs_df, out_dir, model, label, windows):
    sub = runs_df[(runs_df["model"] == model) & (runs_df["track"] == "lora")]
    if sub.empty:
        return None
    win_ranks = {}
    for w in windows:
        if w["model"] == model:
            win_ranks = {r: w["window_ranks"] for r in [w["lr"]]}
    cfgs = lora_configs(runs_df[runs_df["model"] == model])
    cell_text = []
    for rank, lr in cfgs:
        rows = _cfg_rows(runs_df, model, rank, lr)
        bwt = rows["bwt"].dropna()
        acc = rows["final_task_acc"].dropna()
        intr = rows["intruder_total_final"].dropna()
        in_win = "yes" if rank in win_ranks.get(lr, []) else ""
        cell_text.append([
            f"r={rank}", fmt_lr(lr), str(len(rows)),
            f"{bwt.mean() * PP:.2f} ± {bwt.std(ddof=1) * PP:.2f}" if len(bwt) else "n/a",
            f"{acc.mean() * PP:.2f} ± {acc.std(ddof=1) * PP:.2f}" if len(acc) else "n/a",
            f"{intr.mean():.1f}" if len(intr) else "n/a",
            in_win,
        ])
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(cfgs) + 1.6))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        colLabels=["rank", "lr", "seeds", "BWT (pp)", "A33 (%)",
                   f"intruders ε={config.INTRUDER_EPS_PRIMARY}", "in window"],
        loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.4)
    ax.set_title(f"Configuration summary: {label}", fontsize=11)
    return save(fig, out_dir, f"config_table_{model}.png")


# --------------------------------------------------------------------------
# Dashboard: one overview figure per model
# --------------------------------------------------------------------------
def fig_dashboard(plt, runs_df, out_dir, model, label, windows):
    sub = runs_df[(runs_df["model"] == model) & (runs_df["track"] == "lora")]
    if sub.empty:
        return None
    cfgs = lora_configs(runs_df[runs_df["model"] == model])
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))

    def _series(ax, metric, ylabel, title, scale=PP):
        lrs = sorted({lr for _, lr in cfgs})
        for li, lr in enumerate(lrs):
            ranks = sorted({r for r, rate in cfgs if rate == lr})
            ys, es = [], []
            for r in ranks:
                rows = _cfg_rows(runs_df, model, r, lr)
                v = rows[metric].dropna()
                ys.append(float(v.mean()) * scale if len(v) else np.nan)
                es.append(float(v.std(ddof=1)) * scale if len(v) > 1 else 0.0)
            color, marker = style_for(li, len(lrs))
            ax.errorbar(ranks, ys, yerr=es, marker=marker, color=color,
                        label=f"lr={fmt_lr(lr)}", capsize=3)
        ax.set_xlabel("rank r")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6)

    _series(axes[0][0], "bwt", "BWT (pp)", "rank-forgetting curve")
    axes[0][0].axhline(0, color="black", lw=0.8)
    _series(axes[0][1], "final_task_acc", "A33 (%)", "plasticity vs rank")
    ref = full_ft_ref(runs_df, model)
    if ref is not None:
        axes[0][1].axhline(ref * PP, color="red", ls="--", lw=1)
    _series(axes[0][2], "intruder_total_final", "intruders",
            "intruder dimensions vs rank", scale=1.0)

    # scatter: stability-plasticity
    ax = axes[1][0]
    for ci, (rank, lr) in enumerate(cfgs):
        rows = _cfg_rows(runs_df, model, rank, lr)
        if rows["bwt"].notna().any():
            color, marker = style_for(ci, len(cfgs))
            ax.scatter(float(rows["bwt"].mean()) * PP,
                       float(rows["final_task_acc"].mean()) * PP,
                       color=color, marker=marker, s=40)
            ax.annotate(f"r={rank}", (float(rows["bwt"].mean()) * PP,
                                      float(rows["final_task_acc"].mean()) * PP),
                        fontsize=6, textcoords="offset points", xytext=(4, 3))
    ax.set_xlabel("BWT (pp)")
    ax.set_ylabel("A33 (%)")
    ax.set_title("stability–plasticity", fontsize=9)

    # BWT heatmap rank x LR
    ax = axes[1][1]
    ranks = sorted({r for r, _ in cfgs})
    lrs = sorted({lr for _, lr in cfgs})
    mat = np.full((len(ranks), len(lrs)), np.nan)
    for i, r in enumerate(ranks):
        for j, lr in enumerate(lrs):
            rows = _cfg_rows(runs_df, model, r, lr)
            if not rows.empty and rows["bwt"].notna().any():
                mat[i, j] = float(rows["bwt"].mean()) * PP
    im = ax.imshow(np.ma.masked_invalid(mat), cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(lrs)), [fmt_lr(rate) for rate in lrs], rotation=45,
                  ha="right", fontsize=6)
    ax.set_yticks(range(len(ranks)), [str(r) for r in ranks], fontsize=7)
    ax.set_title("BWT (pp): rank × LR", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # feasible window bar
    ax = axes[1][2]
    for w in windows:
        if w["model"] == model and w["window_r_over_d"]:
            xs = w["window_r_over_d"]
            ax.scatter(xs, [0] * len(xs), marker="s", s=100)
            for x, r in zip(xs, w["window_ranks"]):
                ax.text(x, 0, f" r={r}", va="center", fontsize=7)
            ax.set_yticks([])
            ax.set_xlabel("rank / d_model")
            ax.set_title(f"feasible window (lr={fmt_lr(w['lr'])})", fontsize=9)

    fig.suptitle(f"Configuration overview: {label}", fontsize=13)
    return save(fig, out_dir, f"dashboard_{model}.png")


# --------------------------------------------------------------------------
# Cross-model figures (normalized or anchor-only; secondary)
# --------------------------------------------------------------------------
def fig_window_drift_raw(plt, windows, out_dir):
    usable = [w for w in windows if w["window_ranks"]]
    if not usable:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 1.3 * len(usable) + 1.6))
    for i, w in enumerate(usable):
        xs = w["window_ranks"]
        ax.scatter(xs, [i] * len(xs), marker="s", s=100)
        for x in xs:
            ax.text(x, i, f" {x}", va="center", fontsize=7)
    ax.set_yticks(range(len(usable)),
                  [f"{w['model']} @ lr={fmt_lr(w['lr'])}" for w in usable],
                  fontsize=8)
    ax.set_xlabel("rank r (raw units)")
    ax.set_title("Feasible rank window per model — raw rank units")
    return save(fig, out_dir, "window_drift_raw_r.png")


def fig_cross_model_normalized(plt, runs_df, out_dir):
    """All models on the r/d_model axis — the RQ2 convention (Lun & Ming).
    Normalized units; NOT a raw performance comparison."""
    sub = runs_df[(runs_df["track"] == "lora") & runs_df["d_model"].notna()]
    sub = sub[sub.apply(lambda row: np.isclose(row["lr"], config.MODELS[row["model"]].default_lora_lr,
                                              rtol=1e-10, atol=0), axis=1)]
    if sub.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, metric, ylabel in ((axes[0], "bwt", "BWT (pp)"),
                               (axes[1], "final_task_acc", "A33 (%)")):
        models = sorted(sub["model"].unique())
        for mi, model in enumerate(models):
            rows = sub[sub["model"] == model]
            d = float(rows["d_model"].iloc[0])
            pts = rows.groupby("rank")[metric].mean()
            xs = [r / d for r in pts.index]
            ys = pts.values * PP
            order = np.argsort(xs)
            color, marker = style_for(mi, len(models))
            ax.plot(np.array(xs)[order], np.array(ys)[order], marker=marker,
                    color=color, label=model)
        ax.set_xlabel("rank / d_model")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
    fig.suptitle("Normalized rank scale (r/d_model) — cross-model view on the "
                 "RQ2 convention (size-confounded; use per-model figures for "
                 "claims)", fontsize=10)
    return save(fig, out_dir, "cross_model_r_over_d.png")


def fig_full_ft_anchors(plt, runs_df, out_dir):
    full = runs_df[runs_df["track"] == "full"]
    full = full[full.apply(lambda row: np.isclose(row["lr"], config.MODELS[row["model"]].default_full_lr,
                                                rtol=1e-10, atol=0), axis=1)]
    if full.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    models = sorted(full["model"].unique())
    x = np.arange(len(models))
    bw = [full[full["model"] == m]["bwt"].mean() * PP for m in models]
    ac = [full[full["model"] == m]["final_task_acc"].mean() * PP for m in models]
    ax.bar(x - 0.2, bw, 0.4, label="BWT (pp)")
    ax.bar(x + 0.2, ac, 0.4, label="A33 (%)")
    ax.set_xticks(x, models, fontsize=8)
    ax.set_title("Full fine-tuning anchors per model (Track B)\n"
                 "(reference values, not a size comparison)")
    ax.legend(fontsize=8)
    return save(fig, out_dir, "full_ft_anchors.png")
