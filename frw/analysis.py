"""frw-v4 analysis: aggregates, pre-registered windows, statistics, figures (+ JSON for every figure).

    python -m frw.analysis --root results-v4 [--v3 Finetune-result]

Outputs under <root>/analysis/: runs.csv/json, lr_surface.json, selected_lr.json,
main_summary.json, windows.json, paired_tests.json, trends.json, order_robustness.json,
spectral_association.json, figures/*.png + figures/*.json, report.md.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

from . import config as C

RNG_SEED = 0
N_BOOT = 10000
DELTAS = (1.0, 2.0, 3.0, 5.0)
PRIMARY_DELTA = 2.0
TASK_LABEL = {"rte": "RTE", "mnli": "MNLI-m", "sst2": "SST-2"}
# Reference categorical palette (light), fixed order by model registry position.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MODEL_COLOR = {m: SERIES[i] for i, m in enumerate(C.MODEL_ORDER)}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_runs(root: Path) -> list:
    rows = []
    for f in sorted((root / "runs").glob("*/result.json")):
        r = json.loads(f.read_text())
        s = r["spec"]
        row = {"job_id": r["job_id"], "kind": s["kind"], "model": s["model"], "track": s["track"],
               "rank": s["rank"], "lr": s["lr"], "seed": s["seed"], "order": s["order"], "split": s["split"],
               "status": r["status"], "A": r["A"], "NLL": r["NLL"], "probe": r.get("probe"),
               "tasks": r["order_tasks"], "gpu": r.get("gpu"), "wall_s": r.get("wall_s"),
               "timing": r.get("timing_s"), "trainable_params": r.get("trainable_params"),
               "spectral": r.get("spectral_summary"), "dir": str(f.parent),
               "account": (f.parent / "account.txt").read_text().strip() if (f.parent / "account.txt").exists() else None,
               "majority": {t: v.get("majority_eval_accuracy") for t, v in (r.get("data") or {}).items()}}
        m = r.get("metrics")
        if m:
            row.update({k: m[k] for k in ("acc_final", "plasticity", "bwt", "forgetting_max", "avg_incremental_acc")})
            for j, t in enumerate(r["order_tasks"]):
                row[f"diag_{t}"] = m["diag"][j]
                row[f"final_{t}"] = m["final"][j]
            for j, t in enumerate(r["order_tasks"][:-1]):
                row[f"bwt_{t}"] = m["bwt_terms"][j]
            P = r.get("probe")
            if P and all(P[i][j] is not None for i in range(len(P)) for j in range(i + 1)):
                T = len(P)
                row["probe_bwt"] = sum(P[T - 1][j] - P[j][j] for j in range(T - 1)) / (T - 1)
                row["probe_final_mean"] = sum(P[T - 1]) / T
            if r.get("spectral_summary"):
                last = r["spectral_summary"][-1]["summary"].get("all", {})
                row["rel_fro"] = last.get("rel_fro_mean")
                row["entropy_rank"] = last.get("entropy_rank_mean")
                row["stable_rank"] = last.get("stable_rank_mean")
                row["subspace_dist"] = last.get("subspace_dist_k10_mean")
                row["intruder_frac_0.9"] = (last.get("intruder_fraction") or {}).get("0.9")
                row["intruder_frac_0.7"] = (last.get("intruder_fraction") or {}).get("0.7")
        rows.append(row)
    return rows


def load_selections(root: Path, suffix: str = "") -> dict:
    """LR selections for one cohort: q/v (suffix "") or all-linear ("_al")."""
    out = {}
    for f in (root / "accounts").glob("*/selections/*.json"):
        d = json.loads(f.read_text())
        if d.get("suffix", "") == suffix:
            out[(d["model"], d["rank"])] = d
    return out


def load_v3(v3_root: Path) -> list:
    rows = []
    for sub in ("runs", "large-ranks/runs"):
        for f in sorted((v3_root / sub).glob("*/result.json")):
            r = json.loads(f.read_text())
            A = r["A"]
            if r.get("track") != "lora" or any(A[i][j] is None for i in range(3) for j in range(i + 1)):
                continue
            rows.append({"model": r["model"], "rank": r["rank"], "seed": r["seed"], "lr": r["lr"],
                         "diag": [A[i][i] for i in range(3)], "final": A[2],
                         "acc_final": sum(A[2]) / 3, "bwt": ((A[2][0] - A[0][0]) + (A[2][1] - A[1][1])) / 2})
    return rows


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def boot_ci(values, n=N_BOOT, seed=RNG_SEED):
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))],
                   dtype=float)
    if len(v) == 0:
        return None, None, None
    if len(v) == 1:
        return float(v[0]), None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def signflip_p(diffs):
    """Exact two-sided sign-flip permutation p-value for the mean paired difference."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n == 0:
        return None
    obs = abs(d.mean())
    signs = np.array(np.meshgrid(*[[1, -1]] * n)).T.reshape(-1, n)
    perm = np.abs((signs * d).mean(axis=1))
    return float((perm >= obs - 1e-12).mean())


def holm(pvals: dict) -> dict:
    items = sorted((p, k) for k, p in pvals.items() if p is not None)
    m = len(items)
    out, running = {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def summarize(values):
    v = [x for x in values if x is not None]
    mean, lo, hi = boot_ci(v)
    return {"n": len(v), "mean": mean, "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
            "ci95": [lo, hi], "values": v}


def contiguous(ranks_sorted, members):
    runs, cur = [], []
    for r in ranks_sorted:
        if r in members:
            cur.append(r)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return [[c[0], c[-1]] for c in runs]


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
METRICS = ("acc_final", "plasticity", "bwt", "forgetting_max", "avg_incremental_acc", "probe_bwt",
           "probe_final_mean", "rel_fro", "entropy_rank", "stable_rank", "subspace_dist",
           "intruder_frac_0.9", "intruder_frac_0.7")


def pct(x):
    return None if x is None else 100.0 * x


def tune_surface(rows):
    out = {}
    for r in rows:
        if r["kind"] != "tune":
            continue
        key = f"{r['model']}|{r['rank']}|{r['lr']:.6g}"
        d = out.setdefault(key, {"model": r["model"], "rank": r["rank"], "lr": r["lr"], "seeds": [],
                                 "acc_final": [], "plasticity": [], "bwt": [], "status": []})
        d["seeds"].append(r["seed"])
        d["status"].append(r["status"])
        for k in ("acc_final", "plasticity", "bwt"):
            d[k].append(pct(r.get(k)) if r["status"] == "completed" else None)
    for d in out.values():
        for k in ("acc_final", "plasticity", "bwt"):
            v = [x for x in d[k] if x is not None]
            d[k + "_mean"] = float(np.mean(v)) if v else None
        d["selection_score"] = float(np.mean([x if x is not None else 0.0 for x in d["acc_final"]]))
    return out


def main_summary(rows, kind="main", order="O1"):
    out = {}
    for r in rows:
        if r["kind"] != kind or r["order"] != order or r["status"] != "completed":
            continue
        out.setdefault((r["model"], r["rank"]), []).append(r)
    summary = {}
    for (model, rank), rs in sorted(out.items(), key=lambda kv: (C.MODEL_ORDER.index(kv[0][0]), kv[0][1])):
        s = {"model": model, "rank": rank, "lr": rs[0]["lr"], "seeds": sorted(x["seed"] for x in rs)}
        tasks = rs[0]["tasks"]
        for k in METRICS:
            scale = 100.0 if k in ("acc_final", "plasticity", "bwt", "forgetting_max", "avg_incremental_acc",
                                   "probe_bwt", "probe_final_mean") else 1.0
            s[k] = summarize([x.get(k) * scale if x.get(k) is not None else None for x in rs])
        for t in tasks:
            s[f"diag_{t}"] = summarize([100 * x[f"diag_{t}"] for x in rs])
            s[f"final_{t}"] = summarize([100 * x[f"final_{t}"] for x in rs])
            if f"bwt_{t}" in rs[0]:
                s[f"bwt_{t}"] = summarize([100 * x[f"bwt_{t}"] for x in rs])
        T = len(tasks)
        s["A_mean"] = [[float(np.mean([100 * x["A"][i][j] for x in rs])) if j <= i else None for j in range(T)]
                       for i in range(T)]
        s["tasks"] = tasks
        s["by_seed"] = {x["seed"]: {k: x.get(k) for k in ("acc_final", "plasticity", "bwt")} for x in rs}
        summary[f"{model}|{rank}"] = s
    return summary


def windows(summary):
    out = {}
    for model in C.MODEL_ORDER:
        ss = {v["rank"]: v for v in summary.values() if v["model"] == model}
        if not ss:
            continue
        ranks = sorted(ss)
        tasks = ss[ranks[0]]["tasks"]
        m = C.MODELS[model]
        best_acc_rank = max(ranks, key=lambda r: ss[r]["acc_final"]["mean"])
        res = {"ranks_tested": ranks, "n_seeds": {r: ss[r]["acc_final"]["n"] for r in ranks},
               "best_acc_rank": best_acc_rank, "definitions": {}}
        for delta in DELTAS:
            best_diag = {t: max(ss[r][f"diag_{t}"]["mean"] for r in ranks) for t in tasks}
            learn = [r for r in ranks if all(ss[r][f"diag_{t}"]["mean"] >= best_diag[t] - delta for t in tasks)]
            retain = [r for r in ranks if ss[r]["bwt"]["mean"] >= -delta]
            both = [r for r in ranks if r in learn and r in retain]
            acc_best = ss[best_acc_rank]["acc_final"]["mean"]
            acc_point = [r for r in ranks if ss[r]["acc_final"]["mean"] >= acc_best - delta]
            acc_conf = []
            for r in ranks:
                if r == best_acc_rank:
                    acc_conf.append(r)
                    continue
                a = ss[r]["by_seed"]
                b = ss[best_acc_rank]["by_seed"]
                common = sorted(set(a) & set(b))
                d = [100 * (a[s]["acc_final"] - b[s]["acc_final"]) for s in common]
                _, lo, _ = boot_ci(d)
                if lo is not None and lo >= -delta:
                    acc_conf.append(r)
            res["definitions"][f"delta={delta:g}"] = {
                "learning_set": learn, "retention_set": retain, "learning_and_retention": both,
                "learning_and_retention_intervals": contiguous(ranks, both),
                "acc_point": acc_point, "acc_point_intervals": contiguous(ranks, acc_point),
                "acc_confident": acc_conf, "acc_confident_intervals": contiguous(ranks, acc_conf),
                "best_diag": best_diag}
        prim = res["definitions"][f"delta={PRIMARY_DELTA:g}"]
        res["primary"] = {k: prim[k] for k in ("learning_and_retention", "learning_and_retention_intervals",
                                               "acc_point_intervals", "acc_confident_intervals")}
        res["units"] = {"d_model": m.d_model, "v_width": m.v_out,
                        "r_over_d": {r: r / m.d_model for r in ranks}, "r_over_v": {r: r / m.v_out for r in ranks}}
        out[model] = res
    return out


def paired_tests(rows, summary):
    out = {}
    for model in C.MODEL_ORDER:
        ss = {v["rank"]: v for v in summary.values() if v["model"] == model}
        if not ss:
            continue
        best = max(ss, key=lambda r: ss[r]["acc_final"]["mean"])
        res = {"reference_rank": best, "metrics": {}}
        for metric in ("acc_final", "bwt", "plasticity"):
            ps, rows_m = {}, {}
            for r in sorted(ss):
                if r == best:
                    continue
                a, b = ss[r]["by_seed"], ss[best]["by_seed"]
                common = sorted(set(a) & set(b))
                d = [100 * (a[s][metric] - b[s][metric]) for s in common]
                mean, lo, hi = boot_ci(d)
                p = signflip_p(d) if d else None
                ps[r] = p
                rows_m[r] = {"n": len(d), "mean_diff_pp": mean, "ci95": [lo, hi], "p_signflip": p}
            adj = holm(ps)
            for r in rows_m:
                rows_m[r]["p_holm"] = adj.get(r)
            res["metrics"][metric] = rows_m
        out[model] = res
    return out


def trends(rows, kind="main"):
    out = {}
    for model in C.MODEL_ORDER:
        rs = [r for r in rows if r["model"] == model and r["kind"] == kind and r["order"] == "O1"
              and r["status"] == "completed"]
        if len({r["rank"] for r in rs}) < 3:
            continue
        res = {}
        for metric in ("acc_final", "bwt", "plasticity", "forgetting_max"):
            x = np.array([math.log2(r["rank"]) for r in rs])
            y = np.array([100 * r[metric] for r in rs])
            seeds = sorted({r["seed"] for r in rs})
            S = np.array([[1.0 if r["seed"] == s else 0.0 for s in seeds] for r in rs])
            X = np.column_stack([S, x, x ** 2])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            b, c = beta[-2], beta[-1]
            vertex = float(2 ** (-b / (2 * c))) if c != 0 else None
            means = {}
            for r in rs:
                means.setdefault(r["rank"], []).append(100 * r[metric])
            rk = sorted(means)
            from scipy.stats import spearmanr
            rho, p = spearmanr([math.log2(k) for k in rk], [np.mean(means[k]) for k in rk])
            res[metric] = {"linear_log2": float(b), "quadratic_log2": float(c),
                           "vertex_rank": vertex if c < 0 or metric == "forgetting_max" else None,
                           "spearman_rho_rank_means": float(rho), "spearman_p": float(p), "n_runs": len(rs)}
        out[model] = res
    return out


def spectral_association(rows):
    from scipy.stats import spearmanr
    out = {}
    for model in C.MODEL_ORDER:
        rs = [r for r in rows if r["model"] == model and r["kind"] == "main" and r["status"] == "completed"
              and r.get("intruder_frac_0.9") is not None]
        if len(rs) < 5:
            continue
        res = {}
        for feat in ("intruder_frac_0.9", "intruder_frac_0.7", "rel_fro", "entropy_rank", "subspace_dist"):
            x = [r[feat] for r in rs]
            y = [-100 * r["bwt"] for r in rs]
            rho, p = spearmanr(x, y)
            res[feat] = {"spearman_rho_vs_forgetting": float(rho), "p": float(p), "n": len(rs)}
        out[model] = res
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
                         "lines.linewidth": 2, "lines.markersize": 5})
    return plt


class Fig:
    def __init__(self, outdir: Path):
        self.dir = outdir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index = []

    def save(self, fig, name, data, caption):
        fig.savefig(self.dir / f"{name}.png", dpi=160, bbox_inches="tight")
        (self.dir / f"{name}.json").write_text(json.dumps({"caption": caption, "data": data}, indent=1,
                                                          default=float))
        self.index.append({"name": name, "caption": caption})
        import matplotlib.pyplot as plt
        plt.close(fig)


def _rank_axis(ax, ranks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks], rotation=45 if len(ranks) > 7 else 0, fontsize=7)
    ax.minorticks_off()
    ax.set_xlabel("LoRA rank r (log scale)")


def fig_v3_confound(F, v3):
    if not v3:
        return
    plt = _plt()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
    data = {}
    for ax, model in zip(axes, C.MODEL_ORDER):
        rs = [r for r in v3 if r["model"] == model]
        ranks = sorted({r["rank"] for r in rs})
        data[model] = {}
        for j, (t, col, chance) in enumerate((("rte", SERIES[0], 50), ("mnli", SERIES[1], 100 / 3),
                                              ("sst2", SERIES[2], 50))):
            mean = [np.mean([100 * r["diag"][j] for r in rs if r["rank"] == k]) for k in ranks]
            ax.plot(ranks, mean, marker="o", color=col, label=f"{TASK_LABEL[t]} learned (A_jj)")
            ax.axhline(chance, color=col, lw=0.8, ls=":")
            data[model][t] = dict(zip(ranks, mean))
        _rank_axis(ax, ranks)
        ax.set_title(f"{C.MODELS[model].label}\nv3: α=2r, one LR for all ranks")
    axes[0].set_ylabel("accuracy right after learning the task (%)")
    axes[0].legend(loc="lower left", fontsize=7)
    F.save(fig, "v3_rank_lr_confound", data,
           "Prior v3 cohorts (alpha=2r, LR tuned at r=16): diagonal accuracy collapses to chance (dotted) at "
           "high ranks, so their low forgetting is an absence of learning, not retention.")


def fig_lr_surface(F, surface, selections):
    if not surface:
        return
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    for metric, label in (("selection_score", "holdout ACC after the sequence (%)"),
                          ("bwt_mean", "holdout BWT (pp)"), ("plasticity_mean", "holdout mean diagonal (%)")):
        vals = [d.get(metric) for d in surface.values() if d.get(metric) is not None]
        if not vals:
            continue
        norm = Normalize(vmin=min(vals), vmax=max(vals))
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)
        data = {}
        for ax, model in zip(axes, C.MODEL_ORDER):
            cells = [d for d in surface.values() if d["model"] == model]
            if not cells:
                ax.set_visible(False)
                continue
            ranks = sorted({d["rank"] for d in cells})
            lrs = sorted({d["lr"] for d in cells})
            M = np.full((len(lrs), len(ranks)), np.nan)
            for d in cells:
                v = d.get(metric)
                M[lrs.index(d["lr"]), ranks.index(d["rank"])] = np.nan if v is None else v
            ax.imshow(M, aspect="auto", origin="lower", cmap=cmap, norm=norm)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if not np.isnan(M[i, j]):
                        dark = norm(M[i, j]) > 0.6
                        ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6,
                                color=SURFACE if dark else INK)
            ax.set_xticks(range(len(ranks)))
            ax.set_xticklabels(ranks, fontsize=7)
            ax.set_yticks(range(len(lrs)))
            ax.set_yticklabels([f"{x:.1e}" for x in lrs], fontsize=7)
            ax.grid(False)
            ax.set_xlabel("rank")
            ax.set_title(C.MODELS[model].label)
            for k, r in enumerate(ranks):
                sel = selections.get((model, r))
                if sel and sel["selected_lr"] in lrs:
                    ax.plot(k, lrs.index(sel["selected_lr"]), marker="s", mfc="none", mec=INK, ms=11, mew=1.5)
            data[model] = {"ranks": ranks, "lrs": lrs, "matrix": M.tolist()}
        axes[0].set_ylabel("adapter learning rate")
        import matplotlib.cm as cm
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axes), fraction=0.02, pad=0.01, label=label)
        F.save(fig, f"lr_surface_{metric}", data,
               f"LR search (tuning seeds, training holdout): {label} over rank × LR, one colour scale for all "
               "models; squares mark the pre-registered selection.")


def fig_selected_lr(F, selections):
    if not selections:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    data = {}
    for model in C.MODEL_ORDER:
        pts = sorted((r, d["selected_lr"]) for (m, r), d in selections.items() if m == model)
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=MODEL_COLOR[model],
                label=C.MODELS[model].label)
        data[model] = dict(pts)
    ranks = sorted({r for (_, r) in selections})
    _rank_axis(ax, ranks)
    ax.set_yscale("log")
    ax.set_ylabel("selected adapter LR")
    ax.legend(fontsize=7)
    ax.set_title("Selected LR vs rank (α fixed = 32)")
    F.save(fig, "selected_lr_vs_rank", data,
           "Per-rank selected learning rate. A flat line supports the approximate rank-invariance predicted for "
           "the fixed-alpha parameterisation; a slope means rank-specific tuning mattered.")


def _band(ax, ranks, s, key, color, label, scale=1.0):
    mean = [s[r][key]["mean"] * scale if s[r][key]["mean"] is not None else np.nan for r in ranks]
    lo = [(s[r][key]["ci95"][0] if s[r][key]["ci95"][0] is not None else s[r][key]["mean"]) for r in ranks]
    hi = [(s[r][key]["ci95"][1] if s[r][key]["ci95"][1] is not None else s[r][key]["mean"]) for r in ranks]
    ax.plot(ranks, mean, marker="o", color=color, label=label)
    ax.fill_between(ranks, lo, hi, color=color, alpha=0.15, lw=0)
    return {"rank": ranks, "mean": mean, "ci_low": lo, "ci_high": hi}


def fig_main_curves(F, summary, wins, tag="main"):
    if not summary:
        return
    plt = _plt()
    for key, label in (("acc_final", "ACC: mean accuracy over all tasks after the sequence (%)"),
                       ("bwt", "BWT (pp; negative = forgetting)"),
                       ("plasticity", "mean accuracy right after learning each task (%)"),
                       ("forgetting_max", "forgetting F (pp)"),
                       ("probe_bwt", "representation BWT from linear probes (pp)")):
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
        data = {}
        for ax, model in zip(axes, C.MODEL_ORDER):
            s = {v["rank"]: v for v in summary.values() if v["model"] == model}
            if not s or s[min(s)][key]["mean"] is None:
                ax.set_visible(False)
                continue
            ranks = sorted(s)
            data[model] = _band(ax, ranks, s, key, MODEL_COLOR[model], C.MODELS[model].label)
            w = (wins.get(model) or {}).get("primary", {})
            for lo, hi in w.get("acc_confident_intervals", []) if key == "acc_final" else \
                    w.get("learning_and_retention_intervals", []):
                ax.axvspan(lo / 1.3, hi * 1.3, color=GRID, alpha=0.6, lw=0)
            if key == "bwt":
                ax.axhline(-PRIMARY_DELTA, color=INK2, lw=0.8, ls="--")
            _rank_axis(ax, ranks)
            ax.set_title(C.MODELS[model].label)
        axes[0].set_ylabel(label)
        F.save(fig, f"{tag}_{key}_vs_rank", data,
               f"{label} vs rank at each rank's selected LR (mean and 95% bootstrap CI over seeds). Shaded: "
               "primary (δ=2pp) window.")
    # cross-model overlays, raw r and r/d
    for key, label in (("acc_final", "ACC (%)"), ("bwt", "BWT (pp)")):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        data = {}
        for model in C.MODEL_ORDER:
            s = {v["rank"]: v for v in summary.values() if v["model"] == model}
            if not s:
                continue
            ranks = sorted(s)
            y = [s[r][key]["mean"] for r in ranks]
            axes[0].plot(ranks, y, marker="o", color=MODEL_COLOR[model], label=C.MODELS[model].label)
            axes[1].plot([r / C.MODELS[model].d_model for r in ranks], y, marker="o", color=MODEL_COLOR[model],
                         label=C.MODELS[model].label)
            data[model] = {"rank": ranks, "r_over_d": [r / C.MODELS[model].d_model for r in ranks], "mean": y}
        allr = sorted({v["rank"] for v in summary.values()})
        _rank_axis(axes[0], allr)
        axes[1].set_xscale("log", base=2)
        axes[1].set_xlabel("r / d_model (log scale)")
        for ax in axes:
            ax.set_ylabel(label)
        axes[0].legend(fontsize=7)
        F.save(fig, f"{tag}_{key}_cross_model", data, f"{label} vs rank and vs rank/d_model for all models.")


def fig_per_task(F, summary):
    if not summary:
        return
    plt = _plt()
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), sharex=False)
    data = {}
    for col, model in enumerate(C.MODEL_ORDER):
        s = {v["rank"]: v for v in summary.values() if v["model"] == model}
        if not s:
            axes[0, col].set_visible(False)
            axes[1, col].set_visible(False)
            continue
        ranks = sorted(s)
        tasks = s[ranks[0]]["tasks"]
        data[model] = {}
        for j, t in enumerate(tasks):
            d1 = _band(axes[0, col], ranks, s, f"diag_{t}", SERIES[j], TASK_LABEL[t])
            d2 = _band(axes[1, col], ranks, s, f"final_{t}", SERIES[j], TASK_LABEL[t])
            data[model][t] = {"diag": d1, "final": d2}
        axes[0, col].set_title(C.MODELS[model].label)
        _rank_axis(axes[0, col], ranks)
        _rank_axis(axes[1, col], ranks)
    axes[0, 0].set_ylabel("accuracy right after learning (%)")
    axes[1, 0].set_ylabel("accuracy after the whole sequence (%)")
    axes[0, 0].legend(fontsize=7)
    F.save(fig, "main_per_task_learning_and_final", data,
           "Per-task learning (top: A_jj) and end-of-sequence accuracy (bottom: A_Tj) vs rank; mean ± 95% CI.")


def fig_matrices(F, summary):
    if not summary:
        return
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    for model in C.MODEL_ORDER:
        s = {v["rank"]: v for v in summary.values() if v["model"] == model}
        if not s:
            continue
        ranks = sorted(s)
        fig, axes = plt.subplots(1, len(ranks), figsize=(1.7 * len(ranks) + 1, 2.4))
        axes = np.atleast_1d(axes)
        data = {}
        for ax, r in zip(axes, ranks):
            A = np.array([[np.nan if x is None else x for x in row] for row in s[r]["A_mean"]])
            ax.imshow(A, vmin=30, vmax=100, cmap=cmap)
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    if not np.isnan(A[i, j]):
                        ax.text(j, i, f"{A[i, j]:.0f}", ha="center", va="center", fontsize=7,
                                color=INK if A[i, j] < 75 else SURFACE)
            ax.set_xticks(range(3))
            ax.set_xticklabels([TASK_LABEL[t] for t in s[r]["tasks"]], fontsize=6, rotation=45)
            ax.set_yticks(range(3))
            ax.set_yticklabels([f"after {TASK_LABEL[t]}" for t in s[r]["tasks"]], fontsize=6)
            ax.grid(False)
            ax.set_title(f"r={r}", fontsize=8)
            data[r] = s[r]["A_mean"]
        F.save(fig, f"accuracy_matrices_{model}", data,
               f"{C.MODELS[model].label}: mean accuracy matrix per rank (row = after training task i).")


def fig_tradeoff(F, summary):
    if not summary:
        return
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    cmap = LinearSegmentedColormap.from_list("seq", SEQ[1:])
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    data = {}
    for ax, model in zip(axes, C.MODEL_ORDER):
        s = {v["rank"]: v for v in summary.values() if v["model"] == model}
        if not s:
            ax.set_visible(False)
            continue
        ranks = sorted(s)
        x = [s[r]["plasticity"]["mean"] for r in ranks]
        y = [s[r]["bwt"]["mean"] for r in ranks]
        sc = ax.scatter(x, y, c=ranks, cmap=cmap, norm=LogNorm(min(ranks), max(ranks)), s=40,
                        edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.plot(x, y, color=GRID, lw=1, zorder=1)
        for r, xi, yi in zip(ranks, x, y):
            if r in (ranks[0], ranks[-1], 16):
                ax.annotate(f"r={r}", (xi, yi), fontsize=7, color=INK2, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("plasticity: mean A_jj (%)")
        ax.set_title(C.MODELS[model].label)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="rank")
        data[model] = {"rank": ranks, "plasticity": x, "bwt": y}
    axes[0].set_ylabel("BWT (pp)")
    F.save(fig, "learning_forgetting_tradeoff", data,
           "Learning vs forgetting per rank (seed means, selected LR); upper right is better on both.")


def fig_dynamics(F, summary):
    if not summary:
        return
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    cmap = LinearSegmentedColormap.from_list("seq", SEQ[1:])
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    data = {}
    for col, model in enumerate(C.MODEL_ORDER):
        s = {v["rank"]: v for v in summary.values() if v["model"] == model}
        if not s:
            axes[0, col].set_visible(False)
            axes[1, col].set_visible(False)
            continue
        ranks = sorted(s)
        norm = LogNorm(min(ranks), max(ranks))
        data[model] = {}
        for row, j in enumerate((0, 1)):
            ax = axes[row, col]
            t = s[ranks[0]]["tasks"][j]
            for r in ranks:
                traj = [s[r]["A_mean"][i][j] for i in range(j, 3)]
                ax.plot(list(range(j + 1, 4)), traj, color=cmap(norm(r)), marker="o", lw=1.5)
                data[model].setdefault(t, {})[r] = traj
            ax.set_xticks([1, 2, 3])
            ax.set_xlabel("after task #")
            ax.set_title(f"{C.MODELS[model].label}: {TASK_LABEL[t]}", fontsize=9)
        import matplotlib.cm as cm
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[:, col], fraction=0.05, label="rank")
    axes[0, 0].set_ylabel("accuracy on task 1 (%)")
    axes[1, 0].set_ylabel("accuracy on task 2 (%)")
    F.save(fig, "forgetting_dynamics", data,
           "Accuracy of earlier tasks across later boundaries, one line per rank (seed means).")


def fig_spectral(F, summary, assoc_rows):
    if not summary:
        return
    plt = _plt()
    keys = (("rel_fro", "relative update norm ||ΔW||/||W0||"), ("entropy_rank", "entropy effective rank of ΔW"),
            ("intruder_frac_0.9", "intruder fraction (ε=0.9, k=10)"), ("subspace_dist", "top-10 subspace distance"))
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    data = {}
    for ax, (key, label) in zip(axes, keys):
        for model in C.MODEL_ORDER:
            s = {v["rank"]: v for v in summary.values() if v["model"] == model}
            if not s or s[min(s)][key]["mean"] is None:
                continue
            ranks = sorted(s)
            y = [s[r][key]["mean"] for r in ranks]
            ax.plot(ranks, y, marker="o", color=MODEL_COLOR[model], label=C.MODELS[model].label)
            data.setdefault(key, {})[model] = {"rank": ranks, "mean": y}
            _rank_axis(ax, ranks)
        ax.set_title(label, fontsize=9)
        if key in ("rel_fro", "entropy_rank"):
            ax.set_yscale("log")
    axes[0].legend(fontsize=7)
    F.save(fig, "spectral_vs_rank", data, "Final-boundary update diagnostics vs rank (mean over matrices, then seeds).")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    data = {}
    for ax, model in zip(axes, C.MODEL_ORDER):
        rs = [r for r in assoc_rows if r["model"] == model]
        if not rs:
            ax.set_visible(False)
            continue
        x = [r["intruder_frac_0.9"] for r in rs]
        y = [-100 * r["bwt"] for r in rs]
        ax.scatter(x, y, color=MODEL_COLOR[model], s=18, edgecolors=SURFACE, linewidths=1)
        ax.set_xlabel("intruder fraction (ε=0.9)")
        ax.set_title(C.MODELS[model].label)
        data[model] = {"intruder_frac_0.9": x, "forgetting_pp": y, "rank": [r["rank"] for r in rs]}
    axes[0].set_ylabel("forgetting −BWT (pp)")
    F.save(fig, "intruders_vs_forgetting", data, "Run-level intruder fraction vs forgetting (main runs).")


def fig_probe(F, summary):
    plt = _plt()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    data, any_ = {}, False
    for ax, model in zip(axes, C.MODEL_ORDER):
        s = {v["rank"]: v for v in summary.values() if v["model"] == model}
        if not s or s[min(s)]["probe_bwt"]["mean"] is None:
            ax.set_visible(False)
            continue
        any_ = True
        ranks = sorted(s)
        d1 = _band(ax, ranks, s, "bwt", SERIES[0], "task-head BWT")
        d2 = _band(ax, ranks, s, "probe_bwt", SERIES[1], "linear-probe BWT")
        _rank_axis(ax, ranks)
        ax.set_title(C.MODELS[model].label)
        data[model] = {"head_bwt": d1, "probe_bwt": d2}
    if any_:
        axes[0].set_ylabel("BWT (pp)")
        axes[0].legend(fontsize=7)
        F.save(fig, "probe_vs_head_bwt", data,
               "Forgetting measured through the frozen task heads vs re-fitted linear probes: the gap is "
               "head/representation misalignment rather than lost information.")


def fig_orders(F, rows, sel_lr):
    plt = _plt()
    data = {}
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    any_ = False
    for col, model in enumerate(C.MODEL_ORDER):
        for row, key in enumerate(("acc_final", "bwt")):
            ax = axes[row, col]
            for oi, order in enumerate(("O1", "O2", "O3")):
                kind = "main" if order == "O1" else "order"
                rs = [r for r in rows if r["model"] == model and r["order"] == order and r["kind"] == kind
                      and r["status"] == "completed" and r["rank"] in C.ORDER_RANKS]
                if not rs:
                    continue
                any_ = True
                ranks = sorted({r["rank"] for r in rs})
                y = [100 * np.mean([r[key] for r in rs if r["rank"] == k]) for k in ranks]
                ax.plot(ranks, y, marker="o", color=SERIES[oi], label="→".join(TASK_LABEL[t] for t in C.ORDERS[order]))
                data.setdefault(model, {}).setdefault(key, {})[order] = {"rank": ranks, "mean": y}
                _rank_axis(ax, ranks)
            if row == 0:
                ax.set_title(C.MODELS[model].label)
        axes[0, 0].set_ylabel("ACC (%)")
        axes[1, 0].set_ylabel("BWT (pp)")
    if any_:
        axes[0, 0].legend(fontsize=6)
        F.save(fig, "task_order_robustness", data, "ACC and BWT vs rank for the three cyclic task orders.")


def fig_seeds(F, rows):
    plt = _plt()
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    data, any_ = {}, False
    for col, model in enumerate(C.MODEL_ORDER):
        rs = [r for r in rows if r["model"] == model and r["kind"] == "main" and r["status"] == "completed"]
        if not rs:
            axes[0, col].set_visible(False)
            axes[1, col].set_visible(False)
            continue
        any_ = True
        for row, key in enumerate(("acc_final", "bwt")):
            ax = axes[row, col]
            for seed_i, seed in enumerate(sorted({r["seed"] for r in rs})):
                pts = sorted((r["rank"], 100 * r[key]) for r in rs if r["seed"] == seed)
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", lw=1, ms=3,
                        color=SERIES[seed_i % len(SERIES)], alpha=0.85, label=f"seed {seed}")
                data.setdefault(model, {}).setdefault(key, {})[seed] = pts
            _rank_axis(ax, sorted({r["rank"] for r in rs}))
            if row == 0:
                ax.set_title(C.MODELS[model].label)
        axes[0, 0].set_ylabel("ACC (%)")
        axes[1, 0].set_ylabel("BWT (pp)")
    if any_:
        axes[0, 0].legend(fontsize=6)
        F.save(fig, "seed_variability", data, "Every seed's ACC and BWT vs rank (main runs).")


def fig_loss_curves(F, rows, root):
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    cmap = LinearSegmentedColormap.from_list("seq", SEQ[1:])
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    data, any_ = {}, False
    for ax, model in zip(axes, C.MODEL_ORDER):
        rs = [r for r in rows if r["model"] == model and r["kind"] == "main" and r["seed"] == C.MAIN_SEEDS[0]
              and r["status"] == "completed"]
        if not rs:
            ax.set_visible(False)
            continue
        any_ = True
        ranks = sorted({r["rank"] for r in rs})
        norm = LogNorm(min(ranks), max(ranks))
        for r in sorted(rs, key=lambda x: x["rank"]):
            p = Path(r["dir"]) / "train_log.jsonl.gz"
            if not p.exists():
                continue
            with gzip.open(p, "rt") as fh:
                L = [json.loads(x)["loss"] for x in fh]
            k = 25
            sm = np.convolve(L, np.ones(k) / k, mode="valid")
            ax.plot(np.arange(len(sm)), sm, color=cmap(norm(r["rank"])), lw=1)
            data.setdefault(model, {})[r["rank"]] = sm[::5].tolist()
        ax.set_yscale("log")
        ax.set_xlabel("optimizer step (3 tasks)")
        ax.set_title(C.MODELS[model].label)
        import matplotlib.cm as cm
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, label="rank")
    if any_:
        axes[0].set_ylabel("training loss (25-step mean)")
        F.save(fig, "training_loss_curves", data, f"Training loss over the whole sequence, seed {C.MAIN_SEEDS[0]}.")


def fig_v3_vs_v4(F, v3, summary):
    if not v3 or not summary:
        return
    plt = _plt()
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    data = {}
    for col, model in enumerate(C.MODEL_ORDER):
        for row, key in enumerate(("acc_final", "bwt")):
            ax = axes[row, col]
            v3m = {}
            for r in v3:
                if r["model"] == model:
                    v3m.setdefault(r["rank"], []).append(100 * r[key])
            if v3m:
                rk = sorted(v3m)
                ax.plot(rk, [np.mean(v3m[k]) for k in rk], marker="o", color=SERIES[1], label="v3: α=2r, fixed LR")
                data.setdefault(model, {}).setdefault(key, {})["v3"] = {"rank": rk, "mean": [np.mean(v3m[k]) for k in rk]}
            s = {v["rank"]: v for v in summary.values() if v["model"] == model}
            if s:
                rk = sorted(s)
                ax.plot(rk, [s[k][key]["mean"] for k in rk], marker="o", color=SERIES[0],
                        label="v4: α=32, per-rank LR")
                data.setdefault(model, {}).setdefault(key, {})["v4"] = {"rank": rk, "mean": [s[k][key]["mean"] for k in rk]}
            allr = sorted(set(v3m) | set(s))
            if allr:
                _rank_axis(ax, allr)
            if row == 0:
                ax.set_title(C.MODELS[model].label)
    axes[0, 0].set_ylabel("ACC (%)")
    axes[1, 0].set_ylabel("BWT (pp)")
    axes[0, 0].legend(fontsize=7)
    F.save(fig, "v3_vs_v4", data, "Fixed-LR v3 (confounded) vs per-rank-LR v4 on the same models, data and order.")


def fig_window_map(F, wins):
    if not wins:
        return
    plt = _plt()
    crits = (("learning_and_retention", "learning + retention"), ("acc_point", "ACC within δ (point)"),
             ("acc_confident", "ACC within δ (95% CI)"))
    fig, axes = plt.subplots(len(crits), 1, figsize=(9, 2.2 * len(crits) * max(1, len(wins)) / 2 + 1),
                             squeeze=False)
    data = {}
    for ci, (key, label) in enumerate(crits):
        ax = axes[ci, 0]
        ylabels, y = [], 0
        for model in C.MODEL_ORDER:
            w = wins.get(model)
            if not w:
                continue
            ranks = w["ranks_tested"]
            for delta in DELTAS:
                members = w["definitions"][f"delta={delta:g}"][key]
                for r in ranks:
                    ax.add_patch(plt.Rectangle((math.log2(r) - 0.42, y - 0.4), 0.84, 0.8,
                                               color=MODEL_COLOR[model] if r in members else GRID,
                                               alpha=1.0 if r in members else 0.5, lw=0))
                data.setdefault(key, {}).setdefault(model, {})[f"delta={delta:g}"] = members
                ylabels.append(f"{C.MODELS[model].label} δ={delta:g}")
                y += 1
            y += 0.5
        allr = sorted({r for w in wins.values() for r in w["ranks_tested"]})
        ax.set_xlim(math.log2(min(allr)) - 0.6, math.log2(max(allr)) + 0.6)
        ax.set_ylim(-0.7, y)
        ax.set_xticks([math.log2(r) for r in allr])
        ax.set_xticklabels([str(r) for r in allr], fontsize=7)
        ticks = [i for i in range(len(ylabels))]
        pos, k = [], 0
        for model in C.MODEL_ORDER:
            if model in wins:
                pos.extend([k + i for i in range(len(DELTAS))])
                k += len(DELTAS) + 0.5
        ax.set_yticks(pos[:len(ticks)])
        ax.set_yticklabels(ylabels, fontsize=6)
        ax.invert_yaxis()
        ax.grid(False)
        ax.set_title(f"Feasible ranks: {label}", fontsize=9)
    axes[-1, 0].set_xlabel("LoRA rank r")
    fig.tight_layout(h_pad=2.0)
    F.save(fig, "feasible_window_map", data,
           "Which ranks satisfy each pre-registered window criterion, per model and tolerance δ (pp); "
           "filled = feasible.")


def fig_paired_forest(F, tests):
    if not tests:
        return
    plt = _plt()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharex=True)
    data = {}
    for ax, model in zip(axes, C.MODEL_ORDER):
        t = tests.get(model)
        if not t:
            ax.set_visible(False)
            continue
        rows = t["metrics"]["acc_final"]
        ranks = sorted(rows)
        for i, r in enumerate(ranks):
            v = rows[r]
            lo, hi = v["ci95"]
            ax.plot([lo if lo is not None else v["mean_diff_pp"], hi if hi is not None else v["mean_diff_pp"]],
                    [i, i], color=MODEL_COLOR[model], lw=2)
            ax.plot(v["mean_diff_pp"], i, "o", color=MODEL_COLOR[model], ms=6, mec=SURFACE, mew=1.5)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.axvline(-PRIMARY_DELTA, color=INK2, lw=0.8, ls="--")
        ax.set_yticks(range(len(ranks)))
        ax.set_yticklabels([f"r={r}" for r in ranks], fontsize=7)
        ax.set_xlabel(f"ACC(r) − ACC(r={t['reference_rank']}) (pp)")
        ax.set_title(C.MODELS[model].label)
        data[model] = {"reference_rank": t["reference_rank"], "rows": rows}
    F.save(fig, "paired_acc_vs_best_rank", data,
           "Paired (same-seed) ACC difference of every rank from the best rank, mean and 95% bootstrap CI; "
           "dashed line = −2 pp non-inferiority margin.")


def fig_bwt_components(F, summary):
    if not summary:
        return
    plt = _plt()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    data = {}
    for ax, model in zip(axes, C.MODEL_ORDER):
        s = {v["rank"]: v for v in summary.values() if v["model"] == model}
        if not s:
            ax.set_visible(False)
            continue
        ranks = sorted(s)
        tasks = s[ranks[0]]["tasks"][:-1]
        data[model] = {}
        for j, t in enumerate(tasks):
            data[model][t] = _band(ax, ranks, s, f"bwt_{t}", SERIES[j], f"{TASK_LABEL[t]}: final − learned")
        ax.axhline(0, color=INK2, lw=0.8)
        _rank_axis(ax, ranks)
        ax.set_title(C.MODELS[model].label)
    axes[0].set_ylabel("per-task backward transfer (pp)")
    axes[0].legend(fontsize=7)
    F.save(fig, "bwt_components_vs_rank", data,
           "Forgetting of each earlier task separately (A_Tj − A_jj) vs rank; mean and 95% CI over seeds.")


# ---------------------------------------------------------------------------
# extension cohorts (docs/PROTOCOL_V4_EXT.md)
# ---------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def stability(rows, sel):
    """E1: failure rates per (model, rank, LR) pooling main seeds with the fresh stability seeds."""
    from scipy.stats import fisher_exact, norm
    out = {}
    for model in ("granite3.3-2b", "qwen2.5-3b"):
        rs = [r for r in rows if r["model"] == model and r["status"] == "completed" and r["order"] == "O1"
              and r["kind"] in ("main", "stab") and r["rank"] in C.STAB_RANKS]
        if not any(r["kind"] == "stab" for r in rs):
            continue
        ctrl = [100 * r["acc_final"] for r in rs if r["rank"] == 16
                and math.isclose(r["lr"], sel.get((model, 16), {}).get("selected_lr", -1), rel_tol=1e-6)]
        if not ctrl:
            continue
        thr = float(np.median(ctrl)) - 10.0

        def failed(r):
            if 100 * r["acc_final"] < thr:
                return True
            return any(100 * r[f"diag_{t}"] < 100 * (r["majority"].get(t) or 0) + 5 for t in r["tasks"])
        res = {"control_median_acc": float(np.median(ctrl)), "acc_threshold": thr, "cells": {}}
        for rank in C.STAB_RANKS:
            s_lr = sel.get((model, rank), {}).get("selected_lr")
            if s_lr is None:
                continue
            for tag, lr in (("selected", s_lr), ("half", s_lr / 2)):
                cell = [r for r in rs if r["rank"] == rank and math.isclose(r["lr"], lr, rel_tol=1e-6)]
                if not cell:
                    continue
                k = sum(failed(r) for r in cell)
                ok = [r for r in cell if not failed(r)]
                lo, hi = wilson(k, len(cell))
                res["cells"][f"{rank}|{tag}"] = {
                    "rank": rank, "lr": lr, "lr_tag": tag, "n": len(cell), "failures": k, "rate": k / len(cell),
                    "wilson95": [lo, hi], "failed_seeds": sorted(r["seed"] for r in cell if failed(r)),
                    "acc_ok": summarize([100 * r["acc_final"] for r in ok]),
                    "bwt_ok": summarize([100 * r["bwt"] for r in ok]),
                    "acc_all": summarize([100 * r["acc_final"] for r in cell])}
        # Cochran-Armitage trend over log2(rank) at the selected LR
        cs = [v for v in res["cells"].values() if v["lr_tag"] == "selected"]
        if len(cs) >= 3:
            x = np.array([math.log2(v["rank"]) for v in cs])
            n = np.array([v["n"] for v in cs], float)
            k = np.array([v["failures"] for v in cs], float)
            p = k.sum() / n.sum()
            if 0 < p < 1:
                tstat = float((k * (x - (n * x).sum() / n.sum())).sum())
                var = p * (1 - p) * float((n * (x - (n * x).sum() / n.sum()) ** 2).sum())
                z = tstat / math.sqrt(var)
                res["trend_selected_lr"] = {"z": z, "p_two_sided": float(2 * norm.sf(abs(z)))}
            else:
                res["trend_selected_lr"] = {"z": None, "p_two_sided": 1.0, "note": "no variation in failures"}
        hi_sel = [v for v in res["cells"].values() if v["rank"] >= 64 and v["lr_tag"] == "selected"]
        hi_half = [v for v in res["cells"].values() if v["rank"] >= 64 and v["lr_tag"] == "half"]
        if hi_sel and hi_half:
            a, b = sum(v["failures"] for v in hi_sel), sum(v["n"] for v in hi_sel)
            c, d = sum(v["failures"] for v in hi_half), sum(v["n"] for v in hi_half)
            odds, pv = fisher_exact([[a, b - a], [c, d - c]])
            res["lr_effect_rank_ge64"] = {"selected": [a, b], "half": [c, d], "fisher_p": float(pv)}
        feasible = []
        for rank in C.STAB_RANKS:
            for tag in ("selected", "half"):
                v = res["cells"].get(f"{rank}|{tag}")
                c16 = res["cells"].get(f"16|{tag}")
                if v and c16 and v["rate"] <= 0.1 and v["acc_ok"]["mean"] is not None \
                        and v["acc_ok"]["mean"] >= c16["acc_ok"]["mean"] - 2:
                    feasible.append({"rank": rank, "lr_tag": tag, "lr": v["lr"]})
        res["reinstated_feasible"] = feasible
        out[model] = res
    return out


def all_linear(rows):
    """E2: all-linear summary plus paired (same seed) all-linear minus q/v differences."""
    s_al = main_summary(rows, kind="main_al")
    if not s_al:
        return {}, {}
    w_al = windows(s_al)
    qv = {(r["model"], r["rank"], r["seed"]): r for r in rows if r["kind"] == "main" and r["status"] == "completed"}
    paired = {}
    for model in C.MODEL_ORDER:
        ps = {}
        for r in sorted({x["rank"] for x in rows if x["kind"] == "main_al" and x["model"] == model}):
            for metric in ("acc_final", "plasticity", "bwt", "probe_bwt"):
                d = [100 * (a[metric] - qv[(model, r, a["seed"])][metric]) for a in rows
                     if a["kind"] == "main_al" and a["model"] == model and a["rank"] == r and a["status"] == "completed"
                     and (model, r, a["seed"]) in qv and a.get(metric) is not None
                     and qv[(model, r, a["seed"])].get(metric) is not None]
                m, lo, hi = boot_ci(d)
                ps.setdefault(r, {})[metric] = {"n": len(d), "mean_diff_pp": m, "ci95": [lo, hi],
                                                "p_signflip": signflip_p(d) if d else None}
        if ps:
            adj = holm({r: v["acc_final"]["p_signflip"] for r, v in ps.items()})
            for r in ps:
                ps[r]["acc_final"]["p_holm"] = adj.get(r)
            paired[model] = ps
    return {"summary": s_al, "windows": w_al}, paired


def fig_stability(F, stab):
    if not stab:
        return
    plt = _plt()
    fig, axes = plt.subplots(2, len(stab), figsize=(6.5 * len(stab), 6.4), squeeze=False)
    data = {}
    for col, (model, res) in enumerate(stab.items()):
        for row, key in enumerate(("rate", "acc_ok")):
            ax = axes[row, col]
            for li, tag in enumerate(("selected", "half")):
                cells = sorted((v for v in res["cells"].values() if v["lr_tag"] == tag), key=lambda v: v["rank"])
                if not cells:
                    continue
                x = [v["rank"] for v in cells]
                if key == "rate":
                    y = [100 * v["rate"] for v in cells]
                    lo = [100 * v["wilson95"][0] for v in cells]
                    hi = [100 * v["wilson95"][1] for v in cells]
                else:
                    y = [v["acc_ok"]["mean"] for v in cells]
                    lo = [v["acc_ok"]["ci95"][0] if v["acc_ok"]["ci95"][0] is not None else v["acc_ok"]["mean"] for v in cells]
                    hi = [v["acc_ok"]["ci95"][1] if v["acc_ok"]["ci95"][1] is not None else v["acc_ok"]["mean"] for v in cells]
                lab = "selected LR" if tag == "selected" else "half the selected LR"
                ax.plot(x, y, marker="o", color=SERIES[li], label=lab)
                ax.fill_between(x, lo, hi, color=SERIES[li], alpha=0.15, lw=0)
                data.setdefault(model, {}).setdefault(key, {})[tag] = {"rank": x, "value": y, "low": lo, "high": hi}
                _rank_axis(ax, x)
            ax.set_title(C.MODELS[model].label)
        axes[0, col].set_ylabel("failed runs (%) with Wilson 95% CI")
        axes[1, col].set_ylabel("ACC of non-failed runs (%)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(h_pad=2.0)
    F.save(fig, "ext_stability_failure_rate", data,
           "E1: share of runs that failed (pre-registered definition) per rank at the selected LR (n=10) and "
           "at half of it (n=5), and ACC of the runs that did not fail.")


def fig_all_linear(F, al_sum, rows):
    s_al = al_sum.get("summary") if al_sum else None
    if not s_al:
        return
    plt = _plt()
    s_qv = main_summary(rows)
    models = [m for m in C.MODEL_ORDER if any(v["model"] == m for v in s_al.values())]
    fig, axes = plt.subplots(3, len(models), figsize=(6.5 * len(models), 9), squeeze=False)
    data = {}
    for col, model in enumerate(models):
        a = {v["rank"]: v for v in s_al.values() if v["model"] == model}
        q = {v["rank"]: v for v in s_qv.values() if v["model"] == model}
        for row, key in enumerate(("acc_final", "plasticity", "bwt")):
            ax = axes[row, col]
            data.setdefault(model, {})[key] = {
                "q/v": _band(ax, sorted(q), q, key, SERIES[0], "q/v LoRA"),
                "all-linear": _band(ax, sorted(a), a, key, SERIES[1], "all-linear LoRA")}
            _rank_axis(ax, sorted(set(q) | set(a)))
            if row == 0:
                ax.set_title(C.MODELS[model].label)
        axes[0, col].set_ylabel("ACC (%)")
        axes[1, col].set_ylabel("plasticity (%)")
        axes[2, col].set_ylabel("BWT (pp)")
    axes[0, 0].legend(fontsize=7)
    F.save(fig, "ext_all_linear_vs_qv", data,
           "E2: all-linear LoRA vs q/v LoRA at each rank's own selected LR (mean and 95% CI over the same 5 seeds).")


def fig_cost(F, rows):
    plt = _plt()
    by = {}
    for r in rows:
        if r.get("wall_s"):
            by.setdefault((r["model"], r["kind"]), []).append(r["wall_s"])
    if not by:
        return
    fig, ax = plt.subplots(figsize=(8, 3.4))
    kinds = sorted({k for _, k in by})
    data = {}
    width = 0.8 / max(1, len(kinds))
    for ki, kind in enumerate(kinds):
        xs, ys = [], []
        for mi, model in enumerate(C.MODEL_ORDER):
            v = by.get((model, kind))
            if v:
                xs.append(mi + ki * width)
                ys.append(sum(v) / 3600)
                data.setdefault(kind, {})[model] = {"gpu_hours": sum(v) / 3600, "n": len(v),
                                                    "median_min": float(np.median(v) / 60)}
        ax.bar(xs, ys, width=width * 0.9, color=SERIES[ki % len(SERIES)], label=kind)
    ax.set_xticks([i + width * (len(kinds) - 1) / 2 for i in range(len(C.MODEL_ORDER))])
    ax.set_xticklabels([C.MODELS[m].label for m in C.MODEL_ORDER])
    ax.set_ylabel("GPU hours (job wall time)")
    ax.legend(fontsize=7)
    F.save(fig, "compute_by_model_and_stage", data, "GPU hours by model and job kind.")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def write_report(out, rows, summary, wins, tests, trend, sel, assoc, figs):
    L = ["# frw-v4 results (auto-generated)", ""]
    done = [r for r in rows if r["status"] == "completed"]
    L.append(f"Runs loaded: {len(rows)} (completed {len(done)}; diverged "
             f"{sum(r['status'] == 'diverged' for r in rows)}).")
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    L.append("By kind: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())) + ".")
    L.append("")
    L.append("## Selected learning rates (training holdout, pre-registered rule)")
    L.append("")
    L.append("| model | " + " | ".join(f"r={r}" for r in C.RANKS) + " |")
    L.append("|---|" + "---|" * len(C.RANKS))
    for model in C.MODEL_ORDER:
        L.append(f"| {C.MODELS[model].label} | " + " | ".join(
            (f"{sel[(model, r)]['selected_lr']:.1e}" if (model, r) in sel else "–") for r in C.RANKS) + " |")
    L.append("")
    L.append("## Main results (official validation, selected LR, mean ± SD over seeds)")
    for model in C.MODEL_ORDER:
        s = {v["rank"]: v for v in summary.values() if v["model"] == model}
        if not s:
            continue
        L.append("")
        L.append(f"### {C.MODELS[model].label}")
        L.append("")
        tasks = s[min(s)]["tasks"]
        L.append("| r | LR | n | ACC | plasticity | BWT | F | " + " | ".join(
            f"{TASK_LABEL[t]} learn / final" for t in tasks) + " | probe BWT |")
        L.append("|---|---|---|---|---|---|---|" + "---|" * len(tasks) + "---|")

        def f(x):
            return "–" if x["mean"] is None else (f"{x['mean']:.1f}" + (f" ± {x['sd']:.1f}" if x["sd"] is not None else ""))
        for r in sorted(s):
            v = s[r]
            L.append(f"| {r} | {v['lr']:.1e} | {v['acc_final']['n']} | {f(v['acc_final'])} | {f(v['plasticity'])} | "
                     f"{f(v['bwt'])} | {f(v['forgetting_max'])} | " +
                     " | ".join(f"{v[f'diag_{t}']['mean']:.1f} / {v[f'final_{t}']['mean']:.1f}" for t in tasks) +
                     f" | {f(v['probe_bwt'])} |")
        w = wins.get(model)
        if w:
            p = w["primary"]
            L.append("")
            L.append(f"Primary (δ=2pp) windows — learning+retention: {p['learning_and_retention_intervals'] or 'empty'}; "
                     f"ACC point: {p['acc_point_intervals'] or 'empty'}; ACC confident: "
                     f"{p['acc_confident_intervals'] or 'empty'}; best-ACC rank: {w['best_acc_rank']}.")
    L.append("")
    L.append("## Files")
    L.append("")
    L.append("All aggregates are JSON/CSV in this folder; every figure in figures/ has a same-name .json "
             "with the plotted numbers.")
    L.append("")
    for fi in figs:
        L.append(f"- figures/{fi['name']}.png — {fi['caption']}")
    (out / "report.md").write_text("\n".join(L) + "\n")


def write_html(out: Path, figs):
    """Self-contained HTML (figures inlined) next to report.md; opens offline in any browser."""
    import base64
    import html
    md = (out / "report.md").read_text()
    blocks, table = [], []

    def flush_table():
        if not table:
            return
        rows_ = [r.strip().strip("|").split("|") for r in table if not set(r.replace("|", "").strip()) <= set("-: ")]
        head, body = rows_[0], rows_[1:]
        h = "<table><thead><tr>" + "".join(f"<th>{html.escape(c.strip())}</th>" for c in head) + "</tr></thead><tbody>"
        h += "".join("<tr>" + "".join(f"<td>{html.escape(c.strip())}</td>" for c in r) + "</tr>" for r in body)
        blocks.append(h + "</tbody></table>")
        table.clear()
    for line in md.splitlines():
        if line.startswith("|"):
            table.append(line)
            continue
        flush_table()
        if line.startswith("### "):
            blocks.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("## "):
            blocks.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("# "):
            blocks.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("- figures/"):
            continue
        elif line.strip():
            blocks.append(f"<p>{html.escape(line)}</p>")
    flush_table()
    figs_html = []
    for f in figs:
        png = out / "figures" / f"{f['name']}.png"
        if png.exists():
            b64 = base64.b64encode(png.read_bytes()).decode()
            figs_html.append(f"<figure><img alt=\"{html.escape(f['caption'])}\" src=\"data:image/png;base64,{b64}\">"
                             f"<figcaption><b>{html.escape(f['name'])}</b> — {html.escape(f['caption'])} "
                             f"(data: figures/{html.escape(f['name'])}.json)</figcaption></figure>")
    css = ("body{font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px 32px;background:#fcfcfb;color:#0b0b0b;"
           "max-width:1500px}table{border-collapse:collapse;margin:8px 0 20px;font-size:12px;display:block;"
           "overflow-x:auto}th,td{border-bottom:1px solid #e4e3df;padding:4px 8px;text-align:right;white-space:nowrap}"
           "th{color:#52514e}figure{margin:24px 0}img{max-width:100%;border:1px solid #e4e3df}"
           "figcaption{color:#52514e;font-size:12px}h1{font-size:22px}h2{font-size:18px;margin-top:32px}")
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><title>frw-v4 results</title><style>{css}</style>"
           f"</head><body>{''.join(blocks)}<h2>Figures</h2>{''.join(figs_html)}</body></html>")
    (out / "report.html").write_text(doc)


def write_csv(path, rows):
    keys = ["job_id", "kind", "model", "rank", "lr", "seed", "order", "split", "status", "acc_final", "plasticity",
            "bwt", "forgetting_max", "avg_incremental_acc", "probe_bwt", "probe_final_mean", "rel_fro",
            "entropy_rank", "stable_rank", "subspace_dist", "intruder_frac_0.9", "intruder_frac_0.7", "gpu",
            "wall_s", "trainable_params", "account"]
    extra = sorted({k for r in rows for k in r if k.startswith(("diag_", "final_", "bwt_"))})
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys + extra, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results-v4")
    ap.add_argument("--v3", default="Finetune-result")
    a = ap.parse_args(argv)
    root = Path(a.root)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    rows = load_runs(root)
    sel = load_selections(root)
    v3 = load_v3(Path(a.v3)) if Path(a.v3).exists() else []
    surface = tune_surface(rows)
    summary = main_summary(rows)
    wins = windows(summary)
    tests = paired_tests(rows, summary)
    trend = trends(rows)
    assoc = spectral_association(rows)
    orders = {o: main_summary(rows, kind="main" if o == "O1" else "order", order=o) for o in C.ORDERS}
    order_windows = {o: windows(s) for o, s in orders.items() if s}

    dump = lambda name, obj: (out / name).write_text(json.dumps(obj, indent=1, default=float))  # noqa: E731
    slim = [{k: v for k, v in r.items() if k not in ("spectral",)} for r in rows]
    dump("runs.json", slim)
    write_csv(out / "runs.csv", rows)
    dump("lr_surface.json", list(surface.values()))
    dump("selected_lr.json", [{**v, "model": m, "rank": r} for (m, r), v in sorted(sel.items())])
    dump("main_summary.json", summary)
    dump("windows.json", wins)
    dump("paired_tests.json", tests)
    dump("trends.json", trend)
    dump("spectral_association.json", assoc)
    dump("order_robustness.json", {"summaries": orders, "windows": order_windows})
    stab = stability(rows, sel)
    al_sum, al_paired = all_linear(rows)
    if stab:
        dump("ext_stability.json", stab)
    if al_sum:
        dump("ext_all_linear.json", {**al_sum, "paired_all_linear_minus_qv": al_paired,
                                     "selected_lr": [{**v, "model": m, "rank": r} for (m, r), v in
                                                     sorted(load_selections(root, "_al").items())]})

    F = Fig(out / "figures")
    assoc_rows = [r for r in rows if r["kind"] == "main" and r["status"] == "completed"
                  and r.get("intruder_frac_0.9") is not None]
    for fn, args in ((fig_v3_confound, (F, v3)), (fig_lr_surface, (F, surface, sel)), (fig_selected_lr, (F, sel)),
                     (fig_main_curves, (F, summary, wins)), (fig_per_task, (F, summary)),
                     (fig_matrices, (F, summary)), (fig_tradeoff, (F, summary)), (fig_dynamics, (F, summary)),
                     (fig_spectral, (F, summary, assoc_rows)), (fig_probe, (F, summary)),
                     (fig_orders, (F, rows, sel)), (fig_seeds, (F, rows)), (fig_loss_curves, (F, rows, root)),
                     (fig_v3_vs_v4, (F, v3, summary)), (fig_window_map, (F, wins)), (fig_paired_forest, (F, tests)),
                     (fig_bwt_components, (F, summary)), (fig_stability, (F, stab)),
                     (fig_all_linear, (F, al_sum, rows)), (fig_cost, (F, rows))):
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001 - one bad figure must not block the rest
            print(f"figure {fn.__name__} failed: {exc!r}")
    dump("figures_index.json", F.index)
    write_report(out, rows, summary, wins, tests, trend, sel, assoc, F.index)
    write_html(out, F.index)
    print(f"analysis: {len(rows)} runs, {len(summary)} main cells, {len(F.index)} figures -> {out}")


if __name__ == "__main__":
    main()
