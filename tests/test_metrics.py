import numpy as np
import pandas as pd
import pytest

from rank_suite import config, metrics


def test_bwt_is_diagonal_change_and_peak_forgetting_is_distinct():
    A = [[0.6, None, None], [0.8, 0.7, None], [0.5, 0.6, 0.9]]
    out = metrics.compute_run_metrics(A, ["rte", "mnli", "sst2"])
    assert out["bwt"] == pytest.approx(-0.1)
    assert out["peak_forgetting"] == pytest.approx(0.2)
    assert out["avg_final_row"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="Incomplete"):
        metrics.compute_run_metrics([[None]], ["rte"])


def window_data():
    model = "qwen2.5-1.5b"
    m = config.MODELS[model]
    rows = []
    for seed in config.DEFAULT_SEEDS:
        rows.append({"model": model, "track": "full", "rank": None, "lr": m.default_full_lr,
                     "seed": seed, "final_task_acc": .9, "bwt": -.1})
        for rank in config.DEFAULT_RANKS:
            rows.append({"model": model, "track": "lora", "rank": rank, "lr": m.default_lora_lr,
                         "seed": seed, "final_task_acc": .89,
                         "bwt": -.08 if rank == 16 else -.01})
    return pd.DataFrame(rows)


def test_windows_require_baseline_five_seeds_and_keep_disconnected_intervals():
    df = window_data()
    w = metrics.feasible_windows(metrics.config_summary(df), df)[0]
    assert w["window_intervals"] == [[4, 8], [32, 64]]
    assert w["status"] == "complete"
    for partial in (df[df.track != "full"], df[df.seed != 1024], df[df["rank"] != 64]):
        p = metrics.feasible_windows(metrics.config_summary(partial), partial)[0]
        assert p["status"] == "incomplete"
        assert p["window_ranks"] == []


def test_window_baseline_is_not_outcome_selected():
    df = window_data()
    other = df[df.track == "full"].copy()
    other["lr"] *= 2
    other["final_task_acc"] = 1.0
    df = pd.concat([df, other], ignore_index=True)
    w = metrics.feasible_windows(metrics.config_summary(df), df)[0]
    assert w["full_ft_reference_a33"] == pytest.approx(.9)


def test_paired_rank_differences_pair_the_same_seeds():
    df = window_data()
    df["avg_diagonal"] = .8
    out = metrics.paired_rank_comparisons(df)
    item = next(x for x in out if x["rank_a"] == 4 and x["rank_b"] == 16 and x["metric"] == "bwt")
    assert item["n_pairs"] == 5
    assert item["delta_b_minus_a_pp"] == pytest.approx(-7)
    assert np.isfinite(item["ci95_low_pp"])


def test_intruder_correlations_keep_lr_sensitivity_separate():
    model = "qwen2.5-1.5b"
    rows = [{"model": model, "track": "lora", "rank": rank, "lr": lr, "bwt": -i / 100,
             "intruder_total_final": i, "intruder_fraction_final": i / 560}
            for lr in [config.MODELS[model].default_lora_lr, .0001]
            for i, rank in enumerate([4, 8, 16])]
    out = metrics.intruder_correlations(pd.DataFrame(rows))
    assert next(r for r in out if r["scope"] == model)["n_runs"] == 3
    assert next(r for r in out if r["scope"] == "pooled_normalized_exploratory")["n_runs"] == 3


def test_analysis_recomputes_summary_from_raw_matrix(tmp_path):
    from rank_suite import utils
    r = {"mode": "main", "status": "completed", "protocol_sha256": "test",
         "run_identity": {"protocol": config.PROTOCOL_VERSION},
         "task_order": ["rte", "mnli", "sst2"],
         "A": [[.6, None, None], [.8, .7, None], [.5, .6, .9]], "metrics": {"bwt": 999}}
    utils.save_json(tmp_path / "runs" / "stub" / "result.json", r)
    assert metrics.load_results(tmp_path)[0]["metrics"]["bwt"] == pytest.approx(-.1)


def test_all_task_gates_reject_bwt_cancellation_and_require_healthy_reference():
    df = window_data()
    for task, diagonal, majority in [('rte', .8, .5), ('mnli', .7, 1/3), ('sst2', .9, .5)]:
        df[f'A_{task}_{task}'] = diagonal
        df[f'majority_{task}'] = majority
        df[f'A_sst2_{task}'] = diagonal - (.01 if task != 'sst2' else 0)
    # An apparently stable average can conceal a five-point MNLI loss.
    df.loc[df['rank'] == 4, 'A_sst2_rte'] = .85
    df.loc[df['rank'] == 4, 'A_sst2_mnli'] = .65
    sets = metrics.all_task_feasible_sets(df)
    primary = next(s for s in sets if s['tolerance_pp'] == 2)
    assert primary['status'] == 'complete'
    assert primary['window_ranks'] == [8, 16, 32, 64]
    assert primary['interval_supported_ranks'] == [8, 16, 32, 64]
    df.loc[df.track == 'full', 'A_rte_rte'] = .5
    assert all(s['window_ranks'] == [] and s['status'] == 'baseline_underlearning'
               for s in metrics.all_task_feasible_sets(df))


def test_calibrated_baseline_anchor_is_used_instead_of_old_nominal():
    df = window_data()
    calibrated = config.MODELS['qwen2.5-1.5b'].default_full_lr / 2
    df.loc[df.track == 'full', 'lr'] = calibrated
    df['declared_full_lr'] = calibrated
    window = metrics.feasible_windows(metrics.config_summary(df), df)[0]
    assert window['status'] == 'complete'
    assert window['full_ft_reference_lr'] == calibrated
