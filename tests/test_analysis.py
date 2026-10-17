import pytest

from rank_suite import config, metrics, utils
from rank_suite.analysis import run_analysis
from rank_suite.runner import build_grid


@pytest.mark.parametrize("models,seeds,pending", [(["qwen2.5-1.5b"], [42], True),
    (["qwen2.5-1.5b"], list(config.DEFAULT_SEEDS), False), (list(config.DEFAULT_MODEL_KEYS), [42], False)])
def test_report_handles_partial_design_and_completed_spectra(tmp_path, models, seeds, pending):
    spec = config.ExperimentSpec(models=models, seeds=seeds)
    for run in build_grid(spec):
        A = [[.75, None, None], [.74, .7, None], [.74, .69, .9]]
        count = run.rank // 4 if run.rank else 2
        payload = {"total": {str(e): count for e in (.6, .7, .8, .9)}, "mean_max_cosine": .85,
                   "n_matrices": 2 * config.MODELS[run.model].layers, "k": 10,
                   "by_matrix_type": {}, "by_layer": {}}
        result = {"run_id": run.run_id, "run_identity": run.identity, "protocol_sha256": "test",
                  "status": "completed", "model": run.model, "track": run.track, "mode": "main",
                  "rank": run.rank, "alpha": run.alpha, "lr": run.lr, "seed": run.seed,
                  "task_order": list(run.task_keys), "A": A, "metrics": metrics.compute_run_metrics(A, run.task_keys),
                  "intruder": {} if pending else {f"after_{k}": payload for k in config.TASK_KEYS},
                  "intruder_status": "pending_cpu" if pending else "completed",
                  "d_model": config.MODELS[run.model].d_model, "wv_out": config.MODELS[run.model].wv_out,
                  "trainable_params": 10000, "train_time_s": 10,
                  "peak_vram_gb": 5, "train_summary": {},
                  "flags": {"nan_loss_tasks": [], "grad_spike_steps": []}}
        utils.save_json(tmp_path / "runs" / run.run_id / "result.json", result)
    summary = run_analysis(tmp_path)
    assert summary["runs"] == len(seeds) * 6 * len(models)
    assert len(summary["figures"]) >= 10
    report = (tmp_path / "analysis" / "report.md").read_text()
    assert "Study status: incomplete" in report  # three fixed models are still missing
    if len(seeds) == 1:
        assert "SD unavailable" in report
    windows = utils.load_json(tmp_path / "aggregates" / "feasible_windows_full.json")
    assert all(w["status"] == ("complete" if len(seeds) == 5 else "incomplete") for w in windows)
    assert (tmp_path / "aggregates" / "paired_rank_comparisons.csv").exists()
