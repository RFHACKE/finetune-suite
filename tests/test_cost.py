from rank_suite import config, utils
from rank_suite.cost import cost_estimate, scheduling_limit_hours
from rank_suite.runner import build_grid


def test_no_cost_promise_without_measured_pilots(tmp_path):
    grid = build_grid(config.ExperimentSpec())
    estimate = cost_estimate(grid, tmp_path, hourly_rate=.5)
    assert estimate["unmeasured_runs"] == 120
    assert estimate["planning_range_usd"] is None
    for run in build_grid(config.ExperimentSpec(mode="pilot")):
        utils.save_json(tmp_path / "runs" / run.run_id / "result.json", {
            "status": "completed", "mode": "main", "model": run.model, "track": run.track,
            "optimizer": run.optimizer, "train_time_s": 3600,
        })
    estimate = cost_estimate(grid, tmp_path, hourly_rate=.5)
    assert estimate["unmeasured_runs"] == 0
    assert estimate["remaining_runs"] == 112
    assert estimate["planning_range_usd"] == [70, 112]


def test_scheduling_uses_most_restrictive_allowance():
    spec = config.ExperimentSpec(budget=25, reserve=5, hourly_rate=.5, max_hours=10)
    assert scheduling_limit_hours(spec) == 10
