"""Promoted defaults must reproduce the saved empirical LR/head policy."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from rank_suite import calibration, config
from rank_suite.runner import build_grid


def evidence():
    return json.loads((Path(__file__).parents[1] / "docs/calibrated-lora-defaults.json").read_text())


def test_fresh_main_grid_uses_empirical_lora_policies_and_keeps_full_priors(tmp_path):
    source = evidence()
    assert source["completed_calibration_cases"] == 48
    spec = config.ExperimentSpec(output_dir=str(tmp_path))
    grid = build_grid(spec)
    assert len(grid) == 120
    for run in grid:
        if run.track == "lora":
            policy = source["models"][run.model]
            assert run.lr == policy["lr"]
            assert run.head_lr_multiplier == policy["head_lr_multiplier"]
            assert config.MODELS[run.model].revision == policy["model_revision"]
            assert policy["completed_cases"] == 12
        else:
            assert run.lr == pytest.approx(config.METHODOLOGY_LORA_LR[run.model] / 10)
            assert run.head_lr_multiplier == 1
    # Promoting main defaults must not silently change registered sensitivity grids.
    for group in calibration.make_plan(spec)["groups"]:
        if group["track"] == "lora":
            assert group["rates"] == source["models"][group["model"]]["tested_rates"]
    assert len(build_grid(replace(spec, mode="sanity"))) == 15


def test_cohort_and_explicit_overrides_keep_priority_over_project_defaults(tmp_path):
    selected = {"llama3.2-1b": {"lora": {"lr": .0001, "head_lr_multiplier": 3},
                               "full": {"lr": .00001, "head_lr_multiplier": 2}}}
    spec = config.ExperimentSpec(output_dir=str(tmp_path), calibrated_defaults=selected)
    llama = [run for run in build_grid(spec) if run.model == "llama3.2-1b"]
    assert {(run.track, run.lr, run.head_lr_multiplier) for run in llama} == {
        ("lora", .0001, 3), ("full", .00001, 2)}
    overridden = replace(spec, lora_lrs={"llama3.2-1b": [.00015]}, head_lr_multiplier=1)
    llama = [run for run in build_grid(overridden) if run.model == "llama3.2-1b"]
    assert {(run.track, run.lr, run.head_lr_multiplier) for run in llama} == {
        ("lora", .00015, 1), ("full", .00001, 1)}
