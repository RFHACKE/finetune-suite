from dataclasses import replace
import math

import pytest

from rank_suite import config
from rank_suite.protocol import experiment_controls, immutable_json
from rank_suite.runner import build_grid


def test_complete_design_and_pilot_reuse():
    main = build_grid(config.ExperimentSpec())
    assert len(main) == 120
    assert sum(r.track == "lora" for r in main) == 100
    assert len({r.run_id for r in main}) == 120
    pilot = build_grid(config.ExperimentSpec(mode="pilot"))
    assert len(pilot) == 8
    assert {r.run_id for r in pilot} <= {r.run_id for r in main}
    assert len(build_grid(config.ExperimentSpec(mode="sanity"))) == 15
    assert all(r.mode == "smoke" for r in build_grid(config.ExperimentSpec(mode="smoke")))
    assert experiment_controls(config.ExperimentSpec()) == experiment_controls(config.ExperimentSpec(mode="pilot"))
    smoke = experiment_controls(config.ExperimentSpec(mode="smoke"))
    assert (smoke["train_samples"], smoke["epochs"], smoke["evaluation_limit"]) == (64, 1, 32)
    sanity = experiment_controls(config.ExperimentSpec(mode="sanity"))
    assert sanity["task_order"] == ["rte"] and sanity["train_samples_used"] == 1992


def test_shards_cover_design_without_splitting_paired_seed_blocks():
    for n in (2, 3, 4):
        shards = [build_grid(config.ExperimentSpec(num_shards=n, shard_index=i)) for i in range(n)]
        sets = [{r.run_id for r in shard} for shard in shards]
        assert len(set.union(*sets)) == 120
        assert sum(map(len, sets)) == 120
        for i, shard in enumerate(shards):
            blocks = {(r.model, r.seed) for r in shard}
            assert len(shard) == len(blocks) * 6
            pilot = build_grid(config.ExperimentSpec(mode="pilot", num_shards=n, shard_index=i))
            assert {r.run_id for r in pilot} <= sets[i]


def test_resume_ids_include_all_scientific_inputs():
    run = build_grid(config.ExperimentSpec())[0]
    for changed in (replace(run, lr=run.lr * 1.00001), replace(run, epochs=2),
                    replace(run, train_samples=100), replace(run, optimizer="adamw8bit")):
        assert run.run_id != changed.run_id


@pytest.mark.parametrize("kw", [{"micro_batch": 3}, {"micro_batch": 4, "accum": 2},
                                  {"lora_lrs": {"llama3.2-1b": [math.nan]}},
                                  {"models": ["other/model"]}, {"budget": 25},
                                  {"num_shards": 2, "shard_index": 2}, {"seeds": [-1]}])
def test_invalid_controls_fail_before_gpu_work(kw):
    with pytest.raises(ValueError):
        config.ExperimentSpec(**kw)


def test_immutable_records_reject_changes(tmp_path):
    path = tmp_path / "protocol.json"
    immutable_json(path, {"x": 1})
    immutable_json(path, {"x": 1})
    with pytest.raises(ValueError, match="controls changed"):
        immutable_json(path, {"x": 2})


def test_immutable_data_counts_survive_json_round_trip(tmp_path):
    payload = {"label_counts": {0: 1249, 1: 1241}, "shape": (2, 3)}
    immutable_json(tmp_path / "selection.json", payload)
    immutable_json(tmp_path / "selection.json", payload)
