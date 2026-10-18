import copy
from dataclasses import replace
import json
from types import SimpleNamespace

from datasets import Dataset
import numpy as np
import pytest
import torch
from torch import nn

from rank_suite import config, train
from rank_suite.data import TaskData, make_loader
from rank_suite.overfitting import EpochDiagnostics, stratified_indices, summarize_run, write_reports
from rank_suite.protocol import experiment_controls
from rank_suite.runner import build_grid


class Tokenizer:
    def pad(self, features, padding, return_tensors, pad_to_multiple_of=8):
        width = ((max(map(len, features["input_ids"])) + 7) // 8) * 8
        return {key: torch.tensor([row + [0] * (width - len(row)) for row in rows])
                for key, rows in features.items()}


class StochasticTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.dropout = nn.Dropout(.3)

    def forward(self, input_ids, attention_mask, use_cache=False):
        return SimpleNamespace(last_hidden_state=self.dropout(self.embedding(input_ids)))


def dataset():
    return Dataset.from_list([{"input_ids": [1, i % 28 + 3], "attention_mask": [1, 1],
                               "labels": i % 2} for i in range(26)])


def test_high_rank_grid_requires_explicit_separate_protocol():
    with pytest.raises(ValueError, match="allow-overcomplete"):
        config.ExperimentSpec(track="lora", ranks=[512, 1024])
    spec = config.ExperimentSpec(track="lora", ranks=[512, 1024],
                                 allow_overcomplete_ranks=True, epoch_diagnostics=True,
                                 micro_batch=16, accum=1)
    grid = build_grid(spec)
    assert len(grid) == 40
    assert {r.rank for r in grid} == {512, 1024}
    assert {r.model for r in grid} == set(config.MODELS)
    assert all(r.epochs == 3 and r.train_samples == 2490 for r in grid)
    assert all(r.lr == config.DEFAULT_LORA_LR[r.model] for r in grid)
    assert experiment_controls(spec)["epoch_diagnostics"]["model_selection"] is False
    assert experiment_controls(spec)["rank_interpretation"] == "overcomplete_factor_width_allowed"
    assert experiment_controls(spec) != experiment_controls(replace(spec, epoch_diagnostics=False))


def test_probes_preserve_rng_training_weights_and_shuffling(tmp_path):
    torch.manual_seed(7)
    initial, initial_head = StochasticTransformer(), nn.Linear(8, 2)
    ds = dataset()
    task = TaskData(config.TASK_BY_KEY["rte"], ds, ds, list(range(26)), {0: 13, 1: 13})
    states, rng_states = [], []
    for enabled in (False, True):
        model, head = copy.deepcopy(initial), copy.deepcopy(initial_head)
        torch.manual_seed(456)
        path = tmp_path / str(enabled)
        path.mkdir()
        callback = EpochDiagnostics(model, head, task, Tokenizer(), path, "cpu", 12, 8) if enabled else None
        loader = make_loader(ds, 16, True, 123, Tokenizer())
        train._train_task(model, head, [*model.parameters(), *head.parameters()], loader,
                          3, .01, "adamw", "cpu", path / "train.jsonl", 0, "rte",
                          epoch_callback=callback)
        states.append([p.detach().clone() for p in [*model.parameters(), *head.parameters()]])
        rng_states.append(torch.random.get_rng_state())
    for original, observed in zip(*states):
        torch.testing.assert_close(original, observed, atol=0, rtol=0)
    assert torch.equal(*rng_states)
    rows = [json.loads(line) for line in (tmp_path / "True/epoch_diagnostics.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [1, 2, 3]
    assert all(row["train"]["count"] == 12 and row["validation"]["count"] == 12 for row in rows)
    prediction = np.load(tmp_path / "True" / rows[-1]["validation"]["predictions"])
    np.testing.assert_allclose(prediction["probabilities"].sum(axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(prediction["example_nll"].mean(), rows[-1]["validation"]["nll"], atol=1e-6)


def test_probe_sampling_is_stratified_fixed_and_does_not_change_numpy_rng():
    labels = [0] * 60 + [1] * 30 + [2] * 10
    np.random.seed(72)
    before = np.random.get_state()
    selected = stratified_indices(labels, 20)
    after = np.random.get_state()
    assert selected == stratified_indices(labels, 20)
    assert len(set(selected)) == 20
    assert [sum(labels[i] == c for i in selected) for c in (0, 1, 2)] == [12, 6, 2]
    assert np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert stratified_indices(labels, 1000) == list(range(100))


def test_overfitting_warning_requires_opposing_loss_trends():
    rows = [{"task": "rte", "epoch": epoch, "train": {"nll": train_nll},
             "validation": {"nll": val_nll, "accuracy": .7}, "nll_gap": val_nll - train_nll}
            for epoch, train_nll, val_nll in ((1, .6, .7), (2, .4, .8), (3, .3, .6))]
    summary = summarize_run(rows)
    assert summary["possible_overfitting"]
    assert summary["tasks"]["rte"]["warnings"][0]["epoch"] == 2
    assert summary["tasks"]["rte"]["lowest_probe_nll_epoch"] == 3
    assert not summarize_run([rows[0], rows[2]])["possible_overfitting"]


@pytest.mark.parametrize("family", ["llama", "qwen", "granite"])
def test_real_overcomplete_adapters_train_export_and_analyze(family, tmp_path, monkeypatch):
    from transformers import (LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM,
                              GraniteConfig, GraniteForCausalLM)
    classes = {"llama": (LlamaConfig, LlamaForCausalLM), "qwen": (Qwen2Config, Qwen2ForCausalLM),
               "granite": (GraniteConfig, GraniteForCausalLM)}
    cfg_type, model_type = classes[family]
    cfg = cfg_type(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                   num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                   max_position_embeddings=64, pad_token_id=0, tie_word_embeddings=True)
    cfg._attn_implementation = "sdpa"
    monkeypatch.setattr(train, "load_backbone", lambda _: (model_type(cfg), None))
    model = replace(config.MODELS["qwen2.5-1.5b"], d_model=16, layers=1, n_kv=1, head_dim=8, vocab=32)
    ds = dataset()
    data = {key: TaskData(config.TASK_BY_KEY[key], ds, ds, list(range(26)), {0: 13, 1: 13})
            for key in config.TASK_KEYS}
    spec = config.ExperimentSpec(models=[model.key], track="lora", ranks=[32], optimizer="adamw",
                                 allow_overcomplete_ranks=True, epoch_diagnostics=True,
                                 diagnostic_samples=12)
    run = config.RunSpec(model.key, "lora", 32, .001, 42, epochs=3, train_samples=26)
    directory = tmp_path / "runs" / run.run_id
    result = train.run_single(run, directory, data, Tokenizer(), model,
                              experiment=spec, protocol_sha="test", device="cpu")
    assert result["status"] == "completed"
    assert result["rank_semantics"]["maximum_q_update_rank"] == 16
    assert result["rank_semantics"]["maximum_v_update_rank"] == 8
    rows = (directory / "epoch_diagnostics.jsonl").read_text().splitlines()
    assert len(rows) == 9
    assert result["overfitting"]["tasks"]["mnli"]["observed_epochs"] == [1, 2, 3]
    from rank_suite.intruder import analyze_intruders
    assert analyze_intruders(tmp_path, threads=1)["analyzed_runs"] == 1
    if family == "qwen":
        charts = write_reports(tmp_path)
        assert charts["completed_sequences"] == 1
        assert len(charts["charts"]) == 2
        assert all((tmp_path / path).stat().st_size > 1000 for path in charts["charts"])
