from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset
import numpy as np
import pytest
import torch
from torch import nn

from rank_suite import config, metrics, train
from rank_suite.data import TaskData, make_loader


class Tokenizer:
    def pad(self, features, padding, return_tensors, pad_to_multiple_of=8):
        width = max(map(len, features["input_ids"]))
        width = ((width + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        return {k: torch.tensor([v + [0] * (width - len(v)) for v in values])
                for k, values in features.items()}


class ToyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)

    def forward(self, input_ids, attention_mask, use_cache=False):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_optimizer_moment_precision_does_not_switch_at_small_ranks():
    # Inspect the actual pinned library configuration without needing CUDA.
    for rank in (4, 8, 16, 32, 64):
        matrix = nn.Parameter(torch.ones(256, rank))
        vector = nn.Parameter(torch.ones(256))
        head = nn.Linear(256, 2)
        params = [matrix, vector] + list(head.parameters())
        opt = train.make_optimizer(params, .0002, "adamw8bit", head.parameters())
        configs = [opt.get_config(0, i, opt.param_groups[0]) for i in range(len(params))]
        assert configs[0]["optim_bits"] == 8
        assert configs[0]["min_8bit_size"] == 0
        assert all(c["optim_bits"] == 32 for c in configs[1:])


def test_exported_adapter_keeps_the_pinned_base_revision():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from rank_suite.model_utils import wrap_lora
    cfg = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1)
    cfg._commit_hash = config.MODEL_REVISIONS["qwen2.5-1.5b"]
    model = wrap_lora(Qwen2ForCausalLM(cfg), 4)
    assert model.peft_config["default"].revision == cfg._commit_hash


@pytest.mark.parametrize("head_multiplier", [1, 5])
def test_accumulation_matches_full_batches_including_unequal_final_microbatch(tmp_path, head_multiplier):
    torch.manual_seed(4)
    initial = ToyTransformer()
    initial_head = nn.Linear(4, 2)
    ds = Dataset.from_list([{"input_ids": [1, i % 15 + 1], "attention_mask": [1, 1], "labels": i % 2}
                            for i in range(26)])
    states = []
    import copy
    for micro, accum in [(16, 1), (8, 2), (4, 4), (2, 8)]:
        model, head = copy.deepcopy(initial), copy.deepcopy(initial_head)
        loader = make_loader(ds, micro, False, 1, Tokenizer())
        out = train._train_task(model, head, list(model.parameters()) + list(head.parameters()),
                                loader, 2, .01, "adamw", "cpu", tmp_path / f"{micro}.jsonl", 0, "rte", accum,
                                head_lr_multiplier=head_multiplier)
        assert out["steps"] == 4
        states.append([p.detach().clone() for p in list(model.parameters()) + list(head.parameters())])
    for state in states[1:]:
        for expected, actual in zip(states[0], state):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("family", ["llama", "qwen2", "granite"])
@pytest.mark.parametrize("track", ["lora", "full"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA integration requires a local NVIDIA GPU"))])
def test_real_tiny_architectures_and_deferred_spectra(family, track, device, monkeypatch, tmp_path):
    from transformers import (LlamaConfig, LlamaForCausalLM, Qwen2Config,
                              Qwen2ForCausalLM, GraniteConfig, GraniteForCausalLM)
    classes = {"llama": (LlamaConfig, LlamaForCausalLM), "qwen2": (Qwen2Config, Qwen2ForCausalLM),
               "granite": (GraniteConfig, GraniteForCausalLM)}
    cfg_cls, model_cls = classes[family]
    cfg = cfg_cls(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                  num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                  max_position_embeddings=64, pad_token_id=0, tie_word_embeddings=True)
    cfg._attn_implementation = "sdpa"
    def load(_):
        from rank_suite.model_utils import stabilize_norms
        model = model_cls(cfg)
        if device == "cuda":
            model = stabilize_norms(model.bfloat16())
        return model, None
    monkeypatch.setattr(train, "load_backbone", load)
    mspec = replace(config.MODELS["qwen2.5-1.5b"], d_model=16, layers=1, n_kv=1, head_dim=8, vocab=32)
    ds = Dataset.from_list([{"input_ids": [1, 2, i % 28 + 3], "attention_mask": [1, 1, 1], "labels": i % 2}
                            for i in range(18)])
    data = {k: TaskData(config.TASK_BY_KEY[k], ds, ds, list(range(18)), {0: 9, 1: 9})
            for k in config.TASK_KEYS}
    optimizer = "adamw8bit" if device == "cuda" and track == "full" else "adamw"
    run = config.RunSpec(mspec.key, track, 4 if track == "lora" else None, .001, 42,
                         mode="smoke", train_samples=18, epochs=1, optimizer=optimizer)
    spec = config.ExperimentSpec(models=[mspec.key], optimizer="standard" if device == "cuda" else "adamw")
    directory = tmp_path / "runs" / run.run_id
    old_overwrite = torch.__future__.get_overwrite_module_params_on_conversion()
    if device == "cpu" and family == "qwen2" and track == "lora":
        # Reproduce device-conversion implementations that replace Parameters.
        torch.__future__.set_overwrite_module_params_on_conversion(True)
    try:
        result = train.run_single(run, directory, data, Tokenizer(), mspec,
                                  experiment=spec, protocol_sha="test", device=device)
    finally:
        torch.__future__.set_overwrite_module_params_on_conversion(old_overwrite)
    assert result["status"] == "completed"
    assert result["train_summary"]["sst2"]["global_step"] == 6
    assert len(result["boundaries"]) == 3
    assert not (directory / "resume.pt").exists()
    assert result["intruder_status"] == "pending_cpu"
    if track == "lora":
        checkpoint = directory / result["boundaries"][0]["path"] / "projections.pt"
        factors = torch.load(checkpoint, map_location="cpu", weights_only=True)["matrices"]
        assert any(torch.count_nonzero(value["B"]) > 0 for value in factors.values())
    from rank_suite.intruder import analyze_intruders
    assert analyze_intruders(tmp_path, threads=1)["analyzed_runs"] == 1
    assert analyze_intruders(tmp_path, threads=1)["analyzed_runs"] == 1
    import json
    saved = json.loads((directory / "result.json").read_text())
    assert saved["intruder_status"] == "completed"
    if family == "qwen2" and track == "lora" and device == "cpu":
        from rank_suite.analysis import _write_long_tables
        from rank_suite.measurement_plots import measurement_figures
        loaded = metrics.load_results(tmp_path, modes=("smoke",))
        aggregate = tmp_path / "aggregates"
        aggregate.mkdir()
        _write_long_tables(loaded, aggregate)
        import pandas as pd
        diagnostic = pd.read_csv(aggregate / "spectral_diagnostics_long.csv")
        assert len(diagnostic) == 6  # both q/v projections at all three boundaries
        assert (diagnostic["effective_rank"] > 0).all()
        assert set(diagnostic["subspace_k_actual_20"]) == {8, 16}
        plots = measurement_figures(metrics.results_dataframe(loaded), loaded, tmp_path)
        assert len(plots) == 3
        assert all((tmp_path / name).stat().st_size > 1000 for name in plots)


def test_interrupted_run_resumes_identical_predictions(monkeypatch, tmp_path):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, pad_token_id=0,
                      max_position_embeddings=64, tie_word_embeddings=True)
    cfg._attn_implementation = "sdpa"
    monkeypatch.setattr(train, "load_backbone", lambda _: (Qwen2ForCausalLM(cfg), None))
    m = replace(config.MODELS["qwen2.5-1.5b"], d_model=16, layers=1, n_kv=1, head_dim=8, vocab=32)
    ds = Dataset.from_list([{"input_ids": [1, i % 28 + 3], "attention_mask": [1, 1], "labels": i % 2}
                            for i in range(18)])
    data = {k: TaskData(config.TASK_BY_KEY[k], ds, ds, list(range(18)), {0: 9, 1: 9}) for k in config.TASK_KEYS}
    run = config.RunSpec(m.key, "lora", 4, .01, 42, mode="smoke", epochs=1, train_samples=18, optimizer="adamw")
    spec = config.ExperimentSpec(models=[m.key], optimizer="adamw")
    def execute(root):
        return train.run_single(run, root / "runs" / run.run_id, data, Tokenizer(), m,
                                experiment=spec, protocol_sha="test", device="cpu")
    expected = execute(tmp_path / "reference")
    original = train._train_task
    def interrupt(*args, **kwargs):
        if args[10] == "mnli":
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)
    monkeypatch.setattr(train, "_train_task", interrupt)
    with pytest.raises(RuntimeError, match="simulated"):
        execute(tmp_path / "resumed")
    monkeypatch.setattr(train, "_train_task", original)
    actual = execute(tmp_path / "resumed")
    assert actual["A"] == expected["A"]
    for boundary in actual["boundaries"]:
        for key in config.TASK_KEYS[:actual["task_order"].index(boundary["task"]) + 1]:
            rel = Path(boundary["path"]) / f"predictions_{key}.npz"
            ref = np.load(tmp_path / "reference" / "runs" / run.run_id / rel)
            got = np.load(tmp_path / "resumed" / "runs" / run.run_id / rel)
            np.testing.assert_array_equal(got["predictions"], ref["predictions"])
        rel = Path(boundary["path"]) / "projections.pt"
        ref = torch.load(tmp_path / "reference" / "runs" / run.run_id / rel,
                         map_location="cpu", weights_only=True)["matrices"]
        got = torch.load(tmp_path / "resumed" / "runs" / run.run_id / rel,
                         map_location="cpu", weights_only=True)["matrices"]
        for name in ref:
            for factor in ("A", "B"):
                torch.testing.assert_close(got[name][factor], ref[name][factor], atol=0, rtol=0)


def test_head_lr_groups_preserve_full_ft_moment_precision():
    matrix = nn.Parameter(torch.ones(256, 16))
    vector = nn.Parameter(torch.ones(256))
    head = nn.Linear(256, 2)
    opt = train.make_optimizer([matrix, vector, *head.parameters()], .0001, 'adamw8bit',
                                head.parameters(), head_lr_multiplier=5)
    assert [group['lr'] for group in opt.param_groups] == [.0001, .0005]
    assert opt.get_config(0, 0, opt.param_groups[0])['optim_bits'] == 8
    assert opt.get_config(0, 1, opt.param_groups[0])['optim_bits'] == 32
    assert all(opt.get_config(1, i, opt.param_groups[1])['optim_bits'] == 32 for i in range(2))


def test_real_calibration_training_records_nll_and_head_lr_without_large_exports(tmp_path, monkeypatch):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, pad_token_id=0,
                      max_position_embeddings=64, tie_word_embeddings=True)
    cfg._attn_implementation = 'sdpa'
    monkeypatch.setattr(train, 'load_backbone', lambda _: (Qwen2ForCausalLM(cfg), None))
    mspec = replace(config.MODELS['qwen2.5-1.5b'], d_model=16, layers=1, n_kv=1, head_dim=8, vocab=32)
    ds = Dataset.from_list([{'input_ids': [1, i % 28 + 3], 'attention_mask': [1, 1], 'labels': i % 2}
                            for i in range(18)])
    data = {k: TaskData(config.TASK_BY_KEY[k], ds, ds, list(range(18)), {0: 9, 1: 9})
            for k in config.TASK_KEYS}
    run = config.RunSpec(mspec.key, 'lora', 4, .001, 42, mode='lr_test', epochs=1,
                         train_samples=18, optimizer='adamw', head_lr_multiplier=5)
    spec = config.ExperimentSpec(models=[mspec.key], optimizer='adamw', mode='lr_test')
    directory = tmp_path / 'runs' / run.run_id
    result = train.run_single(run, directory, data, Tokenizer(), mspec,
                              experiment=spec, protocol_sha='test', device='cpu')
    assert result['intruder_status'] == 'not_requested'
    assert np.isfinite(result['NLL'][-1]).all()
    assert not list(tmp_path.rglob('projections.pt'))
    assert not list(tmp_path.rglob('adapter_model.safetensors'))
    assert not (directory / 'resume.pt').exists()
    import json
    steps = [json.loads(line) for line in (directory / 'train_log.jsonl').read_text().splitlines()]
    assert any(step['lr'] > 0 for step in steps)
    assert all(step['head_lr'] == pytest.approx(5 * step['lr']) for step in steps)
