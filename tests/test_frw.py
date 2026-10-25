"""End-to-end CPU checks for the frw-v4 pipeline using a tiny random Qwen2 backbone."""
import json
import math

import pytest
import torch

from frw import config as C
from frw import data as D
from frw import jobs as J
from frw import model as M
from frw import train as T

TINY = C.Model("tiny-qwen", "Qwen/Qwen2.5-1.5B", C.MODELS["qwen2.5-1.5b"].revision, "tiny",
               0.001, 2, 64, 64, 32, "qwen")


def _tiny_backbone(m):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      tie_word_embeddings=True)
    model = Qwen2ForCausalLM(cfg).to(torch.bfloat16)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    for mod in model.modules():
        if type(mod).__name__.endswith("RMSNorm"):
            mod.float()
    return model


@pytest.fixture()
def tiny(monkeypatch):
    pytest.importorskip("peft")
    monkeypatch.setitem(C.MODELS, "tiny-qwen", TINY)
    monkeypatch.setattr(T, "load_backbone", _tiny_backbone)
    original = D.load_task

    def small(key, tok, split):
        td = original(key, tok, split)
        cut = lambda s, n: D.Split(s.input_ids[:n], s.labels[:n], s.source_index[:n])  # noqa: E731
        td.train, td.eval = cut(td.train, 48), cut(td.eval, 40)
        return td
    monkeypatch.setattr(T, "load_task", small)
    try:
        ctx = T.Context("tiny-qwen", torch.device("cpu"))
    except OSError:
        pytest.skip("Qwen tokenizer not cached")
    return ctx


def test_sequence_runs_and_base_is_restored(tiny, tmp_path):
    ctx = tiny
    base = ctx.base_model()
    before = {k: v.clone() for k, v in base.state_dict().items()}
    job = J.Job("main", "tiny-qwen", "lora", 4, 4e-3, 42, "O1", "official")
    res = T.run(job, ctx, tmp_path / "a", probes=True, save_adapter=True, log=lambda *a: None)
    assert res["status"] == "completed"
    A = res["A"]
    assert all(A[i][j] is not None for i in range(3) for j in range(i + 1))
    assert res["probe"][2][0] is not None
    assert len(res["spectral_summary"]) == 3
    assert (tmp_path / "a" / "adapter_final_bf16.safetensors").exists()
    # The pristine base must be returned unchanged and without LoRA layers.
    after = ctx.base.state_dict()
    assert set(after) == set(before)
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert not any("lora" in n for n, _ in ctx.base.named_modules())
    # A second job on the same cached base works and is reproducible for the same seed.
    res2 = T.run(job, ctx, tmp_path / "b", probes=False, log=lambda *a: None)
    assert res2["A"] == res["A"]


def test_alpha_fixed_scale(tiny):
    base = tiny.base_model()
    for r in (1, 8, 64):
        pm = M.wrap_lora(base, r)
        scales = {mod.scaling["default"] for _, mod in M.lora_modules(pm)}
        assert scales == {C.LORA_ALPHA / r}
        tiny.base = pm.unload()
        base = tiny.base


def test_spectral_matches_dense(tiny):
    from frw.spectral import _sv_from_factors
    torch.manual_seed(1)
    B, A = torch.randn(32, 6), torch.randn(6, 64)
    s1 = _sv_from_factors(B, A, 0.5)
    s2 = torch.linalg.svdvals((0.5 * B @ A).double())[:6]
    assert torch.allclose(s1, s2, rtol=1e-8, atol=1e-8)


def test_metrics_definitions():
    A = [[0.8, None, None], [0.7, 0.9, None], [0.6, 0.85, 0.95]]
    m = T.metrics(A)
    assert math.isclose(m["bwt"], ((0.6 - 0.8) + (0.85 - 0.9)) / 2)
    assert math.isclose(m["acc_final"], (0.6 + 0.85 + 0.95) / 3)
    assert math.isclose(m["forgetting_max"], ((0.8 - 0.6) + (0.9 - 0.85)) / 2)


def test_selection_rule_extends_edges():
    ok = {"status": "completed"}
    mk = lambda acc: {**ok, "A": [[acc]]}  # noqa: E731
    res = {1e-4: [mk(.5), mk(.5)], 2e-4: [mk(.6), mk(.6)], 4e-4: [mk(.7), mk(.7)], 8e-4: [mk(.8), mk(.8)]}
    st, lr, extra, _ = J.select_lr(res, 2, C.LORA_LR_MIN, C.LORA_LR_MAX)
    assert st == "extend" and math.isclose(extra, 1.6e-3)
    res[1.6e-3] = [mk(.75), mk(.75)]
    st, lr, extra, _ = J.select_lr(res, 2, C.LORA_LR_MIN, C.LORA_LR_MAX)
    assert st == "selected" and lr == 8e-4
    res[2e-4] = [mk(.8), mk(.8)]          # tie -> lower LR
    assert J.select_lr(res, 2, C.LORA_LR_MIN, C.LORA_LR_MAX)[1] == 2e-4
    assert J.select_lr({1e-4: [mk(.5)]}, 2, C.LORA_LR_MIN, C.LORA_LR_MAX)[0] == "pending"


def test_job_ids_are_stable_and_unique():
    jobs = J.tune_jobs("qwen2.5-1.5b")
    ids = [j.job_id for j in jobs]
    assert len(ids) == len(set(ids)) == len(C.RANKS) * len(C.LORA_LR_GRID) * len(C.TUNE_SEEDS)
    again = J.Job.from_spec(json.loads(json.dumps(jobs[0].spec)))
    assert again.job_id == jobs[0].job_id


def test_all_linear_wrap_and_restore(tiny, tmp_path):
    base = tiny.base_model()
    before = {k: v.clone() for k, v in base.state_dict().items()}
    job = J.Job("main_al", "tiny-qwen", "lora", 4, 4e-3, 42, "O1", "official")
    res = T.run(job, tiny, tmp_path / "al", probes=True, save_adapter=True, log=lambda *a: None)
    assert res["status"] == "completed" and res["lora_targets"] == list(C.ALL_LINEAR)
    assert len(res["spectral_summary"][-1]["summary"]["all"]) and \
        res["spectral_summary"][-1]["summary"]["all"]["n_matrices"] == 2 * 2     # q/v only
    from safetensors.torch import load_file
    keys = load_file(str(tmp_path / "al" / "adapter_final_bf16.safetensors"))
    assert len(keys) == 2 * len(C.ALL_LINEAR) * 2                                  # A and B, 7 targets, 2 layers
    after = tiny.base.state_dict()
    assert set(after) == set(before) and all(torch.equal(before[k], after[k]) for k in before)
    # q/v job after an all-linear job is unaffected
    res2 = T.run(J.Job("stab", "tiny-qwen", "lora", 4, 4e-3, 42, "O1", "official"), tiny, tmp_path / "s",
                 log=lambda *a: None)
    assert res2["lora_targets"] == list(C.LORA_TARGETS)


def test_extension_jobs_have_distinct_ids():
    old = J.main_jobs("qwen2.5-1.5b", 16, 2e-4)
    al = J.main_jobs("qwen2.5-1.5b", 16, 2e-4, suffix="_al")
    st = J.stab_jobs("qwen2.5-3b", 256, 4e-4)
    assert not {j.job_id for j in old} & {j.job_id for j in al}
    assert len(st) == 10 and {j.lr for j in st} == {4e-4, 2e-4}
    assert old[0].job_id == J.Job("main", "qwen2.5-1.5b", "lora", 16, 2e-4, 42, "O1", "official").job_id


def test_epoch_cohort_trains_requested_epochs(tiny, tmp_path):
    import gzip
    job = J.Job("ep2_al", "tiny-qwen", "lora", 4, 4e-3, 42, "O1", "official")
    res = T.run(job, tiny, tmp_path / "e", log=lambda *a: None)
    assert res["epochs"] == 2 and res["lora_targets"] == list(C.ALL_LINEAR)
    with gzip.open(tmp_path / "e" / "train_log.jsonl.gz", "rt") as fh:
        epochs = {json.loads(x)["epoch"] for x in fh}
    assert epochs == {0, 1}
    assert T.epochs_for("main_al") == C.EPOCHS == 3 and T.epochs_for("ep5_al") == 5
