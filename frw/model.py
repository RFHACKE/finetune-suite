"""Backbone loading, LoRA (fixed alpha) or full-FT preparation, task heads."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from . import config as C


def load_tokenizer(m: C.Model):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(m.hf_id, revision=m.revision)
    tok.padding_side = "right"
    tok.truncation_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_backbone(m: C.Model):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        m.hf_id, revision=m.revision, dtype=torch.bfloat16, attn_implementation="sdpa",
        use_safetensors=True)
    model.config.use_cache = False
    for mod in model.modules():           # fp32 norm storage in both tracks (as v3)
        if isinstance(mod, nn.LayerNorm) or type(mod).__name__.endswith("RMSNorm"):
            mod.float()
        if isinstance(mod, nn.Dropout):
            mod.p = 0.0
        if hasattr(mod, "attention_dropout"):
            mod.attention_dropout = 0.0
    if (model.config.hidden_size, model.config.num_hidden_layers) != (m.d_model, m.layers):
        raise ValueError("architecture differs from registry")
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def wrap_lora(model, rank: int, targets=C.LORA_TARGETS):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=rank, lora_alpha=C.LORA_ALPHA, target_modules=list(targets),
                     lora_dropout=C.LORA_DROPOUT, bias="none", task_type="CAUSAL_LM",
                     init_lora_weights=True, use_rslora=False)
    peft_model = get_peft_model(model, cfg)
    for name, p in peft_model.named_parameters():
        if "lora_" in name:
            p.data = p.data.float()
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    n = sum(1 for _, mod in lora_modules(peft_model, targets))
    if n != len(targets) * model.config.num_hidden_layers:
        raise RuntimeError(f"LoRA matched {n} modules")
    return peft_model


def lora_modules(model, names=C.LORA_TARGETS):
    """LoRA layers whose module name is in `names` (q/v by default: the spectral reference set)."""
    for name, mod in model.named_modules():
        if name.split(".")[-1] in names and hasattr(mod, "lora_A"):
            yield name, mod


def lora_delta(mod):
    """Return (B, A, scale) for a PEFT LoRA Linear (adapter 'default')."""
    A = mod.lora_A["default"].weight     # r x d_in
    B = mod.lora_B["default"].weight     # d_out x r
    return B, A, mod.scaling["default"]


def prepare_full(model):
    emb = {id(p) for p in model.get_input_embeddings().parameters()}
    out = model.get_output_embeddings()
    if out is not None:
        emb |= {id(p) for p in out.parameters()}
    for p in model.parameters():
        if id(p) in emb:
            p.requires_grad_(False)
        else:
            p.data = p.data.float()           # fp32 master weights; bf16 autocast compute
            p.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return model


def transformer_of(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model


def make_heads(task_keys, d_model: int, seed: int) -> nn.ModuleList:
    heads = []
    for pos, key in enumerate(task_keys):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(C.head_seed(seed, pos))
            h = nn.Linear(d_model, C.TASKS[key].labels)
        heads.append(h.float())
    return nn.ModuleList(heads)


def features(transformer, ids, mask):
    out = transformer(input_ids=ids, attention_mask=mask, use_cache=False)
    hidden = out.last_hidden_state
    last = mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.size(0), device=hidden.device), last].float()


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def count(params) -> int:
    return int(sum(p.numel() for p in params))


def lora_param_count(m: C.Model, rank: int) -> int:
    # q: d->q_out ; v: d->v_out ; LoRA factors A (r x d) + B (out x r)
    return m.layers * rank * ((m.d_model + m.q_out) + (m.d_model + m.v_out))


def effective_rank_cap(m: C.Model, rank: int) -> dict:
    return {"q": min(rank, m.d_model, m.q_out), "v": min(rank, m.d_model, m.v_out)}


def sanity_scale(rank: int) -> float:
    return C.LORA_ALPHA / rank if rank else math.nan
