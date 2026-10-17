"""Model construction: backbone loading, LoRA injection, full-FT scope,
per-task classification heads (methodology Secs. 3 and 4)."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from . import config
from .utils import head_seed

logger = logging.getLogger("rank_suite.models")


# --------------------------------------------------------------------------
# Model resolution: registry keys or arbitrary Hugging Face model ids
# --------------------------------------------------------------------------
def resolve_model_spec(identifier: str):
    reverse = {m.hf_id: key for key, m in config.MODELS.items()}
    key = reverse.get(identifier, identifier)
    if key not in config.MODELS:
        raise ValueError("Model selection is fixed by the methodology: " + ", ".join(config.MODELS))
    return config.MODELS[key]


# --------------------------------------------------------------------------
# Backbone
# --------------------------------------------------------------------------
def load_tokenizer(mspec: "config.ModelSpec"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(mspec.hf_id, revision=mspec.revision,
                                       trust_remote_code=False)
    tok.truncation_side = "right"
    tok.padding_side = "right"
    if tok.pad_token is None:
        # Base decoder checkpoints may lack a pad token; use EOS (eval is
        # unaffected: the representation is taken at the last non-pad token).
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None:
        raise ValueError(f"{mspec.hf_id} has neither a padding token nor EOS token")
    return tok


def stabilize_norms(model):
    """Same normalization storage in the frozen and trainable backbones."""
    for module in model.modules():
        if isinstance(module, nn.LayerNorm) or type(module).__name__.endswith("RMSNorm"):
            module.float()
    return model


def load_backbone(mspec: "config.ModelSpec", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        mspec.hf_id,
        revision=mspec.revision,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=dtype,                # bf16 compute everywhere (Sec. 9.2)
        attn_implementation=config.ATTN_IMPL,  # torch SDPA for all models
    )
    model.config.use_cache = False
    # Keep normalization arithmetic consistent when Track B promotes its
    # trainable parameters. Frozen norms in Track A also stay in fp32.
    stabilize_norms(model)
    # Explicitly disable every dropout, including architecture-specific fields.
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        if hasattr(module, "attention_dropout"):
            module.attention_dropout = 0.0
    for key in ("attention_dropout", "hidden_dropout", "classifier_dropout"):
        if hasattr(model.config, key):
            setattr(model.config, key, 0.0)
    if mspec.key in config.MODELS:
        actual = (model.config.hidden_size, model.config.num_hidden_layers)
        if actual != (mspec.d_model, mspec.layers):
            raise ValueError(f"Architecture differs from the fixed registry: {actual}")
    return model, None


def backbone_transformer(model) -> nn.Module:
    """The base transformer stack (embeddings + decoder layers), where the
    classification representation is read. Works for plain CausalLMs and
    PEFT-wrapped ones alike."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    transformer = getattr(base, "model", None)
    if transformer is None:
        raise TypeError(f"cannot locate transformer stack on {type(base).__name__}")
    return transformer


def adapted_linear_modules(model):
    """Yield (name, module) for every q_proj/v_proj Linear of the backbone.

    After PEFT wrapping the module is a LoRA Linear (has lora_A/base_layer);
    before wrapping it is a plain nn.Linear. The module name is identical in
    both cases because PEFT replaces layers in place.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for name, module in base.named_modules():
        if name.split(".")[-1] not in config.INTRUDER_MATRICES:
            continue
        if isinstance(module, nn.Linear) or hasattr(module, "base_layer"):
            yield name, module


# --------------------------------------------------------------------------
# Classification heads (one per task, fp32, methodology Sec. 3.4)
# --------------------------------------------------------------------------
class TaskHead(nn.Module):
    def __init__(self, d_model: int, n_labels: int):
        super().__init__()
        self.linear = nn.Linear(d_model, n_labels)

    def forward(self, x):
        return self.linear(x)


def make_task_heads(task_keys, d_model: int, run_seed: int, device) -> nn.ModuleList:
    """Heads seeded from (run seed, task position) only, so they are
    identical across ranks and learning rates for a given seed."""
    heads = []
    for i, key in enumerate(task_keys):
        n_labels = config.TASK_BY_KEY[key].num_labels
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(head_seed(run_seed, i))
            head = TaskHead(d_model, n_labels)
        heads.append(head.to(device=device, dtype=torch.float32))
    return nn.ModuleList(heads)


# --------------------------------------------------------------------------
# LoRA (Track A, methodology Sec. 3.1 / 3.6)
# --------------------------------------------------------------------------
def wrap_lora(model, rank: int):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        revision=getattr(model.config, "_commit_hash", None),
        r=rank,
        lora_alpha=config.ALPHA_FACTOR * rank,   # alpha = 2r
        target_modules=list(config.INTRUDER_MATRICES),
        lora_dropout=config.DROPOUT,             # 0.0
        bias="none",                             # lora_bias = False
        task_type="CAUSAL_LM",
        init_lora_weights=True,                  # PEFT default: A Kaiming-uniform fan-in, B zeros
    )
    peft_model = get_peft_model(model, cfg)

    # Adapter parameters stored/optimized in fp32, compute in bf16 (Sec. 3.1)
    for name, p in peft_model.named_parameters():
        if "lora_" in name and p.dtype != torch.float32:
            p.data = p.data.float()

    n_lora = sum(1 for _, m in adapted_linear_modules(peft_model) if hasattr(m, "lora_A"))
    transformer = backbone_transformer(peft_model)
    expected = 2 * len(getattr(transformer, "layers", []))
    if n_lora == 0 or n_lora != expected:
        raise RuntimeError(
            f"LoRA target modules {config.INTRUDER_MATRICES} matched nothing "
            f"for {type(model).__name__}"
        )
    logger.info("LoRA injected into %d projection matrices (expected %d)", n_lora, expected)

    # The backbone must be fully frozen.
    frozen_ok = all(
        not p.requires_grad for n, p in peft_model.named_parameters() if "lora_" not in n
    )
    if not frozen_ok:
        raise RuntimeError("backbone parameters unexpectedly trainable after LoRA wrap")
    return peft_model


# --------------------------------------------------------------------------
# Full fine-tuning (Track B, methodology Sec. 4.2)
# --------------------------------------------------------------------------
def prepare_full_ft(model):
    """All non-embedding backbone weights trainable; embeddings and the tied
    lm_head stay frozen; gradient checkpointing on."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model

    frozen_ids = set()
    for getter in (base.get_input_embeddings, base.get_output_embeddings):
        module = getter() if callable(getter) else getter
        if module is not None:
            for p in module.parameters():
                frozen_ids.add(id(p))

    n_trainable = n_frozen = 0
    for p in base.parameters():
        if id(p) in frozen_ids:
            p.requires_grad = False
            n_frozen += 1
        else:
            p.requires_grad = True
            # Low-rate updates must accumulate in fp32; directly updating stored
            # bf16 weights can round small Adam steps away. Autocast is still bf16.
            p.data = p.data.float()
            n_trainable += 1

    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    base.enable_input_require_grads()
    base.config.use_cache = False
    logger.info("Full FT scope: %d trainable / %d frozen tensors (embeddings frozen)",
                n_trainable, n_frozen)
    return base


# --------------------------------------------------------------------------
# Forward pass: last non-pad token representation -> task head
# --------------------------------------------------------------------------
def forward_logits(transformer, head, batch) -> torch.Tensor:
    out = transformer(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    hidden = out.last_hidden_state                       # bf16 under autocast
    last_idx = batch["attention_mask"].long().sum(dim=1) - 1  # right padding
    reps = hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]
    # An fp32 tensor alone does not disable autocast for the head's linear op.
    with torch.autocast(device_type=reps.device.type, enabled=False):
        return head(reps.float())


def trainable_param_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())
