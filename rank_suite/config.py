"""Fixed four-model registry and protocol-v2 controls.

The nominal model-specific LRs retain methodology v1.1's printed values as
heuristic priors, not proven size-transfer laws. See docs/LITERATURE_REVIEW.md
and docs/PROTOCOL_V2.md for source checks and explicit numerical amendments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Optional, Sequence

SUITE_NAME = "feasible-rank-window"
SUITE_VERSION = "2.0.0"

# --------------------------------------------------------------------------
# Controlled constants (methodology Sec. 2.4 / Appendix A)
# --------------------------------------------------------------------------
TRAIN_SAMPLES = 2490          # per task, drawn once with DATA_SEED
DATA_SEED = 42                # data subset selection seed (shared by all runs)
EPOCHS = 3                    # epochs per task
TRAIN_BATCH = 16              # effective training batch size
EVAL_BATCH = 32
MAX_LEN = 256                 # right truncation, dynamic right padding
WARMUP_FRAC = 0.03            # 3% linear warmup then cosine to 0, per task
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.0
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
DROPOUT = 0.0                 # everywhere
ATTN_IMPL = "sdpa"            # torch SDPA for all models (no FlashAttention-2)
COMPUTE_DTYPE = "bfloat16"

DEFAULT_SEEDS = (42, 123, 456, 789, 1024)

# Rank sweep
METHODOLOGY_RANKS = (4, 8, 16, 32, 64)
DEFAULT_RANKS = METHODOLOGY_RANKS
ALPHA_FACTOR = 2              # alpha = 2r

# Common micro-batch default; full FT additionally uses gradient checkpointing.
FULL_FT_MICRO_BATCH = 4
FULL_FT_ACCUM = 4

# Intruder dimensions (methodology Sec. 7 / Shuttleworth et al. 2025 Def. 1)
INTRUDER_K = 10
INTRUDER_EPS_PRIMARY = 0.9
INTRUDER_EPS_SENSITIVITY = (0.6, 0.7, 0.8)
INTRUDER_MATRICES = ("q_proj", "v_proj")

# Flagging thresholds (methodology Sec. 11)
GRAD_SPIKE_FACTOR = 10.0      # > 10x running median grad norm

# Feasible rank window (methodology Sec. 6.4)
WINDOW_MARGIN_PP = 2.0        # max(2, 2*sigma) percentage points

# Frozen upstream commits, verified against the Hugging Face API 2026-10-17.
MODEL_REVISIONS = {
    "llama3.2-1b": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
    "qwen2.5-1.5b": "8faed761d45a263340a0528343f099c05c9a4323",
    "granite3.3-2b": "c15dc681f4b2aeb4e576cdaae5434edfc17ed14d",
    "qwen2.5-3b": "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
}
DATA_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
ENCODING_VERSION = "labelled-pairs-v2"
PROTOCOL_VERSION = "frw-v2"


# --------------------------------------------------------------------------
# Task definitions (methodology Sec. 5.1)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskSpec:
    key: str            # internal name
    glue_name: str      # nyu-mll/glue config name
    num_labels: int
    pair: bool          # premise/hypothesis pair encoding?
    eval_split: str
    text_fields: tuple  # fields fed to the tokenizer


TASKS: tuple = (
    TaskSpec("rte", "rte", 2, True, "validation", ("sentence1", "sentence2")),
    TaskSpec("mnli", "mnli", 3, True, "validation_matched", ("premise", "hypothesis")),
    TaskSpec("sst2", "sst2", 2, False, "validation", ("sentence",)),
)
TASK_KEYS = tuple(t.key for t in TASKS)
TASK_BY_KEY = {t.key: t for t in TASKS}


# --------------------------------------------------------------------------
# Learning-rate rules (methodology Sec. 2.2)
# --------------------------------------------------------------------------
def lora_lr_rule(params_total_b: float) -> float:
    """Pre-registered per-model LoRA learning rate (methodology Sec. 2.2).

    LR_LoRA(m) = 2e-4 * (1.24B / N_m)^(1/3)
    """
    return 2e-4 * (1.24 / params_total_b) ** (1.0 / 3.0)


# Pre-registered values as printed in methodology Sec. 2.2 (authoritative).
DEFAULT_LORA_LR = {
    "llama3.2-1b": 2.00e-4,
    "qwen2.5-1.5b": 1.85e-4,
    "granite3.3-2b": 1.60e-4,
    "qwen2.5-3b": 1.50e-4,
}
# Full fine-tuning: one order of magnitude below LoRA (Biderman et al. 2024).
DEFAULT_FULL_LR = {"llama3.2-1b": 2e-5, "qwen2.5-1.5b": 1.85e-5,
                   "granite3.3-2b": 1.6e-5, "qwen2.5-3b": 1.5e-5}

# The proposal's constant LR, retained for the Sec. 11 sanity comparison.
SANITY_COMPARISON_LR = 2e-4

# Fallback for custom models whose size cannot be estimated.
FALLBACK_LORA_LR = 2e-4
FALLBACK_FULL_LR = 2e-5


# --------------------------------------------------------------------------
# Model registry (methodology Sec. 1.2 / 1.3)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    label: str
    params_total: float     # billions (exact for the four; estimated for custom)
    params_nonemb: float    # billions
    layers: int
    d_model: int
    n_kv: int               # KV heads (GQA)
    head_dim: int
    vocab: int
    license: str
    lora_lr: float = 0.0    # explicit default LoRA LR (0 = look up / compute)
    full_lr: float = 0.0    # explicit default full-FT LR (0 = LoRA LR / 10)
    custom: bool = False    # user-supplied HF id, not one of the four
    params_estimated: bool = False   # params_total is config-based, not official

    @property
    def wv_out(self) -> int:
        """Output width of the v_proj matrix (n_kv * head_dim)."""
        return self.n_kv * self.head_dim

    @property
    def revision(self) -> str:
        return MODEL_REVISIONS[self.key]

    @property
    def default_lora_lr(self) -> float:
        if self.lora_lr:
            return self.lora_lr
        return DEFAULT_LORA_LR.get(self.key) or (
            lora_lr_rule(self.params_total) if self.params_total
            else FALLBACK_LORA_LR
        )

    @property
    def default_full_lr(self) -> float:
        if self.full_lr:
            return self.full_lr
        return DEFAULT_FULL_LR.get(self.key) or self.default_lora_lr / 10.0


MODELS: dict = {
    "llama3.2-1b": ModelSpec(
        key="llama3.2-1b",
        hf_id="meta-llama/Llama-3.2-1B",
        label="Llama-3.2-1B",
        params_total=1.24, params_nonemb=0.97,
        layers=16, d_model=2048, n_kv=8, head_dim=64, vocab=128256,
        license="Llama 3.2 Community",
    ),
    "qwen2.5-1.5b": ModelSpec(
        key="qwen2.5-1.5b",
        hf_id="Qwen/Qwen2.5-1.5B",
        label="Qwen2.5-1.5B",
        params_total=1.54, params_nonemb=1.31,
        layers=28, d_model=1536, n_kv=2, head_dim=128, vocab=151936,
        license="Apache 2.0",
    ),
    "granite3.3-2b": ModelSpec(
        key="granite3.3-2b",
        hf_id="ibm-granite/granite-3.3-2b-base",
        label="Granite-3.3-2B",
        params_total=2.53, params_nonemb=2.43,
        layers=40, d_model=2048, n_kv=8, head_dim=64, vocab=49152,
        license="Apache 2.0",
    ),
    "qwen2.5-3b": ModelSpec(
        key="qwen2.5-3b",
        hf_id="Qwen/Qwen2.5-3B",
        label="Qwen2.5-3B",
        params_total=3.09, params_nonemb=2.77,
        layers=36, d_model=2048, n_kv=2, head_dim=128, vocab=151936,
        license="Qwen Research License",
    ),
}
DEFAULT_MODEL_KEYS = tuple(MODELS.keys())


# --------------------------------------------------------------------------
# Experiment specification
# --------------------------------------------------------------------------
@dataclass
class ExperimentSpec:
    """Everything needed to plan and execute a set of runs."""

    models: Sequence[str] = field(default_factory=lambda: list(DEFAULT_MODEL_KEYS))
    track: str = "both"                      # "lora" | "full" | "both"
    ranks: Sequence[int] = field(default_factory=lambda: list(DEFAULT_RANKS))
    lora_lrs: Optional[dict] = None          # model key -> [float]; None = defaults
    full_lrs: Optional[dict] = None          # model key -> [float]; None = defaults
    seeds: Sequence[int] = field(default_factory=lambda: list(DEFAULT_SEEDS))
    output_dir: str = "results"
    mode: str = "main"                       # main | pilot | sanity | smoke
    force: bool = False                      # rerun completed runs
    deterministic: bool = False
    save_full_weights: bool = False          # Track B weight snapshots (large!)
    intruder_ref: str = "all"
    intruder_side: str = "left"              # singular-vector side for Def. 1
    micro_batch: Optional[int] = None        # one partition for all tracks/ranks
    accum: Optional[int] = None              # micro * accum must be 16
    train_samples: int = TRAIN_SAMPLES
    epochs: int = EPOCHS
    optimizer: str = "standard"  # LoRA fp32 moments, full FT 8-bit matrix moments
    eval_batch: int = EVAL_BATCH
    num_shards: int = 1
    shard_index: int = 0
    hourly_rate: Optional[float] = None       # total instance rate, including all GPUs
    budget: Optional[float] = None           # optional per-invocation scheduling guard
    reserve: float = 0.0
    max_hours: Optional[float] = None
    fail_fast: bool = True

    def __post_init__(self):
        if not self.models or any(m not in MODELS for m in self.models):
            raise ValueError("Choose only the four fixed methodology models")
        if self.track not in ("lora", "full", "both"):
            raise ValueError("invalid track")
        if self.mode not in ("main", "pilot", "sanity", "smoke"):
            raise ValueError("invalid mode")
        if not self.ranks or any(r <= 0 or r > min(MODELS[m].wv_out for m in self.models)
                                 for r in self.ranks):
            raise ValueError("rank must be positive and no greater than the narrowest v projection")
        if not self.seeds or any(s < 0 or s >= 2**32 for s in self.seeds):
            raise ValueError("seeds must be in [0, 2**32)")
        if self.train_samples <= 0 or self.epochs <= 0 or self.eval_batch <= 0:
            raise ValueError("sample, epoch and evaluation batch counts must be positive")
        if self.micro_batch is None:
            self.micro_batch = FULL_FT_MICRO_BATCH
        if self.micro_batch <= 0 or TRAIN_BATCH % self.micro_batch:
            raise ValueError("micro-batch must be a positive divisor of 16")
        if self.accum is None:
            self.accum = TRAIN_BATCH // self.micro_batch
        if self.micro_batch * self.accum != TRAIN_BATCH:
            raise ValueError("micro-batch × accumulation must equal 16")
        if self.optimizer not in ("standard", "adamw"):
            raise ValueError("invalid optimizer")
        if self.intruder_ref != "all" or self.intruder_side != "left":
            raise ValueError("The study uses all pretrained left singular vectors")
        if self.num_shards <= 0 or not 0 <= self.shard_index < self.num_shards:
            raise ValueError("require 0 <= shard-index < num-shards")
        for name in ("hourly_rate", "budget", "max_hours"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.reserve < 0 or not math.isfinite(self.reserve):
            raise ValueError("reserve must be finite and nonnegative")
        if self.budget is not None and (self.hourly_rate is None or self.budget <= self.reserve):
            raise ValueError("budget requires --hourly-rate and must exceed reserve")
        for m in self.models:
            for lr in self.resolved_lora_lrs(m) + self.resolved_full_lrs(m):
                if not math.isfinite(lr) or lr <= 0:
                    raise ValueError("learning rates must be finite and positive")

    def resolved_lora_lrs(self, model_key: str) -> list:
        if self.lora_lrs and self.lora_lrs.get(model_key):
            return list(self.lora_lrs[model_key])
        return [MODELS[model_key].default_lora_lr]

    def resolved_full_lrs(self, model_key: str) -> list:
        if self.full_lrs and self.full_lrs.get(model_key):
            return list(self.full_lrs[model_key])
        return [MODELS[model_key].default_full_lr]


@dataclass
class RunSpec:
    """A single training run: one model x track x (rank) x lr x seed."""

    model: str
    track: str                 # "lora" | "full"
    rank: Optional[int]        # None for full fine-tuning
    lr: float
    seed: int
    task_keys: tuple = TASK_KEYS
    mode: str = "main"
    epochs: int = EPOCHS
    train_samples: int = TRAIN_SAMPLES
    optimizer: str = "adamw"

    @property
    def identity(self) -> dict:
        return {**asdict(self), "protocol": PROTOCOL_VERSION,
                "revision": MODEL_REVISIONS[self.model], "data_revision": DATA_REVISION,
                "encoding": ENCODING_VERSION, "max_len": MAX_LEN,
                "batch": TRAIN_BATCH, "warmup": WARMUP_FRAC, "alpha_factor": ALPHA_FACTOR,
                "trainable_dtype": "float32", "compute_dtype": COMPUTE_DTYPE}

    @property
    def alpha(self) -> Optional[int]:
        return ALPHA_FACTOR * self.rank if self.rank is not None else None

    @property
    def run_id(self) -> str:
        rank_part = f"r{self.rank}" if self.rank is not None else "rNA"
        lr_part = f"lr{self.lr:.2e}".replace("+", "")
        model_part = self.model.replace("/", "--").replace(":", "-")
        base = f"{model_part}__{self.track}__{rank_part}__{lr_part}__s{self.seed}"
        if self.mode != "main":
            base += f"__{self.mode}"
        digest = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()[:12]
        return f"{base}__{digest}"
