"""Protocol frw-v4: rank x learning-rate decoupled LoRA forgetting study.

v4 replaces the v3 constant multiplier (alpha = 2r, one LR per model) with the
standard fixed-alpha parameterisation (scale = alpha / r) plus an explicit
per-rank learning-rate search on a training-only holdout. Under alpha = 2r the
optimal LR shrinks as rank grows (Chen, Villar & Hayou 2026, muA), so a single
LR calibrated at r=16 was several times too large at r >= 128 and those runs
never learned (prior cohorts: chance accuracy on every task). See
docs/PROTOCOL_V4.md for the full rationale.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

PROTOCOL = "frw-v4"

# ---------------------------------------------------------------------------
# Data (unchanged from v3 so data selection is directly comparable)
# ---------------------------------------------------------------------------
DATA_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"   # nyu-mll/glue
DATA_SEED = 42
TRAIN_SAMPLES = 2490          # RTE's full training split; same size for every task
HOLDOUT_FRACTION = 0.2        # LR-search split inside the 2,490 (1,992 fit / 498 holdout)
MAX_LEN = 256
ENCODING = "labelled-pairs-v2"  # "Premise: ... \nHypothesis: ..." / "Text: ..."


@dataclass(frozen=True)
class Task:
    key: str
    glue: str
    labels: int
    pair: bool
    eval_split: str
    fields: tuple


TASKS = {
    "rte": Task("rte", "rte", 2, True, "validation", ("sentence1", "sentence2")),
    "mnli": Task("mnli", "mnli", 3, True, "validation_matched", ("premise", "hypothesis")),
    "sst2": Task("sst2", "sst2", 2, False, "validation", ("sentence",)),
}

# Primary order is the proposal's. O2/O3 complete a cyclic Latin square so each
# task occupies each position once (task-order robustness).
ORDERS = {
    "O1": ("rte", "mnli", "sst2"),
    "O2": ("mnli", "sst2", "rte"),
    "O3": ("sst2", "rte", "mnli"),
}

# ---------------------------------------------------------------------------
# Optimisation (identical for every rank)
# ---------------------------------------------------------------------------
EPOCHS = 3
BATCH = 16
WARMUP_FRAC = 0.06
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.0
BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
HEAD_LR = 1e-3                # fixed for every rank/LR so the head never co-varies with rank
EVAL_BATCH = 64

# LoRA parameterisation: fixed alpha, scale alpha/r (PEFT default semantics).
LORA_ALPHA = 32
LORA_TARGETS = ("q_proj", "v_proj")
# Extension cohort (2026-10-25): LoRA on every linear layer of every block.
ALL_LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LORA_DROPOUT = 0.0

RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
LORA_LR_GRID = (1e-4, 2e-4, 4e-4, 8e-4)        # factor-2 grid, extended adaptively at edges
LORA_LR_MIN, LORA_LR_MAX = 1.25e-5, 6.4e-3     # bounds for adaptive extension (<=3 steps out)
FULL_LR_GRID = (1e-5, 2e-5, 4e-5)
FULL_LR_MIN, FULL_LR_MAX = 1.25e-6, 3.2e-4

TUNE_SEEDS = (11, 22)                          # LR search only (holdout evaluation)
MAIN_SEEDS = (42, 123, 456, 789, 1024)         # reported results (official validation)
ORDER_SEEDS = (42, 123, 456)                   # task-order robustness cohort
ORDER_RANKS = (1, 4, 16, 64, 256, 1024)
# Extension cohorts (pre-registered in docs/PROTOCOL_V4_EXT.md before launch).
STAB_SEEDS = (2024, 3033, 4044, 5055, 6066)        # fresh seeds, never used for selection
STAB_RANKS = (16, 64, 128, 256, 512, 1024)
AL_RANKS = (4, 16, 64, 256)

INTRUDER_K = 10
INTRUDER_EPS = (0.5, 0.6, 0.7, 0.8, 0.9)


# ---------------------------------------------------------------------------
# Models (pinned commits, verified 2026-10-17 for v3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Model:
    key: str
    hf_id: str
    revision: str
    label: str
    params_b: float
    layers: int
    d_model: int
    q_out: int
    v_out: int
    family: str
    gated: bool = False


MODELS = {
    "llama3.2-1b": Model("llama3.2-1b", "meta-llama/Llama-3.2-1B",
                         "4e20de362430cd3b72f300e6b0f18e50e7166e08", "Llama-3.2-1B",
                         1.24, 16, 2048, 2048, 512, "llama", gated=True),
    "qwen2.5-1.5b": Model("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B",
                          "8faed761d45a263340a0528343f099c05c9a4323", "Qwen2.5-1.5B",
                          1.54, 28, 1536, 1536, 256, "qwen"),
    "granite3.3-2b": Model("granite3.3-2b", "ibm-granite/granite-3.3-2b-base",
                           "c15dc681f4b2aeb4e576cdaae5434edfc17ed14d", "Granite-3.3-2B",
                           2.53, 40, 2048, 2048, 512, "granite"),
    "qwen2.5-3b": Model("qwen2.5-3b", "Qwen/Qwen2.5-3B",
                        "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b", "Qwen2.5-3B",
                        3.09, 36, 2048, 2048, 256, "qwen"),
}
MODEL_ORDER = tuple(MODELS)


def head_seed(seed: int, position: int) -> int:
    """Head init depends only on (run seed, task position): identical across ranks/LRs."""
    return (seed * 1000 + position + 1) % (2**31)


def shuffle_seed(seed: int, position: int) -> int:
    return seed * 100 + position


def stable_hash(obj, n=12) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:n]


def fmt_lr(lr: float) -> str:
    return f"{lr:.2e}".replace("+", "")
