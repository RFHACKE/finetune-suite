"""Immutable experiment controls and Linux process locks for independent GPU workers."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import platform

from . import config, utils


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_digest() -> str:
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


@contextmanager
def file_lock(path: Path, blocking=True, shared=False):
    """Lock files persist; kernel locks disappear automatically after a crash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
                        | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def immutable_json(path: Path, payload):
    # Compare the on-disk JSON representation: integer dict keys become strings,
    # and tuples become lists. Otherwise a legitimate second invocation fails.
    payload = json.loads(json.dumps(payload, sort_keys=True, default=utils._json_default, allow_nan=False))
    with file_lock(path.with_suffix(".lock")):
        previous = utils.load_json(path)
        if previous is not None and previous != payload:
            raise ValueError(f"Experiment controls changed: {path}. Use a new --output directory; "
                             "existing results cannot be silently mixed or reused.")
        if previous is None:
            utils.save_json(path, payload)


def experiment_controls(spec) -> dict:
    # Grid subsets, sharding and price do not change a run's scientific controls.
    smoke = spec.mode == "smoke"
    sanity = spec.mode == "sanity"
    calibration = sanity or spec.mode == "lr_test"
    selected = 64 if smoke else spec.train_samples
    return {
        "protocol": config.PROTOCOL_VERSION, "source_sha256": source_digest(),
        "cohort": spec.mode if smoke or calibration else "main",
        "model_revisions": config.MODEL_REVISIONS, "data_revision": config.DATA_REVISION,
        "task_order": ["rte"] if sanity else list(config.TASK_KEYS), "train_samples": selected,
        "train_samples_used": int(.8 * selected) if calibration else selected,
        "evaluation_policy": "train_holdout_20percent" if calibration else "official_validation",
        "evaluation_limit": 32 if smoke else None,
        "data_seed": config.DATA_SEED, "epochs": 1 if smoke else spec.epochs,
        "max_len": config.MAX_LEN, "encoding": config.ENCODING_VERSION,
        "optimizer": spec.optimizer, "betas": list(config.ADAM_BETAS),
        "optimizer_moment_policy": ("lora_all_32bit_full_matrices_8bit_full_heads_vectors_32bit"
                                    if spec.optimizer == "standard" else "all_32bit"),
        "adam_eps": config.ADAM_EPS, "weight_decay": config.WEIGHT_DECAY,
        "clip": config.GRAD_CLIP, "batch": config.TRAIN_BATCH,
        "micro_batch": spec.micro_batch, "accum": spec.accum, "eval_batch": spec.eval_batch,
        "warmup_frac": config.WARMUP_FRAC, "schedule": "cosine_per_task_fresh_optimizer",
        "alpha_factor": config.ALPHA_FACTOR, "lora_init": "kaiming_A_zero_B",
        "rank_interpretation": ("overcomplete_factor_width_allowed"
                                if spec.allow_overcomplete_ranks else "bounded_by_narrowest_v"),
        "epoch_diagnostics": {"enabled": spec.epoch_diagnostics,
                              "samples_per_split": spec.diagnostic_samples,
                              "selection": "fixed_stratified_seed42",
                              "rng_isolation": True, "model_selection": False,
                              "nll_rise_threshold": 0.02, "train_nll_fall_threshold": 0.02},
        "targets": list(config.INTRUDER_MATRICES), "dropout": config.DROPOUT,
        "dtype": config.COMPUTE_DTYPE, "trainable_storage": "float32",
        "normalization_storage": "float32_both_tracks", "autocast_weight_cache": False,
        "attention": config.ATTN_IMPL, "deterministic": spec.deterministic,
        "frozen_embeddings": True, "pooling": "last_nonpad_head_fp32",
        "learning_rate_defaults": spec.default_policy(),
        "intruder_ref": "all", "intruder_side": "left", "intruder_k": config.INTRUDER_K,
        "intruder_eps": [0.6, 0.7, 0.8, 0.9],
        "required_seeds": list(config.DEFAULT_SEEDS),
        "required_ranks": list(config.METHODOLOGY_RANKS),
        "model_registry": {k: asdict(m) for k, m in config.MODELS.items()},
    }


def register_invocation(root: Path, spec, environment: dict) -> str:
    controls = experiment_controls(spec)
    immutable_json(root / "protocol.json", controls)
    fingerprint = digest(controls)
    software = {"python": platform.python_version(), "packages": environment["packages"],
                "cuda_runtime": environment["cuda"].get("torch_version")}
    immutable_json(root / "software.json", software)
    # Every invocation gets a new file; never overwrite the provenance of an earlier batch.
    import uuid
    utils.save_json(root / "invocations" / f"{uuid.uuid4().hex}.json", {
        "experiment": asdict(spec), "environment": environment,
        "protocol_sha256": fingerprint, "pip_freeze": utils.pip_freeze(),
    })
    return fingerprint


def register_hardware(root: Path, model_key: str) -> dict:
    import torch
    i = torch.cuda.current_device()
    hardware = {"name": torch.cuda.get_device_name(i),
                "capability": list(torch.cuda.get_device_capability(i)),
                "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory}
    immutable_json(root / "hardware" / f"{model_key}.json", hardware)
    return hardware
