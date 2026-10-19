"""Independent SST-2 epoch budgets: four models, six budgets, two seeds.

Each fit starts from the pinned backbone with a fresh optimizer and a cosine
schedule planned for its entire budget. No model checkpoints are exported.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time
import traceback

from rank_suite import config, utils
from rank_suite.protocol import digest, immutable_json, source_digest

HERE = Path(__file__).resolve().parent
MODELS = tuple(config.DEFAULT_MODEL_KEYS)
SEEDS = (42, 123)
EPOCH_BUDGETS = tuple(range(1, 7))
STEPS_PER_EPOCH = 156
RANK = 16
SCOPES = {"sst2": ("sst2",)}


def study_source_digest():
    return digest(
        {"suite": source_digest(), "study": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    )


def protocol(scope):
    if scope not in SCOPES:
        raise ValueError(f"Unknown scope: {scope}")
    return {
        "schema": "independent-epoch-budget-study-v1",
        "scope": scope,
        "task_order": list(SCOPES[scope]),
        "models": list(MODELS),
        "model_revisions": config.MODEL_REVISIONS,
        "data_revision": config.DATA_REVISION,
        "source_sha256": study_source_digest(),
        "suite_source_sha256": source_digest(),
        "epoch_budgets": list(EPOCH_BUDGETS),
        "expected_runs": len(MODELS) * len(SEEDS) * len(EPOCH_BUDGETS),
        "expected_epoch_observations": len(MODELS) * len(SEEDS) * sum(EPOCH_BUDGETS),
        "expected_optimizer_steps": len(MODELS) * len(SEEDS) * sum(EPOCH_BUDGETS) * STEPS_PER_EPOCH,
        "seeds": list(SEEDS),
        "rank": RANK,
        "learning_rates": config.DEFAULT_LORA_LR,
        "head_lr_multipliers": config.DEFAULT_LORA_HEAD_MULTIPLIER,
        "train_samples_per_task": config.TRAIN_SAMPLES,
        "data_seed": config.DATA_SEED,
        "train_batch": 16,
        "micro_batch": 16,
        "accum": 1,
        "eval_batch": 32,
        "max_len": config.MAX_LEN,
        "encoding": config.ENCODING_VERSION,
        "optimizer": "adamw",
        "optimizer_storage": "all_fp32",
        "betas": list(config.ADAM_BETAS),
        "eps": config.ADAM_EPS,
        "weight_decay": config.WEIGHT_DECAY,
        "dropout": config.DROPOUT,
        "clip": config.GRAD_CLIP,
        "dtype": config.COMPUTE_DTYPE,
        "targets": list(config.INTRUDER_MATRICES),
        "alpha": config.ALPHA_FACTOR * RANK,
        "attention": config.ATTN_IMPL,
        "deterministic_algorithms": False,
        "schedule": {
            "policy": "fresh_budget_matched_cosine_schedule_per_fit",
            "warmup_fraction": config.WARMUP_FRAC,
            "total_steps": "ceil(2490/16) * epoch_budget = 156 * epoch_budget",
            "warmup_steps": "ceil(0.03 * total_steps)",
            "reason": "Compare independently completed training policies. Duration and its matched cosine schedule vary together.",
        },
        "initialization": "fresh pinned backbone, adapter, heads, optimizer, scheduler and shuffle generator per model/seed/budget; matching initial adapter/head checksums across budgets within a seed",
        "task_head_seed_positions": list(config.TASK_KEYS),
        "shuffle_seed": "run_seed * 100 + original_task_position",
        "observations": {
            "full_training_set_every_epoch": True,
            "full_official_validation_every_epoch": True,
            "n_train": 2490,
            "n_validation": 872,
            "rng_isolation": True,
            "adaptive_changes": False,
            "saved": ["JSON", "JSONL", "per-example probabilities/labels/NLL NPZ"],
            "not_saved": ["model weights", "adapter weights", "optimizer states", "projection tensors"],
        },
        "analysis": {
            "primary": "full SST-2 validation NLL at the FINAL epoch of each independent budget, mean across two paired seeds",
            "secondary": [
                "validation accuracy",
                "training NLL",
                "generalization gap",
                "confidence on wrong predictions",
                "Brier score",
            ],
            "paired_reference_epoch_budget": 3,
            "planned_later_comparisons": [4, 5, 6],
            "practical_nll_rise": 0.02,
            "overfitting_pattern": "endpoint validation NLL rises while same-split endpoint training NLL falls versus budget3",
            "consistent_pattern": "mean endpoint validation NLL rise >0.02 and paired overfitting direction in both seeds; a descriptive screening rule, not a significance test",
            "uncertainty": "individual paired seeds and sample SD (n=2); no confidence interval or universal optimum claim",
            "selection_caveat": "Previously inspected official validation sets are reused for this planned exploratory comparison. No untouched test-set claim.",
            "schedule_caveat": "Each budget has its own cosine horizon. Intermediate observations are for learning curves only and are never pooled into endpoint budget comparisons. This estimates duration plus its matched schedule, not duration at a fixed learning-rate path.",
            "scope_caveat": "SST-2 only, rank16, two seeds. Does not determine optimal epochs for RTE/MNLI, other ranks, or continual retention. Global suite defaults are unchanged.",
            "hypothesis": "Budget3 has the lowest observed mean validation NLL and budgets4-6 show overfitting; evaluate, do not assume.",
        },
    }


def expected_grid(scope):
    return [
        {
            "run_id": f"{model}-e{budget}-s{seed}",
            "model": model,
            "epoch_budget": budget,
            "training_epochs": budget,
            "seed": seed,
            "rank": RANK,
            "task_order": list(SCOPES[scope]),
            "lr": config.DEFAULT_LORA_LR[model],
            "head_lr_multiplier": config.DEFAULT_LORA_HEAD_MULTIPLIER[model],
        }
        for model in MODELS
        for seed in SEEDS
        for budget in EPOCH_BUDGETS
    ]


@contextmanager
def observation_rng(device):
    import torch

    devices = [torch.cuda.current_device()] if str(device).startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        yield


def parameter_digest(parameters):
    """Small initialization checksum, not a weight export."""
    h = hashlib.sha256()
    for name, value in parameters:
        if value.requires_grad:
            h.update(name.encode())
            h.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def reusable(path, protocol_sha):
    result = utils.load_json(path)
    if result and result.get("protocol_sha256") != protocol_sha:
        raise ValueError(f"Existing run has different controls: {path}")
    return bool(result and result.get("status") == "completed")


def run_fit(identity, control, data, tokenizer, model_root, flush, deadline):
    import torch
    from rank_suite.data import make_loader
    from rank_suite.model_utils import backbone_transformer, load_backbone, make_task_heads, wrap_lora
    from rank_suite.train import _train_task, evaluate

    started = time.monotonic()
    protocol_sha = digest(control)
    root = Path(model_root) / "runs" / identity["run_id"]
    attempt = root / "attempts" / str(time.time_ns())
    attempt.mkdir(parents=True, exist_ok=False)
    result = {
        **identity,
        "status": "running",
        "protocol_sha256": protocol_sha,
        "attempt": str(attempt.relative_to(root)),
        "started_unix": time.time(),
    }
    utils.save_json(root / "result.json", result)
    flush()
    utils.seed_all(identity["seed"], deterministic=False)
    torch.cuda.reset_peak_memory_stats()
    mspec = config.MODELS[identity["model"]]
    model, _ = load_backbone(mspec)
    # Preserve original task-position head seeds even in the SST-2-only study.
    all_heads = make_task_heads(config.TASK_KEYS, mspec.d_model, identity["seed"], "cpu")
    trained = wrap_lora(model, RANK)
    initial = {
        "adapter_sha256": parameter_digest(trained.named_parameters()),
        "heads_sha256": parameter_digest(all_heads.named_parameters()),
    }
    paired = Path(model_root) / "initialization" / f"seed{identity['seed']}.json"
    immutable_json(paired, initial)
    utils.save_json(attempt / "initialization.json", initial)
    trained.to("cuda")
    all_heads.to("cuda")
    # Collect live Parameter objects after device transfer.
    adapter_parameters = [p for p in trained.parameters() if p.requires_grad]
    transformer = backbone_transformer(trained)
    step = 0
    summaries, boundaries = {}, []

    def observe(head, dataset, path):
        loader = make_loader(dataset, config.EVAL_BATCH, False, config.DATA_SEED, tokenizer)
        with observation_rng("cuda"):
            metrics = evaluate(transformer, head, loader, "cuda", path, return_details=True, save_scores=True)
        return {**metrics, "count": len(dataset), "predictions": str(path.relative_to(root))}

    for task_index, key in enumerate(control["task_order"]):
        original_index = config.TASK_KEYS.index(key)
        head = all_heads[original_index]
        for index, other in enumerate(all_heads):
            other.requires_grad_(index == original_index)
        td = data[key]
        observation_seconds = []

        def callback(epoch, global_step, online_train_loss):
            observation_started = time.monotonic()
            if time.monotonic() >= deadline:
                raise TimeoutError("Study worker time allowance reached; partial observations preserved")
            row = {
                **identity,
                "task": key,
                "epoch": epoch,
                "step": global_step,
                "online_train_loss": online_train_loss,
                "selection": False,
                "timestamp_unix": time.time(),
            }
            for split, dataset in (("train", td.train_dataset), ("validation", td.eval_dataset)):
                path = attempt / "predictions" / key / f"epoch{epoch}_{split}_full.npz"
                row[split] = observe(head, dataset, path)
            row["nll_gap"] = row["validation"]["nll"] - row["train"]["nll"]
            row["accuracy_gap"] = row["train"]["accuracy"] - row["validation"]["accuracy"]
            row["observation_s"] = time.monotonic() - observation_started
            utils.append_jsonl(attempt / "epoch_metrics.jsonl", row)
            utils.save_json(
                Path(model_root) / "progress.json",
                {
                    "status": "running",
                    "run_id": identity["run_id"],
                    "task": key,
                    "epoch": epoch,
                    "training_epochs": identity["training_epochs"],
                    "epoch_budget": identity["epoch_budget"],
                    "updated_unix": time.time(),
                },
            )
            flush()
            observation_seconds.append(time.monotonic() - observation_started)

        loader = make_loader(td.train_dataset, 16, True, identity["seed"] * 100 + original_index, tokenizer)
        params = adapter_parameters + list(head.parameters())
        training = _train_task(
            transformer,
            head,
            params,
            loader,
            identity["training_epochs"],
            identity["lr"],
            "adamw",
            "cuda",
            attempt / "train_steps.jsonl",
            step,
            key,
            accum=1,
            head_lr_multiplier=identity["head_lr_multiplier"],
            epoch_callback=callback,
        )
        step = training["global_step"]
        rows = [json.loads(line) for line in (attempt / "epoch_metrics.jsonl").read_text().splitlines()]
        endpoint = rows[-1]
        training["observation_and_flush_s"] = sum(observation_seconds)
        training["optimization_s_estimate"] = training["elapsed_s"] - sum(observation_seconds)
        summaries[key] = training
        boundaries.append(endpoint)
        flush()
        if time.monotonic() >= deadline:
            raise TimeoutError("Study worker allowance reached after a task; scalar data preserved")

    result.update(
        {
            "status": "completed",
            "finished_unix": time.time(),
            "elapsed_s": time.monotonic() - started,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
            "initialization": initial,
            "final_adapter_sha256": parameter_digest(trained.named_parameters()),
            "train_summary": summaries,
            "boundaries": boundaries,
            "epoch_metrics": str((attempt / "epoch_metrics.jsonl").relative_to(root)),
        }
    )
    utils.save_json(root / "result.json", result)
    return result


def run_model(model, scope, output, seconds, flush=lambda: None, retry_failed=False):
    import torch
    from rank_suite.data import prepare_task_data
    from rank_suite.model_utils import load_tokenizer
    from rank_suite.train import _cleanup

    begin = time.monotonic()
    deadline = begin + seconds - 45
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    control = protocol(scope)
    immutable_json(output / "protocol.json", control)
    protocol_sha = digest(control)
    utils.get_logger("rank_suite")
    utils.save_json(output / "environment.json", utils.capture_environment())
    utils.save_json(
        output / "hardware.json",
        {
            "name": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
    )
    tokenizer = load_tokenizer(config.MODELS[model])
    data = prepare_task_data(tokenizer, task_keys=SCOPES[scope], selection_dir=output / "selection")
    grid = [r for r in expected_grid(scope) if r["model"] == model]
    returned, failed, deferred = [], [], []
    for identity in grid:
        result_path = output / "runs" / identity["run_id"] / "result.json"
        if reusable(result_path, protocol_sha):
            returned.append(identity["run_id"])
            continue
        previous = utils.load_json(result_path)
        if previous and not retry_failed:
            failed.append(
                {
                    "run_id": identity["run_id"],
                    "reason": "Prior incomplete/failed attempt requires explicit retry_failed",
                }
            )
            continue
        if time.monotonic() + 120 >= deadline:
            deferred.append(identity["run_id"])
            continue
        try:
            run_fit(identity, control, data, tokenizer, output, flush, deadline)
            returned.append(identity["run_id"])
        except Exception as exc:
            failure = {
                **identity,
                "status": "failed",
                "protocol_sha256": protocol_sha,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "finished_unix": time.time(),
            }
            prior = utils.load_json(result_path, {})
            failure["attempt"] = prior.get("attempt")
            utils.save_json(result_path, failure)
            failed.append({"run_id": identity["run_id"], "error": repr(exc)})
        finally:
            _cleanup()
            flush()
    report = {
        "model": model,
        "status": "completed" if len(returned) == len(grid) else "incomplete",
        "expected": len(grid),
        "completed": returned,
        "failed": failed,
        "deferred": deferred,
        "worker_elapsed_s": time.monotonic() - begin,
        "protocol_sha256": protocol_sha,
        "updated_unix": time.time(),
    }
    utils.save_json(output / "worker_summary.json", report)
    flush()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=SCOPES, default="sst2")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--seconds", type=int, default=3000)
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.protocol_only:
        args.output.mkdir(parents=True, exist_ok=True)
        immutable_json(args.output / "protocol.json", protocol(args.scope))
        immutable_json(args.output / "expected_grid.json", expected_grid(args.scope))
    elif not args.model:
        parser.error("--model is required for training")
    else:
        print(
            json.dumps(
                run_model(args.model, args.scope, args.output, args.seconds, retry_failed=args.retry_failed),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
