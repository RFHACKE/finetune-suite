"""Sequential training with exact accumulation and task-boundary recovery.

GPU work ends after training/evaluation/artifact export. Spectral analysis
runs separately on a CPU machine. An interrupted task restarts from the last
completed boundary; optimizer moments and schedules intentionally reset per task.
"""
from __future__ import annotations

import contextlib
import gc
import json
import logging
import math
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F

from . import config, metrics, utils
from .data import make_loader
from .intruder import export_base_projections, export_boundary_projections, save_torch
from .model_utils import (backbone_transformer, forward_logits, load_backbone,
                          make_task_heads, prepare_full_ft, total_param_count,
                          trainable_param_count, wrap_lora)

logger = logging.getLogger("rank_suite.train")


def _autocast(device):
    return (torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
            if str(device).startswith("cuda") else contextlib.nullcontext())


def _to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def make_optimizer(params, lr, optimizer, head_params=()):
    if optimizer == "adamw8bit":
        import bitsandbytes as bnb
        params = list(params)
        head_ids = {id(p) for p in head_params}
        # Full-FT moment precision follows tensor role. Dense matrices use
        # 8-bit states; heads and normalization/bias vectors stay in fp32,
        # independent of model width or the classification label count.
        # Exactly one optimizer is live per worker. Clear stale per-index overrides
        # before registering the next task's parameters with the public manager API.
        manager = bnb.optim.GlobalOptimManager.get_instance()
        manager.initialize()
        protected = [p for p in params if p.ndim < 2 or id(p) in head_ids]
        if protected:
            manager.override_config(protected, "optim_bits", 32)
        manager.register_parameters(params)
        return bnb.optim.AdamW8bit(params, lr=lr, betas=config.ADAM_BETAS,
                                  eps=config.ADAM_EPS, weight_decay=config.WEIGHT_DECAY,
                                  min_8bit_size=0, block_wise=True,
                                  percentile_clipping=100, is_paged=False)
    if optimizer != "adamw":
        raise ValueError(f"Unknown optimizer: {optimizer}")
    return torch.optim.AdamW(params, lr=lr, betas=config.ADAM_BETAS,
                             eps=config.ADAM_EPS, weight_decay=config.WEIGHT_DECAY,
                             foreach=False)


@torch.inference_mode()
def evaluate(transformer, head, loader, device, prediction_path=None):
    import numpy as np
    was_training, head_training = transformer.training, head.training
    transformer.eval()
    head.eval()
    predictions, labels = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        with _autocast(device):
            logits = forward_logits(transformer, head, batch)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite evaluation logits")
        predictions.append(logits.argmax(dim=-1).cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
    if not labels:
        raise ValueError("Empty evaluation set")
    pred, gold = np.concatenate(predictions), np.concatenate(labels)
    if prediction_path is not None:
        Path(prediction_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prediction_path, predictions=pred, labels=gold)
    transformer.train(was_training)
    head.train(head_training)
    return float((pred == gold).mean())


def _train_task(transformer, head, params, loader, epochs, lr, optimizer,
                device, log_path, global_step, task_key, accum=1):
    from transformers import get_cosine_schedule_with_warmup
    transformer.train()
    head.train()
    opt = make_optimizer(params, lr, optimizer, head.parameters())
    steps_per_epoch = math.ceil(len(loader) / accum)
    total_steps = steps_per_epoch * epochs
    warmup = math.ceil(config.WARMUP_FRAC * total_steps)
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)
    initial_step = global_step
    flags = {"nan_loss_steps": [], "grad_spikes": []}
    history, epoch_losses = [], []
    start = time.monotonic()
    for epoch in range(epochs):
        total_loss = total_examples = 0
        group_loss = group_examples = 0
        opt.zero_grad(set_to_none=True)
        for bi, batch in enumerate(loader):
            batch = _to_device(batch, device)
            # Exact sample mean over this optimizer batch, including the final
            # 10 examples (8+2, 4+4+2, etc.). Dividing each micro loss by accum
            # overweights small last micro-batches and underweights a partial group.
            first_micro = (bi // accum) * accum
            group_size = min(accum * loader.batch_size,
                             len(loader.dataset) - first_micro * loader.batch_size)
            with _autocast(device):
                logits = forward_logits(transformer, head, batch)
                loss_sum = F.cross_entropy(logits.float(), batch["labels"], reduction="sum")
            if not torch.isfinite(loss_sum):
                raise FloatingPointError(f"Nonfinite loss in {task_key}, epoch {epoch}, batch {bi}")
            (loss_sum / group_size).backward()
            n = batch["labels"].numel()
            value = float(loss_sum.detach())
            total_loss += value
            total_examples += n
            group_loss += value
            group_examples += n
            if (bi + 1) % accum == 0 or bi + 1 == len(loader):
                norm = torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP,
                                                      error_if_nonfinite=True)
                gn = float(norm)
                if len(history) >= 10:
                    median = statistics.median(history[-50:])
                    if median > 0 and gn > config.GRAD_SPIKE_FACTOR * median:
                        flags["grad_spikes"].append(global_step + 1)
                history.append(gn)
                used_lr = opt.param_groups[0]["lr"]
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                utils.append_jsonl(log_path, {
                    "task": task_key, "epoch": epoch + 1, "step": global_step,
                    "loss": group_loss / group_examples, "examples": group_examples,
                    "lr": used_lr, "grad_norm": gn,
                })
                group_loss = group_examples = 0
                if (global_step - initial_step) % 50 == 0:
                    logger.info("%s epoch %d step %d/%d loss %.4f", task_key,
                                epoch + 1, global_step - initial_step, total_steps,
                                total_loss / total_examples)
        epoch_losses.append(total_loss / total_examples)
    return {"epoch_losses": epoch_losses, "flags": flags,
            "steps": global_step - initial_step, "global_step": global_step,
            "elapsed_s": time.monotonic() - start}


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _truncate_log(path, step):
    if not path.exists():
        return
    kept = []
    for line in path.read_text().splitlines():
        try:
            obj = json.loads(line)
            if obj["step"] <= step:
                kept.append(line)
        except (ValueError, KeyError):
            pass
    path.write_text("\n".join(kept) + ("\n" if kept else ""))


def run_single(run, run_dir, data, tokenizer, mspec, *, experiment,
               protocol_sha, hardware=None, device="cuda"):
    try:
        return _execute(run, Path(run_dir), data, tokenizer, mspec,
                        experiment, protocol_sha, hardware or {}, device)
    finally:
        _cleanup()


def _execute(run, run_dir, data, tokenizer, mspec, experiment, protocol_sha, hardware, device):
    run_dir.mkdir(parents=True, exist_ok=True)
    timer = utils.Timer()
    utils.seed_all(run.seed, experiment.deterministic)
    identity = json.loads(json.dumps(run.identity))
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model, _ = load_backbone(mspec)
    export_base_projections(model, run_dir.parents[1], mspec)
    heads = make_task_heads(run.task_keys, mspec.d_model, run.seed, "cpu")
    if run.track == "lora":
        trained_model = wrap_lora(model, run.rank)
    else:
        trained_model = prepare_full_ft(model)
    base_params = [p for p in trained_model.parameters() if p.requires_grad]
    if any(p.dtype != torch.float32 for p in base_params):
        raise ValueError("Every trainable parameter must be stored in fp32")
    if str(device).startswith("cuda"):
        free, _ = torch.cuda.mem_get_info()
        parameter_bytes = sum(p.numel() * p.element_size() for p in trained_model.parameters())
        state_bytes = sum(p.numel() * (6 if run.optimizer == "adamw8bit" else 12)
                          for p in base_params)
        if parameter_bytes + state_bytes > 0.94 * free:
            raise RuntimeError("Insufficient VRAM for fp32 trainable weights, gradients and optimizer. "
                               "Use a 32/48 GB GPU for the largest full-FT runs; "
                               "reducing the micro-batch cannot remove parameter memory.")
    trained_model.to(device)
    heads.to(device)
    # Device conversion may replace Parameter objects. Optimizers must refer
    # to the live device parameters rather than a pre-transfer CPU list.
    base_params = [p for p in trained_model.parameters() if p.requires_grad]
    transformer = backbone_transformer(trained_model)
    A = [[None] * len(run.task_keys) for _ in run.task_keys]
    summaries, boundaries = {}, []
    flags = {"nan_loss_tasks": [], "grad_spike_steps": []}
    step = next_task = 0
    prior_elapsed = 0.0
    resume_path = run_dir / "resume.pt"
    log_path = run_dir / "train_log.jsonl"
    if resume_path.exists() and not experiment.force:
        state = torch.load(resume_path, map_location="cpu", weights_only=True)
        if state["identity"] != identity or state["protocol_sha256"] != protocol_sha:
            raise ValueError("Resume checkpoint controls differ from the requested run")
        if run.track == "lora":
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(trained_model, state["weights"])
        else:
            trained_model.load_state_dict(state["weights"], strict=True)
        for head, saved in zip(heads, state["heads"]):
            head.load_state_dict(saved)
        A, summaries, boundaries = state["A"], state["train_summary"], state["boundaries"]
        flags, step, next_task = state["flags"], state["step"], state["next_task"]
        prior_elapsed = state["elapsed_s"]
        torch.set_rng_state(state["torch_rng"])
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state
        logger.info("Resume %s after %d tasks", run.run_id, next_task)
    _truncate_log(log_path, step)
    for ti in range(next_task, len(run.task_keys)):
        key = run.task_keys[ti]
        for i, head in enumerate(heads):
            head.requires_grad_(i == ti)
        params = base_params + list(heads[ti].parameters())
        loader = make_loader(data[key].train_dataset, experiment.micro_batch, True,
                             run.seed * 100 + ti, tokenizer)
        out = _train_task(transformer, heads[ti], params, loader, run.epochs,
                          run.lr, run.optimizer, device, log_path, step, key,
                          accum=experiment.accum)
        step = out["global_step"]
        flags["grad_spike_steps"].extend(out["flags"]["grad_spikes"])
        summaries[key] = {"epoch_losses": out["epoch_losses"], "steps": out["steps"],
                          "global_step": step, "train_s": out["elapsed_s"],
                          "grad_spikes": out["flags"]["grad_spikes"]}
        ckpt = run_dir / "checkpoints" / f"task{ti + 1}_{key}"
        ckpt.mkdir(parents=True, exist_ok=True)
        eval_start = time.monotonic()
        for j in range(ti + 1):
            eval_key = run.task_keys[j]
            eval_loader = make_loader(data[eval_key].eval_dataset, experiment.eval_batch,
                                      False, run.seed, tokenizer)
            A[ti][j] = evaluate(transformer, heads[j], eval_loader, device,
                                ckpt / f"predictions_{eval_key}.npz")
        summaries[key]["eval_s"] = time.monotonic() - eval_start
        export_start = time.monotonic()
        sha = export_boundary_projections(trained_model, run.track,
                                          ckpt / "projections.pt", identity)
        save_torch(ckpt / "heads.pt", [h.state_dict() for h in heads])
        if run.track == "lora":
            trained_model.save_pretrained(ckpt, safe_serialization=True)
        elif experiment.save_full_weights:
            save_torch(ckpt / "full_weights.pt", trained_model.state_dict())
        boundaries.append({"task": key, "path": str(ckpt.relative_to(run_dir)),
                           "projections_sha256": sha})
        summaries[key]["export_s"] = time.monotonic() - export_start
        logger.info("%s after %s: %s", run.run_id, key, A[ti])
        if ti + 1 < len(run.task_keys):
            if run.track == "lora":
                from peft import get_peft_model_state_dict
                weights = get_peft_model_state_dict(trained_model)
            else:
                weights = trained_model.state_dict()
            save_torch(resume_path, {
                "identity": identity, "protocol_sha256": protocol_sha,
                "weights": weights, "heads": [h.state_dict() for h in heads],
                "A": A, "train_summary": summaries, "boundaries": boundaries,
                "flags": flags, "step": step, "next_task": ti + 1,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else [],
                "elapsed_s": prior_elapsed + timer.elapsed,
            })
            del weights
    run_metrics = metrics.compute_run_metrics(A, list(run.task_keys))
    sizes = {str(p.relative_to(run_dir)): p.stat().st_size
             for p in (run_dir / "checkpoints").rglob("*") if p.is_file()}
    result = {
        "status": "completed", "run_id": run.run_id, "run_identity": identity,
        "protocol_sha256": protocol_sha, "mode": run.mode,
        "model": mspec.key, "hf_id": mspec.hf_id, "hf_revision": mspec.revision,
        "track": run.track, "rank": run.rank, "alpha": run.alpha,
        "lr": run.lr, "seed": run.seed, "task_order": list(run.task_keys),
        "epochs": run.epochs, "train_samples": run.train_samples,
        "d_model": mspec.d_model, "wv_out": mspec.wv_out,
        "batch": {"effective": config.TRAIN_BATCH, "micro": experiment.micro_batch,
                  "accum": experiment.accum, "eval": experiment.eval_batch},
        "optimizer": run.optimizer, "trainable_storage": "float32",
        "A": A, "metrics": run_metrics, "train_summary": summaries,
        "trainable_params": trainable_param_count(trained_model),
        "params": {"total_params": total_param_count(model),
                   "trainable_params": trainable_param_count(trained_model),
                   "heads_params": sum(p.numel() for p in heads.parameters())},
        "checkpoint_sizes": sizes, "checkpoint_total_bytes": sum(sizes.values()),
        "boundaries": boundaries, "intruder": {}, "intruder_status": "pending_cpu",
        "flags": flags, "hardware": hardware,
        "data": {key: data[key].provenance for key in run.task_keys},
        "train_time_s": round(prior_elapsed + timer.elapsed, 3),
        "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1e9
                         if str(device).startswith("cuda") else None),
    }
    utils.save_json(run_dir / "result.json", result)
    resume_path.unlink(missing_ok=True)
    return result
