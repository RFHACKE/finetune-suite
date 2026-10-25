"""One sequential run: tasks in order, evaluation and diagnostics at every boundary.

Jobs are atomic: a preempted or failed run restarts from the pretrained model.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .data import batches, load_task
from .jobs import Job
from .model import (count, features, load_backbone, lora_modules, make_heads, prepare_full,
                    trainable, transformer_of, wrap_lora)
from .spectral import Reference, aggregate, boundary_metrics, full_ft_metrics


def targets_for(kind: str):
    """All-linear LoRA for the *_al extension cohort; q/v everywhere else."""
    return C.ALL_LINEAR if kind.endswith("_al") else C.LORA_TARGETS


def epochs_for(kind: str) -> int:
    """Epoch-budget cohort kinds look like 'ep2_al'; every other kind uses the protocol's 3 epochs."""
    if kind.startswith("ep") and "_" in kind:
        return int(kind[2:kind.index("_")])
    return C.EPOCHS


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def autocast(device):
    # bf16 compute with fp32 trainable storage on every device (CPU only for tests).
    return torch.autocast(device.type, dtype=torch.bfloat16, cache_enabled=False)


class Context:
    """Per-worker cache: tokenizer, pristine base model, task data and spectral reference."""

    def __init__(self, model_key: str, device, ref_dir: Path | None = None):
        from .model import load_tokenizer
        self.m = C.MODELS[model_key]
        self.device = device
        self.tok = load_tokenizer(self.m)
        self.base = None
        self.data = {}
        self.ref = None
        self.ref_dir = ref_dir

    def base_model(self):
        if self.base is None:
            self.base = load_backbone(self.m).to(self.device)
        return self.base

    def drop_base(self):
        self.base = None
        torch.cuda.empty_cache()

    def task(self, key, split):
        if (key, split) not in self.data:
            self.data[(key, split)] = load_task(key, self.tok, split)
        return self.data[(key, split)]

    def reference(self):
        if self.ref is not None:
            return self.ref
        ref = Reference()
        path = self.ref_dir / f"{self.m.key}.pt" if self.ref_dir else None
        if path is not None and path.exists():
            saved = torch.load(path, map_location=self.device, weights_only=True)
            ref.u = {k: v.float() for k, v in saved["u"].items()}
            ref.norm = saved["norm"]
            ref.topu = {k: v[:, :C.INTRUDER_K].contiguous() for k, v in ref.u.items()}
        else:
            base = self.base_model()
            for name, mod in base.named_modules():
                if name.split(".")[-1] in C.LORA_TARGETS and hasattr(mod, "weight"):
                    ref.add(name, mod.weight)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".tmp{os.getpid()}")
                torch.save({"u": {k: v.half().cpu() for k, v in ref.u.items()}, "norm": ref.norm}, tmp)
                os.replace(tmp, path)
        self.ref = ref
        return ref


@torch.no_grad()
def evaluate(transformer, head, split, pad_id, device, keep_features=False):
    transformer.eval()
    preds, gold, nll, feats, order = [], [], 0.0, [], []
    for ids, mask, labels, pos in batches(split, C.EVAL_BATCH, pad_id, sort=True):
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        with autocast(device):
            f = features(transformer, ids, mask)
        logits = head(f)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite eval logits")
        nll += float(F.cross_entropy(logits, labels, reduction="sum"))
        preds.append(logits.argmax(-1).cpu())
        gold.append(labels.cpu())
        order.extend(pos)
        if keep_features:
            feats.append(f.cpu())
    inv = np.argsort(np.asarray(order))
    p = torch.cat(preds).numpy()[inv]
    g = torch.cat(gold).numpy()[inv]
    out = {"accuracy": float((p == g).mean()), "nll": nll / len(g), "pred": p.astype(np.int8)}
    if keep_features:
        out["features"] = torch.cat(feats)[torch.as_tensor(inv)]
    return out


def fit_probe(xtr, ytr, xte, yte, n_classes, device, l2=1e-3):
    """L2-regularised multinomial logistic regression (LBFGS) on standardised features."""
    xtr, xte = xtr.to(device).double(), xte.to(device).double()
    ytr, yte = torch.as_tensor(ytr, device=device), torch.as_tensor(yte, device=device)
    mu, sd = xtr.mean(0), xtr.std(0).clamp_min(1e-6)
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    W = torch.zeros(xtr.shape[1], n_classes, dtype=torch.float64, device=device, requires_grad=True)
    b = torch.zeros(n_classes, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=300, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(xtr @ W + b, ytr) + l2 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        acc = float(((xte @ W + b).argmax(-1) == yte).double().mean())
        tr_acc = float(((xtr @ W + b).argmax(-1) == ytr).double().mean())
    return acc, tr_acc


def run(job: Job, ctx: Context, out_dir: Path, probes: bool = False, save_adapter: bool = False,
        log=print) -> dict:
    t0 = time.time()
    device = ctx.device
    m = ctx.m
    order = C.ORDERS[job.order]
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(job.seed)
    timing = {"train": 0.0, "eval": 0.0, "spectral": 0.0, "probe": 0.0, "setup": 0.0}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ref = ctx.reference()
    if job.track == "lora":
        base = ctx.base_model()
        model = wrap_lora(base, job.rank, targets_for(job.kind))
        initial = None
    else:
        ctx.drop_base()
        model = prepare_full(load_backbone(m)).to(device)
        initial = {name[:-len(".weight")]: p.detach().to("cpu", torch.bfloat16).clone()
                   for name, p in model.named_parameters()
                   if name.split(".")[-2] in C.LORA_TARGETS and name.endswith(".weight")}
    model.to(device)
    transformer = transformer_of(model)
    heads = make_heads(order, m.d_model, job.seed).to(device)
    base_params = trainable(model)
    pad = ctx.tok.pad_token_id
    T = len(order)
    A = [[None] * T for _ in range(T)]
    NLL = [[None] * T for _ in range(T)]
    P = [[None] * T for _ in range(T)]
    P_train = [[None] * T for _ in range(T)]
    spectral = []
    log_rows = []
    status, error = "completed", None
    data = {k: ctx.task(k, job.split) for k in order}
    timing["setup"] = time.time() - t0
    micro = C.BATCH if job.track == "lora" else 8
    try:
        for i, key in enumerate(order):
            td = data[key]
            for j, h in enumerate(heads):
                h.requires_grad_(j == i)
            groups = [{"params": base_params, "lr": job.lr},
                      {"params": list(heads[i].parameters()), "lr": C.HEAD_LR}]
            opt = _optimizer(groups, job.track)
            n = len(td.train)
            steps_per_epoch = math.ceil(n / C.BATCH)
            total = steps_per_epoch * epochs_for(job.kind)
            warm = math.ceil(C.WARMUP_FRAC * total)
            from transformers import get_cosine_schedule_with_warmup
            sched = get_cosine_schedule_with_warmup(opt, warm, total)
            transformer.train()
            step = 0
            ts = time.time()
            params = base_params + list(heads[i].parameters())
            for ep in range(epochs_for(job.kind)):
                seed_e = C.shuffle_seed(job.seed, i) * 10 + ep
                for ids, mask, labels, _ in batches(td.train, C.BATCH, pad, shuffle_seed=seed_e):
                    loss_val = 0.0
                    for s in range(0, len(labels), micro):
                        mi, mm, ml = (ids[s:s + micro].to(device), mask[s:s + micro].to(device),
                                      labels[s:s + micro].to(device))
                        with autocast(device):
                            f = features(transformer, mi, mm)
                        loss = F.cross_entropy(heads[i](f), ml, reduction="sum") / len(labels)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"nonfinite loss task={key} epoch={ep} step={step}")
                        loss.backward()
                        loss_val += float(loss.detach())
                    gn = float(torch.nn.utils.clip_grad_norm_(params, C.GRAD_CLIP))
                    if not math.isfinite(gn):
                        raise FloatingPointError(f"nonfinite grad task={key} step={step}")
                    lr_now = opt.param_groups[0]["lr"]
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    log_rows.append((i, ep, step, round(loss_val, 6), round(gn, 5), lr_now))
            timing["train"] += time.time() - ts
            del opt
            # ---- boundary evaluation ----
            te = time.time()
            for j in range(i + 1):
                ev = evaluate(transformer, heads[j], data[order[j]].eval, pad, device, keep_features=probes)
                A[i][j], NLL[i][j] = ev["accuracy"], ev["nll"]
                np.savez_compressed(out_dir / f"pred_b{i + 1}_{order[j]}.npz", pred=ev["pred"])
                if probes:
                    tr = evaluate(transformer, heads[j], data[order[j]].train, pad, device, keep_features=True)
                    tp = time.time()
                    P[i][j], P_train[i][j] = fit_probe(tr["features"], data[order[j]].train.labels,
                                                       ev["features"], data[order[j]].eval.labels,
                                                       C.TASKS[order[j]].labels, device)
                    timing["probe"] += time.time() - tp
            timing["eval"] += time.time() - te
            tsp = time.time()
            rows = (boundary_metrics(model, ref, seed=job.seed) if job.track == "lora"
                    else full_ft_metrics(model, initial, ref))
            spectral.append({"boundary": i + 1, "task": key, "summary": aggregate(rows), "matrices": rows})
            timing["spectral"] += time.time() - tsp
            log(f"{job.job_id} after {key}: A={[None if a is None else round(a, 4) for a in A[i]]}")
    except FloatingPointError as exc:
        status, error = "diverged", str(exc)
        log(f"{job.job_id} DIVERGED: {exc}")
    finally:
        if job.track == "lora" and save_adapter and status == "completed":
            from safetensors.torch import save_file
            tensors = {}
            for name, mod in lora_modules(model, targets_for(job.kind)):
                tensors[name + ".lora_A"] = mod.lora_A["default"].weight.detach().to(torch.bfloat16).cpu().contiguous()
                tensors[name + ".lora_B"] = mod.lora_B["default"].weight.detach().to(torch.bfloat16).cpu().contiguous()
            save_file(tensors, str(out_dir / "adapter_final_bf16.safetensors"))
            torch.save([h.state_dict() for h in heads], out_dir / "heads_final.pt")
        n_trainable = count(base_params)
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
        if job.track == "lora":
            ctx.base = model.unload()          # removes LoRA layers; base weights untouched
            if hasattr(ctx.base, "peft_config"):
                del ctx.base.peft_config
        else:
            del model
            ctx.drop_base()
        del heads
        torch.cuda.empty_cache()

    result = {"status": status, "error": error, "job_id": job.job_id, "spec": job.spec,
              "model_hf": m.hf_id, "revision": m.revision, "order_tasks": list(order),
              "alpha": C.LORA_ALPHA if job.track == "lora" else None,
              "lora_targets": list(targets_for(job.kind)) if job.track == "lora" else None,
              "scale": (C.LORA_ALPHA / job.rank) if job.track == "lora" else None,
              "head_lr": C.HEAD_LR, "epochs": epochs_for(job.kind), "batch": C.BATCH, "warmup_frac": C.WARMUP_FRAC,
              "A": A, "NLL": NLL, "probe": P if probes else None, "probe_train": P_train if probes else None,
              "metrics": metrics(A) if status == "completed" else None,
              "data": {k: data[k].provenance for k in order},
              "trainable_params": n_trainable, "peak_vram_gb": peak,
              "spectral_summary": [{"boundary": s["boundary"], "task": s["task"], "summary": s["summary"]}
                                   for s in spectral],
              "timing_s": {k: round(v, 2) for k, v in timing.items()}, "wall_s": round(time.time() - t0, 2),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "torch": torch.__version__}
    with gzip.open(out_dir / "train_log.jsonl.gz", "wt") as fh:
        for r in log_rows:
            fh.write(json.dumps({"task_pos": r[0], "epoch": r[1], "step": r[2], "loss": r[3],
                                 "grad_norm": r[4], "lr": r[5]}) + "\n")
    with gzip.open(out_dir / "spectral.json.gz", "wt") as fh:
        json.dump(spectral, fh)
    tmp = out_dir / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=1, allow_nan=True))
    os.replace(tmp, out_dir / "result.json")
    return result


def _optimizer(groups, track):
    if track == "lora":
        return torch.optim.AdamW(groups, betas=C.BETAS, eps=C.ADAM_EPS, weight_decay=C.WEIGHT_DECAY)
    import bitsandbytes as bnb
    mgr = bnb.optim.GlobalOptimManager.get_instance()
    mgr.initialize()
    small = [p for g in groups for p in g["params"] if p.ndim < 2] + list(groups[1]["params"])
    if small:
        mgr.override_config(small, "optim_bits", 32)
    mgr.register_parameters(groups)
    return bnb.optim.AdamW8bit(groups, betas=C.BETAS, eps=C.ADAM_EPS, weight_decay=C.WEIGHT_DECAY,
                               min_8bit_size=0, block_wise=True, is_paged=False)


def metrics(A) -> dict:
    T = len(A)
    final = A[-1]
    diag = [A[i][i] for i in range(T)]
    bwt_terms = [A[T - 1][j] - A[j][j] for j in range(T - 1)]
    # Chaudhry et al. (2018): best accuracy before the final boundary minus final accuracy.
    max_forget = [max(A[i][j] for i in range(j, T - 1)) - A[T - 1][j] for j in range(T - 1)]
    return {"acc_final": sum(final) / T, "plasticity": sum(diag) / T, "bwt": sum(bwt_terms) / (T - 1),
            "bwt_terms": bwt_terms, "forgetting_max": sum(max_forget) / (T - 1),
            "avg_incremental_acc": sum(sum(A[i][:i + 1]) / (i + 1) for i in range(T)) / T,
            "diag": diag, "final": final}
