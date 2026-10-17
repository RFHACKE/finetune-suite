"""Fail before paid experiments if CUDA, bf16 kernels, or the optimizer do not work."""
from __future__ import annotations

import sys


def check_environment(require_gpu=True, optimizer="adamw8bit"):
    import torch
    errors = []
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        errors.append("Use Python 3.10–3.12 (setup.sh provisions 3.11 if needed)")
    for name in ("transformers", "peft", "datasets", "scipy", "matplotlib", "threadpoolctl"):
        try:
            __import__(name)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    available = torch.cuda.is_available()
    if require_gpu and not available:
        errors.append("Training requires an NVIDIA GPU; CPU is supported for preparation/analysis only")
    if available and require_gpu:
        if require_gpu and torch.cuda.device_count() != 1:
            errors.append("Expose exactly one GPU per worker with CUDA_VISIBLE_DEVICES=N")
        if torch.cuda.get_device_capability()[0] < 8:
            errors.append("Native bf16 requires Ampere or newer; T4 is not supported")
        try:
            x = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
            (x @ x.T).sum().item()
            if optimizer == "adamw8bit":
                import bitsandbytes as bnb
                p = torch.nn.Parameter(torch.ones(8192, device="cuda", dtype=torch.float32))
                opt = bnb.optim.AdamW8bit([p], lr=1e-3)
                p.grad = torch.full_like(p, 0.1)
                opt.step()
                if not torch.isfinite(p).all() or torch.equal(p, torch.ones_like(p)):
                    errors.append("8-bit AdamW kernel did not produce a finite parameter update")
                del p, opt
            del x
            torch.cuda.empty_cache()
        except Exception as exc:
            errors.append(f"CUDA/bf16/optimizer kernel test failed: {exc}")
    if errors:
        raise RuntimeError("Environment check failed:\n  " + "\n  ".join(errors))
    return {"cuda": available, "kernel_test": "passed" if available else "CPU analysis environment"}
