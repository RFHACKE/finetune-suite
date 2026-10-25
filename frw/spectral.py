"""Update/subspace diagnostics computed on the GPU at every task boundary.

For each adapted matrix W0 (q_proj / v_proj) with LoRA update dW = s * B A:
  * ||dW||_F, ||dW||_F / ||W0||_F, spectral norm, stable rank, entropy effective
    rank and 90%-energy rank, from the exact singular values of dW obtained via
    two thin QR factorisations (an r x r SVD, invariant to A/B rescaling);
  * intruder dimensions (Shuttleworth et al. 2025, Alg. 1): a top-k left
    singular vector of W0 + dW is an intruder if its maximum |cosine| with all
    left singular vectors of W0 is below epsilon; counted at several epsilons;
  * leading-subspace distance D_k = 1 - ||U0_k^T Ut_k||_F^2 / k.
The top-k SVD of W0 + dW uses torch.svd_lowrank (oversampled, 6 power
iterations); W0's full left basis is computed once per container and cached.
"""
from __future__ import annotations

import math

import torch

from . import config as C


def _sv_from_factors(B, A, scale):
    # float32 on GPU (L40S float64 throughput is ~1/64 of float32; r=1024 cost 40 s/boundary in
    # float64), float64 on CPU. Relative error of the singular values is ~1e-6 either way.
    dt = torch.float32 if B.is_cuda else torch.float64
    qb, rb = torch.linalg.qr(B.to(dt), mode="reduced")
    qa, ra = torch.linalg.qr(A.to(dt).T, mode="reduced")
    core = scale * (rb @ ra.T)
    return torch.linalg.svdvals(core).double()


def _summaries(sv):
    sv = sv.clamp_min(0)
    total2 = float((sv ** 2).sum())
    if total2 <= 0:
        return {"fro": 0.0, "spec": 0.0, "stable_rank": 0.0, "entropy_rank": 0.0, "energy90_rank": 0}
    p = sv / sv.sum()
    p = p[p > 0]
    ent = float(torch.exp(-(p * p.log()).sum()))
    cum = torch.cumsum(sv ** 2, 0) / total2
    return {"fro": math.sqrt(total2), "spec": float(sv[0]), "stable_rank": total2 / float(sv[0]) ** 2,
            "entropy_rank": ent, "energy90_rank": int((cum < 0.9).sum()) + 1}


class Reference:
    """Full left singular bases of the pretrained q/v matrices (kept on device)."""

    def __init__(self):
        self.u = {}
        self.norm = {}
        self.topu = {}

    def add(self, name, W0):
        W = W0.detach().float()
        U, S, _ = torch.linalg.svd(W, full_matrices=False)
        self.u[name] = U.contiguous()
        self.norm[name] = float(torch.linalg.matrix_norm(W))
        self.topu[name] = U[:, :C.INTRUDER_K].contiguous()


@torch.no_grad()
def boundary_metrics(model, ref: Reference, seed: int):
    from .model import lora_delta, lora_modules
    rows = []
    gen_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)  # svd_lowrank randomness must not perturb training RNG
    try:
        for name, mod in lora_modules(model):
            B, A, scale = lora_delta(mod)
            sv = _sv_from_factors(B, A, scale)
            row = {"name": name, "type": name.split(".")[-1], "layer": _layer(name)}
            row.update(_summaries(sv))
            key = name.split("base_model.model.")[-1]
            row["rel_fro"] = row["fro"] / ref.norm[key] if ref.norm.get(key) else None
            W0 = mod.base_layer.weight.float()
            Wt = W0 + scale * (B.float() @ A.float())
            k = C.INTRUDER_K
            U, S, _ = torch.svd_lowrank(Wt, q=min(k + 20, min(Wt.shape)), niter=6)
            Ut = U[:, :k]
            cos = (ref.u[key].T @ Ut).abs().max(dim=0).values      # max |cos| per tuned vector
            row["intruders"] = {str(e): int((cos < e).sum()) for e in C.INTRUDER_EPS}
            row["max_cos_min"] = float(cos.min())
            overlap = torch.linalg.matrix_norm(ref.topu[key].T @ Ut) ** 2 / k
            row["subspace_dist_k10"] = float(1 - overlap)
            rows.append(row)
    finally:
        torch.random.set_rng_state(gen_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    return rows


@torch.no_grad()
def full_ft_metrics(model, initial: dict, ref: Reference):
    """Update norms/intruders for full FT on q/v (dense; the same definitions)."""
    rows = []
    for name, p in model.named_parameters():
        parts = name.split(".")
        if len(parts) < 2 or parts[-2] not in C.LORA_TARGETS or parts[-1] != "weight":
            continue
        key = name[: -len(".weight")]
        W0 = initial[key].to(p.device).float()
        dW = p.detach().float() - W0
        sv = torch.linalg.svdvals(dW)
        row = {"name": key, "type": parts[-2], "layer": _layer(key)}
        row.update(_summaries(sv))
        row["rel_fro"] = row["fro"] / ref.norm[key]
        U, S, _ = torch.svd_lowrank(p.detach().float(), q=C.INTRUDER_K + 20, niter=6)
        Ut = U[:, :C.INTRUDER_K]
        cos = (ref.u[key].T @ Ut).abs().max(dim=0).values
        row["intruders"] = {str(e): int((cos < e).sum()) for e in C.INTRUDER_EPS}
        row["max_cos_min"] = float(cos.min())
        row["subspace_dist_k10"] = float(1 - torch.linalg.matrix_norm(ref.topu[key].T @ Ut) ** 2 / C.INTRUDER_K)
        rows.append(row)
    return rows


def _layer(name: str):
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return None


def aggregate(rows) -> dict:
    """Run-level summary (mean over matrices within a run; never pooled as replicates)."""
    if not rows:
        return {}
    out = {}
    for t in ("q_proj", "v_proj", "all"):
        sel = [r for r in rows if t == "all" or r["type"] == t]
        if not sel:
            continue
        mean = lambda k: float(sum(r[k] for r in sel if r[k] is not None) / len(sel))  # noqa: E731
        out[t] = {"rel_fro_mean": mean("rel_fro"), "fro_mean": mean("fro"),
                  "stable_rank_mean": mean("stable_rank"), "entropy_rank_mean": mean("entropy_rank"),
                  "energy90_rank_mean": mean("energy90_rank"), "subspace_dist_k10_mean": mean("subspace_dist_k10"),
                  "intruders_total": {e: int(sum(r["intruders"][e] for r in sel)) for e in sel[0]["intruders"]},
                  "intruder_fraction": {e: float(sum(r["intruders"][e] for r in sel)) / (C.INTRUDER_K * len(sel))
                                        for e in sel[0]["intruders"]},
                  "n_matrices": len(sel)}
    return out
