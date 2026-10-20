"""Deferred CPU spectral analysis of exact task-boundary q/v weights.

Compare top-k tuned LEFT vectors with ALL pretrained left vectors using
absolute cosine. Counts for full FT are observations, never assertions.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import time

import numpy as np
import torch

from . import config, utils
from .checkpoints import load_boundary_projections, projection_reference
from .model_utils import adapted_linear_modules
from .protocol import digest, file_lock

logger = logging.getLogger("rank_suite.intruder")
EPS_VALUES = (0.6, 0.7, 0.8, 0.9)
ANALYSIS_VERSION = "all-left-v3-update-subspace"


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_torch(path, value):
    """Atomic replacement with unique temporary names for workers."""
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(value, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def projection_path(root, model_key, revision):
    return Path(root) / "cache" / "projections" / model_key / revision / "base.pt"


def export_base_projections(model, root, mspec):
    path = projection_path(root, mspec.key, mspec.revision)
    with file_lock(path.with_suffix(".lock")):
        meta = utils.load_json(path.with_suffix(".json"), {})
        valid = (path.exists() and meta.get("revision") == mspec.revision
                 and meta.get("sha256") == file_sha256(path))
        if not valid:
            weights = {name: module.weight.detach().cpu().clone()
                       for name, module in adapted_linear_modules(model)}
            if len(weights) != 2 * mspec.layers:
                raise ValueError("Missing pretrained q/v projections")
            save_torch(path, {"revision": mspec.revision, "weights": weights})
            utils.save_json(path.with_suffix(".json"), {
                "revision": mspec.revision, "sha256": file_sha256(path),
                "n_matrices": len(weights), "storage": "original_bfloat16",
            })
    return path


def export_boundary_projections(model, track, path, identity):
    matrices = {}
    for name, module in adapted_linear_modules(model):
        if track == "lora":
            matrices[name] = {"A": module.lora_A["default"].weight.detach().float().cpu().clone(),
                              "B": module.lora_B["default"].weight.detach().float().cpu().clone(),
                              "scale": float(module.scaling["default"])}
        else:
            # Preserve fp32 changes; rounding snapshots to bf16 changes the metric.
            matrices[name] = module.weight.detach().float().cpu().clone()
    save_torch(path, {"track": track, "identity": identity, "matrices": matrices})
    return file_sha256(path)


def left_spectrum(weight, k=None, solver="eigh"):
    """Dense deterministic LAPACK decompositions, without randomized SVD.

    A full fp32 reference SVD is computed once. For tuned top-k vectors,
    float64 Gram + subset eigensolve avoids unused right singular vectors.
    Select svd for a direct implementation cross-check.
    """
    from scipy import linalg
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2 or not np.isfinite(w).all():
        raise ValueError("SVD input must be a finite matrix")
    if k is None or solver == "svd":
        u, s, _ = linalg.svd(w, full_matrices=False, check_finite=False, lapack_driver="gesdd")
        return (u, s) if k is None else (u[:, :min(k, len(s))], s[:min(k, len(s))])
    if solver != "eigh":
        raise ValueError("solver must be eigh or svd")
    w64 = w.astype(np.float64)
    n = w64.shape[0]
    k = min(k, min(w64.shape))
    values, u = linalg.eigh(w64 @ w64.T, subset_by_index=(n - k, n - 1),
                           driver="evr", check_finite=False)
    return np.ascontiguousarray(u[:, ::-1], dtype=np.float32), np.sqrt(np.maximum(values[::-1], 0))


def left_vectors(weight, k=None, solver="eigh"):
    return left_spectrum(weight, k, solver)[0]


def _counts_from_vectors(top, ref):
    ref = ref / np.maximum(np.linalg.norm(ref, axis=0, keepdims=True), 1e-30)
    top = top / np.maximum(np.linalg.norm(top, axis=0, keepdims=True), 1e-30)
    max_sim = np.clip(np.abs(top.T @ ref).max(axis=1), 0, 1)
    return {f"{e:.1f}": int((max_sim < e).sum()) for e in EPS_VALUES}, max_sim


def intruder_counts(W_ft, U0, k=config.INTRUDER_K, side="left", solver="eigh"):
    if side != "left":
        raise ValueError("Only left vectors implement the registered metric")
    w = W_ft.detach().float().cpu().numpy() if torch.is_tensor(W_ft) else W_ft
    ref = U0.detach().float().cpu().numpy() if torch.is_tensor(U0) else np.asarray(U0)
    top = left_vectors(w, k, solver)
    return _counts_from_vectors(top, ref)


def _cached_reference(base_weight, path):
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            return saved["vectors"], saved["singular_values"]
    vectors, singular_values = left_spectrum(base_weight.float().numpy())
    import os
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
        np.savez(f, vectors=vectors, singular_values=singular_values)
        tmp = f.name
    os.replace(tmp, path)
    return vectors, singular_values


def update_diagnostics(original, value, track):
    """Gauge-invariant merged-update metrics; LoRA needs only an r×r SVD."""
    from scipy import linalg
    base = (original.detach().float().cpu().numpy().astype(np.float64)
            if torch.is_tensor(original) else np.asarray(original, dtype=np.float64))
    if track == "lora":
        a, b = np.asarray(value["A"], dtype=np.float64), np.asarray(value["B"], dtype=np.float64)
        _, rb = linalg.qr(b, mode="economic", check_finite=False)
        _, ra = linalg.qr(a.T, mode="economic", check_finite=False)
        singular = linalg.svdvals(float(value["scale"]) * rb @ ra.T, check_finite=False)
        denominator = a.shape[0]
    else:
        delta = np.asarray(value, dtype=np.float64) - base
        singular = linalg.svdvals(delta, check_finite=False)
        denominator = min(base.shape)
    if not np.isfinite(singular).all():
        raise ValueError("Nonfinite update spectrum")
    tolerance = max(base.shape) * np.finfo(np.float64).eps * (singular[0] if len(singular) else 0)
    singular = singular[singular > tolerance]
    energy = float(np.sum(singular ** 2))
    zero = not len(singular)
    frob = float(np.sqrt(energy))
    norm0 = float(linalg.norm(base, "fro"))
    spectral = float(singular[0]) if not zero else 0.0
    p = singular / singular.sum() if not zero else singular
    erank = float(np.exp(-np.sum(p * np.log(p)))) if not zero else 0.0
    return {"zero_update": zero, "frobenius_norm": frob,
            "relative_frobenius": frob / norm0 if norm0 else None,
            "spectral_norm": spectral, "stable_rank": energy / spectral ** 2 if spectral else 0.0,
            "effective_rank": erank, "effective_rank_fraction": erank / denominator,
            "rank_denominator": denominator, "numerical_rank": int(len(singular)),
            "energy_rank_90": int(np.searchsorted(np.cumsum(singular ** 2), .9 * energy) + 1) if not zero else 0,
            "singular_value_tolerance": float(tolerance)}


def subspace_diagnostics(reference, ref_s, tuned, tuned_s, requested=(10, 20)):
    """Leading-subspace distance is invariant to rotations inside the span."""
    from scipy import linalg
    output = {}
    for target in requested:
        k = min(target, reference.shape[1], tuned.shape[1])
        cosines = np.clip(linalg.svdvals(reference[:, :k].astype(np.float64).T
                                        @ tuned[:, :k].astype(np.float64)), 0, 1)
        distance = float(np.clip(1 - np.sum(cosines ** 2) / k, 0, 1))
        def gap(s):
            if k >= len(s):
                return {"absolute": None, "relative": None, "near_tie": None}
            absolute = float(max(s[k - 1] - s[k], 0))
            relative = absolute / max(float(s[k - 1]), 1e-30)
            return {"absolute": absolute, "relative": relative, "near_tie": relative < 1e-3}
        output[str(target)] = {"k_actual": k, "distance": distance,
            "principal_angles_radians": np.arccos(cosines).tolist(),
            "reference_cutoff_gap": gap(ref_s), "tuned_cutoff_gap": gap(tuned_s)}
    return output


def analyze_intruders(root, models=None, threads=4, solver="eigh"):
    """Resume at matrix granularity, using only CPU and saved tensors."""
    from threadpoolctl import threadpool_limits
    import scipy
    from .metrics import load_results
    root = Path(root)
    if threads <= 0:
        raise ValueError("threads must be positive")
    results = load_results(root, modes=("main", "smoke", "sanity"), validate_spectra=False)
    numerical_stack = {"numpy": np.__version__, "scipy": scipy.__version__, "torch": str(torch.__version__)}
    done = skipped = 0
    torch.set_num_threads(threads)
    with threadpool_limits(limits=threads):
        for result in results:
            if models and result["model"] not in models:
                continue
            run_dir = Path(result["_dir"])
            with file_lock(run_dir / "run.lock", blocking=False) as acquired:
                if not acquired:
                    skipped += 1
                    continue
                rev = result["hf_revision"]
                base_path = projection_path(root, result["model"], rev)
                base_meta = utils.load_json(base_path.with_suffix(".json"))
                if not base_meta or file_sha256(base_path) != base_meta["sha256"]:
                    raise ValueError(f"Missing or corrupt pretrained projections: {base_path}")
                base = torch.load(base_path, map_location="cpu", weights_only=True)
                if base["revision"] != rev:
                    raise ValueError("Pretrained projection revision mismatch")
                payloads = {}
                for boundary in result["boundaries"]:
                    task = boundary["task"]
                    ckpt = run_dir / boundary["path"]
                    tuned = load_boundary_projections(run_dir, result, boundary)
                    if tuned["identity"] != result["run_identity"]:
                        raise ValueError("Snapshot belongs to a different run")
                    analysis_id = digest({"version": ANALYSIS_VERSION, "solver": solver,
                                          "numerical_stack": numerical_stack,
                                          "base": base_meta["sha256"],
                                          "boundary": projection_reference(boundary)})
                    adir = ckpt / "spectra" / analysis_id
                    cache = utils.load_json(adir / "matrices.json", {})
                    start = time.monotonic()
                    if set(tuned["matrices"]) != set(base["weights"]):
                        raise ValueError("Boundary projections differ from the pretrained matrices")
                    for name, value in tuned["matrices"].items():
                        if name in cache:
                            continue
                        original = base["weights"][name]
                        ref_path = (root / "cache" / "spectra-v3" / base_meta["sha256"]
                                    / digest(numerical_stack) / f"{name}.npz")
                        with file_lock(ref_path.with_suffix(".lock")):
                            ref, ref_s = _cached_reference(original, ref_path)
                        w = (original.float() + value["scale"] * (value["B"] @ value["A"])
                             if result["track"] == "lora" else value)
                        top, singular = left_spectrum(w, k=21, solver=solver)
                        counts, sims = _counts_from_vectors(top[:, :config.INTRUDER_K], ref)
                        near = bool(any(np.any(np.abs(sims - e) < 1e-4) for e in EPS_VALUES))
                        if near and solver == "eigh":
                            top, singular = left_spectrum(w, k=21, solver="svd")
                            counts, sims = _counts_from_vectors(top[:, :config.INTRUDER_K], ref)
                        cache[name] = {"counts": counts, "max_cosines": sims.tolist(),
                                       "near_threshold": near,
                                       "direct_svd_check": near and solver == "eigh",
                                       "update": update_diagnostics(original, value, result["track"]),
                                       "subspace": subspace_diagnostics(ref, ref_s, top, singular)}
                        utils.save_json(adir / "matrices.json", cache)
                    total = {f"{e:.1f}": sum(x["counts"][f"{e:.1f}"] for x in cache.values())
                             for e in EPS_VALUES}
                    by_type = {kind: {f"{e:.1f}": sum(v["counts"][f"{e:.1f}"]
                                  for n, v in cache.items() if n.endswith(kind)) for e in EPS_VALUES}
                               for kind in config.INTRUDER_MATRICES}
                    by_layer = {}
                    for name, value in cache.items():
                        layer = name.rsplit(".", 1)[0]
                        counts = by_layer.setdefault(layer, {f"{e:.1f}": 0 for e in EPS_VALUES})
                        for eps, number in value["counts"].items():
                            counts[eps] += number
                    n_vectors = sum(len(value["max_cosines"]) for value in cache.values())
                    payloads[f"after_{task}"] = {
                        "task_boundary": f"after_{task}", "total": total,
                        "by_matrix_type": by_type,
                        "by_layer": by_layer,
                        "by_matrix": {n: v["counts"] for n, v in cache.items()},
                        "diagnostics_by_matrix": {n: {"update": v["update"], "subspace": v["subspace"]}
                                                  for n, v in cache.items()},
                        "mean_max_cosine": float(np.mean([s for v in cache.values()
                                                           for s in v["max_cosines"]])),
                        "intruder_fraction": {e: n / n_vectors
                                               for e, n in total.items()},
                        "n_vectors": n_vectors,
                        "n_matrices": len(cache), "k": config.INTRUDER_K,
                        "ref_mode": "all", "side": "left", "solver": solver,
                        "analysis_id": analysis_id,
                        "analysis_version": ANALYSIS_VERSION,
                        "numerical_stack": numerical_stack, "threads": threads,
                        "near_threshold_matrices": [n for n, v in cache.items() if v["near_threshold"]],
                        "elapsed_this_invocation_s": time.monotonic() - start,
                    }
                    logger.info("%s after %s: %s", result["run_id"], task, total)
                result["intruder"] = payloads
                result["intruder_status"] = "completed"
                result.pop("_dir", None)
                utils.save_json(run_dir / "result.json", result)
                done += 1
    return {"analyzed_runs": done, "locked_runs_skipped": skipped, "solver": solver}
