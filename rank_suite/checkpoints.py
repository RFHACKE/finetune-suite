"""Lossless, CPU-only LoRA storage and verified legacy checkpoint compaction.

Original result/protocol/checksum records are never rewritten by maintenance.
An adjacent manifest binds retained PEFT exports to the original snapshot.
Imports of torch/safetensors are lazy so metadata observers need no ML stack.
"""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

from . import utils
from .protocol import digest, file_lock

STORAGE_NAME = "projection-storage.json"
FORMAT = "peft-lora-factors-v1"
SOURCE_NAMES = ("adapter_model.safetensors", "adapter_config.json")


def sha256(path):
    path = Path(path)
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f"File changed during verification: {path}")
    return h.hexdigest()


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _checkpoint(run_dir, boundary):
    run_dir = Path(run_dir).resolve()
    path = (run_dir / boundary["path"]).resolve()
    if run_dir not in path.parents:
        raise ValueError("Checkpoint path escapes its run directory")
    return path


def projection_reference(boundary):
    """Keep historical spectral cache identities stable after compaction."""
    value = boundary.get("projections_sha256") or boundary.get("adapter_factors_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("Missing projection/factor provenance checksum")
    return value


def _sources(checkpoint):
    return {name: {"bytes": (checkpoint / name).stat().st_size,
                   "sha256": sha256(checkpoint / name)} for name in SOURCE_NAMES}


def adapter_payload(checkpoint, identity):
    """Decode this suite's plain FP32 q/v LoRA exports without a base model."""
    from safetensors import safe_open
    import torch

    checkpoint = Path(checkpoint)
    cfg = json.loads((checkpoint / "adapter_config.json").read_text())
    if (cfg.get("peft_type") != "LORA" or cfg.get("bias", "none") != "none"
            or cfg.get("fan_in_fan_out", False)
            or any(cfg.get(key) for key in ("use_rslora", "use_dora", "use_qalora", "lora_bias",
                                           "rank_pattern", "alpha_pattern", "modules_to_save",
                                           "target_parameters", "trainable_token_indices"))):
        raise ValueError("Unsupported adapter variant; keep its original projections")
    rank = cfg["r"]
    if not isinstance(rank, int) or rank <= 0 or rank != identity.get("rank"):
        raise ValueError("Adapter rank disagrees with run identity")
    if set(cfg.get("target_modules", [])) != {"q_proj", "v_proj"}:
        raise ValueError("Expected the suite's q/v targets")
    if cfg.get("revision") is not None and cfg["revision"] != identity.get("revision"):
        raise ValueError("Adapter base revision disagrees with run identity")
    scale = float(cfg["lora_alpha"]) / rank
    if not math.isfinite(scale) or scale <= 0 or scale != identity.get("alpha_factor", 2):
        raise ValueError("Adapter scaling disagrees with run identity")
    matrices = {}
    prefix = "base_model.model."
    with safe_open(checkpoint / "adapter_model.safetensors", framework="pt", device="cpu") as source:
        for key in source.keys():
            if not key.startswith(prefix):
                raise ValueError(f"Unexpected adapter key: {key}")
            name, factor = None, None
            for letter in ("A", "B"):
                suffix = f".lora_{letter}.weight"
                if key.endswith(suffix):
                    name, factor = key[len(prefix):-len(suffix)], letter
                    break
            if name is None or name.rsplit(".", 1)[-1] not in ("q_proj", "v_proj"):
                raise ValueError(f"Unexpected adapter key: {key}")
            tensor = source.get_tensor(key)
            if tensor.dtype != torch.float32 or tensor.ndim != 2:
                raise ValueError(f"Expected a two-dimensional FP32 factor: {key}")
            matrices.setdefault(name, {"scale": scale})[factor] = tensor
    if not matrices:
        raise ValueError("Empty adapter export")
    for name, value in matrices.items():
        if set(value) != {"A", "B", "scale"} or value["A"].shape[0] != rank or value["B"].shape[1] != rank:
            raise ValueError(f"Missing or wrong-shaped factor pair: {name}")
    return {"track": "lora", "identity": identity, "matrices": matrices}


def factor_digest(payload):
    """Bit-exact content identity, independent of PyTorch serialization bytes."""
    h = hashlib.sha256()
    h.update(digest({"track": payload["track"], "identity": payload["identity"]}).encode())
    for name, value in sorted(payload["matrices"].items()):
        h.update(digest({"name": name, "scale": value["scale"]}).encode())
        for letter in ("A", "B"):
            tensor = value[letter].detach().cpu().contiguous()
            h.update(digest({"factor": letter, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}).encode())
            h.update(memoryview(tensor.numpy()).cast("B"))
    return h.hexdigest()


def _manifest(checkpoint, result, boundary, verify_hashes=False):
    manifest = utils.load_json(checkpoint / STORAGE_NAME)
    if (not manifest or manifest.get("format") != FORMAT or manifest.get("version") != 1
            or manifest.get("identity_sha256") != digest(result["run_identity"])
            or manifest.get("original_projection_sha256") != boundary.get("projections_sha256")
            or (boundary.get("adapter_factors_sha256") is not None
                and manifest.get("factors_sha256") != boundary["adapter_factors_sha256"])):
        raise ValueError(f"Missing or mismatched adapter-storage manifest: {checkpoint}")
    if bool(boundary.get("projections_sha256")) != bool(manifest.get("original_projection_bytes")):
        raise ValueError("Invalid original projection provenance")
    if (not boundary.get("projections_sha256")
            and boundary.get("projection_storage") != FORMAT):
        raise ValueError("Run does not declare adapter-backed storage")
    if set(manifest.get("sources", {})) != set(SOURCE_NAMES):
        raise ValueError("Missing adapter/config source checksums")
    for name, record in manifest["sources"].items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Missing/wrong-size retained source: {path}")
        if verify_hashes and sha256(path) != record["sha256"]:
            raise ValueError(f"Retained source checksum mismatch: {path}")
    return manifest


def _original_payload(checkpoint, result, boundary, verified_sha=None):
    import torch

    path = checkpoint / "projections.pt"
    current_sha = verified_sha or sha256(path)
    if current_sha != boundary.get("projections_sha256"):
        manifest = _manifest(checkpoint, result, boundary, verify_hashes=True)
        if current_sha != manifest.get("materialized_sha256"):
            raise ValueError(f"Projection checksum mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if digest(payload.get("identity")) != digest(result["run_identity"]) or payload.get("track") != result["track"]:
        raise ValueError("Snapshot belongs to a different run")
    return payload


def write_adapter_manifest(checkpoint, identity, original_sha=None):
    """Native adapter-only export; called after save_pretrained finishes."""
    checkpoint = Path(checkpoint)
    payload = adapter_payload(checkpoint, identity)
    manifest = {"version": 1, "format": FORMAT, "created_utc": _utc(),
                "verification": "native_adapter_export", "identity_sha256": digest(identity),
                "factors_sha256": factor_digest(payload), "n_matrices": len(payload["matrices"]),
                "sources": _sources(checkpoint), "original_projection_sha256": original_sha,
                "original_projection_bytes": ((checkpoint / "projections.pt").stat().st_size
                                              if original_sha else None)}
    utils.save_json(checkpoint / STORAGE_NAME, manifest)
    return manifest


def verify_boundary(run_dir, result, boundary):
    """Verify all factor bytes before attesting a historical replacement."""
    checkpoint = _checkpoint(run_dir, boundary)
    projection = checkpoint / "projections.pt"
    if not projection.exists():
        payload = load_boundary_projections(run_dir, result, boundary)
        manifest = _manifest(checkpoint, result, boundary)
        return {"path": str(checkpoint), "factors_sha256": factor_digest(payload),
                "n_matrices": manifest["n_matrices"], "projection_bytes": 0,
                "already_adapter_backed": True}
    current_sha = sha256(projection)
    original = _original_payload(checkpoint, result, boundary, verified_sha=current_sha)
    if result["track"] != "lora":
        raise ValueError("Only LoRA projections can be compacted")
    sources = _sources(checkpoint)
    payload = adapter_payload(checkpoint, result["run_identity"])
    expected, actual = factor_digest(original), factor_digest(payload)
    if expected != actual:
        raise ValueError(f"LoRA factors/keys/shapes/scaling differ: {checkpoint}")
    # Existing manifests keep their original provenance and reconstruction hash.
    prior = utils.load_json(checkpoint / STORAGE_NAME, {})
    if prior:
        _manifest(checkpoint, result, boundary, verify_hashes=True)
        if prior["factors_sha256"] != actual:
            raise ValueError("Factor checksum disagrees with existing storage manifest")
    manifest = {**prior, "version": 1, "format": FORMAT, "verified_utc": _utc(),
                "verification": "all_factor_bytes_keys_shapes_scaling_and_identity_equal",
                "identity_sha256": digest(result["run_identity"]), "factors_sha256": actual,
                "n_matrices": len(payload["matrices"]), "sources": sources,
                "original_projection_sha256": boundary.get("projections_sha256"),
                "original_projection_bytes": (prior.get("original_projection_bytes") or
                                              projection.stat().st_size)
                if boundary.get("projections_sha256") else None}
    utils.save_json(checkpoint / STORAGE_NAME, manifest)
    return {"path": str(checkpoint), "factors_sha256": actual,
            "n_matrices": len(payload["matrices"]), "projection_bytes": projection.stat().st_size,
            "projection_sha256": current_sha, "projection_mtime_ns": projection.stat().st_mtime_ns,
            "already_adapter_backed": False}


def load_boundary_projections(run_dir, result, boundary):
    """Read original exports or reconstruct in memory; never write a duplicate."""
    checkpoint = _checkpoint(run_dir, boundary)
    if (checkpoint / "projections.pt").exists():
        return _original_payload(checkpoint, result, boundary)
    if result["track"] != "lora":
        raise ValueError(f"Missing full-FT projections: {checkpoint}")
    manifest = _manifest(checkpoint, result, boundary, verify_hashes=True)
    payload = adapter_payload(checkpoint, result["run_identity"])
    if factor_digest(payload) != manifest["factors_sha256"] or len(payload["matrices"]) != manifest["n_matrices"]:
        raise ValueError(f"Reconstructed factors disagree with verified snapshot: {checkpoint}")
    return payload


def validate_boundary_storage(run_dir, result, boundary, verify_hashes=False):
    """Small-metadata validation, optionally hashing the retained files."""
    checkpoint = _checkpoint(run_dir, boundary)
    projection = checkpoint / "projections.pt"
    if projection.exists() and boundary.get("projections_sha256"):
        if not verify_hashes or sha256(projection) == boundary["projections_sha256"]:
            return {"format": "original_projections", "bytes": projection.stat().st_size}
        # A deliberately materialized equivalent file has a separate checksum.
    if result["track"] != "lora":
        raise ValueError(f"Missing or corrupt full-FT projections: {checkpoint}")
    manifest = _manifest(checkpoint, result, boundary, verify_hashes)
    if projection.exists():
        if (not manifest.get("materialized_sha256")
                or projection.stat().st_size != manifest.get("materialized_bytes")):
            raise ValueError(f"Unrecorded/wrong-size materialized projection: {projection}")
        if verify_hashes and sha256(projection) != manifest["materialized_sha256"]:
            raise ValueError(f"Projection checksum mismatch: {projection}")
    return {"format": FORMAT, "bytes": sum(r["bytes"] for r in manifest["sources"].values()),
            "original_projection_sha256": manifest["original_projection_sha256"],
            "factors_sha256": manifest["factors_sha256"]}


def checkpoint_inventory_errors(run_dir, result, verify_hashes=False):
    """Accept recorded compaction without altering historical file inventories."""
    run_dir = Path(run_dir)
    errors = []
    boundaries = {str(Path(b["path"]) / "projections.pt"): b for b in result.get("boundaries", [])}
    for name, size in result.get("checkpoint_sizes", {}).items():
        path = run_dir / name
        if name in boundaries and result.get("track") == "lora":
            if path.is_file() and path.stat().st_size == size:
                continue
            try:
                manifest = _manifest(_checkpoint(run_dir, boundaries[name]), result, boundaries[name], verify_hashes)
                if path.exists() and (path.stat().st_size != manifest.get("materialized_bytes")
                                      or not manifest.get("materialized_sha256")):
                    raise ValueError("Unexpected/wrong-size projection file")
            except (ValueError, OSError, KeyError) as exc:
                errors.append(f"Missing/wrong-size export: {name}: {exc}")
        elif not path.is_file() or path.stat().st_size != size:
            errors.append("Missing/wrong-size export: " + name)
    for boundary in result.get("boundaries", []):
        try:
            validate_boundary_storage(run_dir, result, boundary, verify_hashes)
        except (ValueError, OSError, KeyError) as exc:
            errors.append(str(exc))
    return errors


def compacted_projection_record(path):
    """Let a resumed downloader recognize an intentionally absent duplicate.

    Invalid retained sources raise an error rather than silently fetching over
    an intentional compaction. No cloud access or tensor library is needed.
    """
    path = Path(path)
    if path.name != "projections.pt" or path.exists() or not (path.parent / STORAGE_NAME).is_file():
        return None
    run_dir = path.parent.parent.parent
    result = utils.load_json(run_dir / "result.json")
    if not result:
        raise ValueError("Compacted checkpoint has no original run metadata")
    for boundary in result["boundaries"]:
        if _checkpoint(run_dir, boundary) == path.parent.resolve():
            manifest = _manifest(path.parent, result, boundary, verify_hashes=True)
            if manifest["original_projection_sha256"]:
                return {"bytes": manifest["original_projection_bytes"],
                        "sha256": manifest["original_projection_sha256"], "retained_as": FORMAT}
    raise ValueError("Compacted checkpoint is absent from the original run metadata")


def restore_boundary(run_dir, result, boundary):
    """Materialize a compatible .pt; original checksum evidence stays untouched."""
    import torch

    checkpoint = _checkpoint(run_dir, boundary)
    path = checkpoint / "projections.pt"
    if path.exists():
        verify_boundary(run_dir, result, boundary)
        return False
    payload = load_boundary_projections(run_dir, result, boundary)
    manifest = _manifest(checkpoint, result, boundary, verify_hashes=True)
    fd, temporary = tempfile.mkstemp(dir=checkpoint, suffix=".pt.tmp")
    try:
        with os.fdopen(fd, "wb") as target:
            torch.save(payload, target)
            target.flush()
            os.fsync(target.fileno())
        roundtrip = torch.load(temporary, map_location="cpu", weights_only=True, mmap=True)
        if factor_digest(roundtrip) != manifest["factors_sha256"]:
            raise ValueError("Restored snapshot failed its bit-exact round-trip check")
        del roundtrip
        # Save the separate materialization checksum before exposing the new file.
        manifest.update(materialized_sha256=sha256(temporary), materialized_bytes=Path(temporary).stat().st_size,
                        materialized_utc=_utc())
        utils.save_json(checkpoint / STORAGE_NAME, manifest)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return True


def maintain_checkpoints(roots, action, run_ids=None, tasks=None):
    """Verify the entire selected set before any deletion; journal every action."""
    if action not in ("verify", "prune", "restore"):
        raise ValueError("Choose verify, prune, or restore")
    roots = [Path(root).resolve() for root in roots]
    if len(set(roots)) != len(roots):
        raise ValueError("Duplicate output directory")
    selected, seen_ids = [], set()
    for root in roots:
        if not (root / "runs").is_dir():
            raise ValueError(f"Missing runs directory: {root}")
        for path in sorted((root / "runs").glob("*/result.json")):
            result = utils.load_json(path)
            if (result.get("status") != "completed" or result.get("track") != "lora"
                    or result.get("mode") not in ("main", "smoke")):
                continue
            if run_ids and result["run_id"] not in run_ids:
                continue
            seen_ids.add(result["run_id"])
            for boundary in result["boundaries"]:
                if not tasks or boundary["task"] in tasks:
                    selected.append((root, path.parent, result, boundary))
    if run_ids and set(run_ids) != seen_ids:
        raise ValueError("Unknown or ineligible run IDs: " + ", ".join(sorted(set(run_ids) - seen_ids)))
    if not selected:
        raise ValueError("No completed LoRA task boundaries selected")
    begin = time.monotonic()
    operation = f"{time.time_ns()}-{action}"
    journals = {root: root / "execution" / "checkpoint-storage" / f"{operation}.jsonl" for root in roots}
    reports = {root: {"operation": operation, "action": action, "started_utc": _utc(),
                      "status": "verifying", "verified_boundaries": 0, "removed_files": 0,
                      "reclaimed_bytes": 0, "restored_files": 0,
                      "original_results_modified": False} for root in roots}

    def record(root, event):
        event = {"utc": _utc(), **event}
        journal = journals[root]
        journal.parent.mkdir(parents=True, exist_ok=True)
        with journal.open("a") as target:
            target.write(json.dumps(event, sort_keys=True) + "\n")
            target.flush()
            os.fsync(target.fileno())
        utils.save_json(journal.with_suffix(".json"), reports[root])

    # Run locks also exclude training and intruder analysis. Hold them throughout
    # the two-phase operation so no selected checkpoint can change underneath it.
    with ExitStack() as stack:
        for run_dir in sorted({item[1] for item in selected}):
            if not stack.enter_context(file_lock(run_dir / "run.lock", blocking=False)):
                raise ValueError(f"Run is busy: {run_dir}")
        verified = []
        try:
            for i, (root, run_dir, result, boundary) in enumerate(selected, 1):
                checkpoint = _checkpoint(run_dir, boundary)
                if not (checkpoint / "heads.pt").is_file():
                    raise ValueError(f"Missing classification heads: {checkpoint}")
                item = verify_boundary(run_dir, result, boundary)
                reports[root]["verified_boundaries"] += 1
                verified.append((root, run_dir, result, boundary, item))
                record(root, {"event": "verified", "run_id": result["run_id"], "task": boundary["task"], **item})
                if i == 1 or i % 10 == 0 or i == len(selected):
                    print(f"{_utc()} verified {i}/{len(selected)} task boundaries", flush=True)
            # Each replacement is now proven against its original. For pruning,
            # recheck every retained source before the first destructive action.
            if action == "prune":
                for root, run_dir, result, boundary, item in verified:
                    _manifest(_checkpoint(run_dir, boundary), result, boundary, verify_hashes=True)
                for root, run_dir, result, boundary, item in verified:
                    path = _checkpoint(run_dir, boundary) / "projections.pt"
                    if not path.exists():
                        continue
                    if path.stat().st_size != item["projection_bytes"] or path.stat().st_mtime_ns != item["projection_mtime_ns"]:
                        raise ValueError(f"Projection changed after verification: {path}")
                    record(root, {"event": "delete_intent", "path": str(path),
                                  "sha256": item["projection_sha256"], "bytes": item["projection_bytes"]})
                    path.unlink()
                    reports[root]["removed_files"] += 1
                    reports[root]["reclaimed_bytes"] += item["projection_bytes"]
                    record(root, {"event": "deleted", "path": str(path), "bytes": item["projection_bytes"]})
            elif action == "restore":
                for root, run_dir, result, boundary, item in verified:
                    if restore_boundary(run_dir, result, boundary):
                        reports[root]["restored_files"] += 1
                    record(root, {"event": "restored_or_already_present", "path": item["path"]})
        except BaseException as exc:
            for root in roots:
                reports[root].update(status="failed", error=str(exc), finished_utc=_utc(),
                                     failed_run_id=result["run_id"], failed_task=boundary["task"])
                record(root, {"event": "failed", "error": str(exc),
                              "run_id": result["run_id"], "task": boundary["task"]})
            raise
        for root in roots:
            reports[root].update(status="complete", finished_utc=_utc(), elapsed_s=time.monotonic() - begin)
            record(root, {"event": "complete", **reports[root]})
    return {"status": "complete", "action": action, "verified_boundaries": len(selected),
            "removed_files": sum(r["removed_files"] for r in reports.values()),
            "reclaimed_bytes": sum(r["reclaimed_bytes"] for r in reports.values()),
            "reports": [str(p.with_suffix(".json")) for p in journals.values()]}
