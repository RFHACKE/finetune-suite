"""Lossless, CPU-only packaging of completed runs for faster result transfer.

Only size/checksum-verified completed checkpoints are archived. Both original
dual exports and verified adapter-backed compacted checkpoints are supported.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import modal

from rank_suite.checkpoints import checkpoint_inventory_errors

RUN = os.environ.get("FYP_MODAL_RUN", "frw-20261018")
MODELS = ["llama3.2-1b", "qwen2.5-1.5b", "granite3.3-2b", "qwen2.5-3b"]
app = modal.App("fyp-artifact-packaging", include_source=False)
volume = modal.Volume.from_name("fyp-finetune-results", version=2)
vault = modal.Volume.from_name("fyp-finetune-archive", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11").apt_install("zstd")
         .env({"PYTHONPATH": "/app", "PYTHONUNBUFFERED": "1", "FYP_MODAL_RUN": RUN})
         .add_local_dir(Path(__file__).resolve().parents[1] / "rank_suite", "/app/rank_suite", copy=True,
                        ignore=["__pycache__", "*.pyc"])
         .add_local_file(__file__, "/app/modal_archive.py", copy=True))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def package_run(model_root, run_dir, archive_root, include_base=False):
    result = json.loads((run_dir / "result.json").read_text())
    if result.get("status") != "completed":
        return False
    if checkpoint_inventory_errors(run_dir, result, verify_hashes=True):
        return False  # A volume copy/commit may still be in progress.
    if include_base:
        base_files = list((model_root / "cache" / "projections").glob("*/*/base.pt"))
        if len(base_files) != 1:
            return False
        metadata = json.loads(base_files[0].with_suffix(".json").read_text())
        if sha256(base_files[0]) != metadata["sha256"]:
            return False
    archive_root.mkdir(parents=True, exist_ok=True)
    destination = archive_root / f"{run_dir.name}.tar.zst"
    temporary = destination.with_name(destination.name + ".tmp")
    begin = time.monotonic()
    members = ["runs/" + run_dir.name] + (["cache/projections"] if include_base else [])
    tar = subprocess.Popen(["tar", "--exclude=resume.pt", "--exclude=*.lock", "-C", str(model_root),
                            "-cf", "-", *members], stdout=subprocess.PIPE)
    try:
        compressed = subprocess.run(["zstd", "-T2", "-3", "--long=30", "--no-progress", "-q", "-f",
                                     "-o", str(temporary)], stdin=tar.stdout, check=False)
        tar.stdout.close()
        tar_status = tar.wait()
        if tar_status or compressed.returncode:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Archive subprocess failed: tar={tar_status}, zstd={compressed.returncode}")
    finally:
        if tar.poll() is None:
            tar.terminate()
            tar.wait()
    os.replace(temporary, destination)
    payload = {"run_id": run_dir.name, "model": result["model"],
               "protocol_sha256": result["protocol_sha256"], "archive": destination.name,
               "archive_bytes": destination.stat().st_size, "archive_sha256": sha256(destination),
               "checkpoint_bytes_before_compression": sum(
                   p.stat().st_size for p in (run_dir / "checkpoints").rglob("*")
                   if p.is_file() and not p.name.endswith(".lock")),
               "original_checkpoint_export_bytes": result["checkpoint_total_bytes"],
               "elapsed_s": time.monotonic() - begin,
               "lossless": True, "includes_base_projections": include_base,
               "omitted": ["obsolete resume.pt", "advisory lock files"]}
    save(archive_root / f"{run_dir.name}.json", payload)
    vault.commit()
    print(json.dumps(payload), flush=True)
    return True


@app.function(image=image, volumes={"/results": volume, "/vault": vault}, cpu=(0.125, 2),
              memory=(256, 6144), timeout=28800, single_use_containers=True, retries=0)
def archive_completed_runs():
    base = Path("/results") / RUN
    archive_base = Path("/vault") / RUN
    packaged, failures, based_models = set(), [], set()
    for path in (archive_base / "archives").glob("*/*.json"):
        packaged.add(path.stem)
        metadata = json.loads(path.read_text())
        if metadata.get("includes_base_projections"):
            based_models.add(metadata["model"])
    while True:
        volume.reload()  # Observe newly committed files from GPU containers.
        pending = []
        for model in MODELS:
            model_root = base / "main" / model
            for result_path in (model_root / "runs").glob("*/result.json"):
                run_dir = result_path.parent
                if run_dir.name in packaged:
                    continue
                try:
                    if package_run(model_root, run_dir, archive_base / "archives" / model, model not in based_models):
                        packaged.add(run_dir.name)
                        based_models.add(model)
                    else:
                        pending.append(run_dir.name)
                except Exception as exc:
                    failures.append({"run_id": run_dir.name, "error": repr(exc)})
                    pending.append(run_dir.name)
        monitor = base / "execution" / "main-workers.json"
        try:
            state = json.loads(monitor.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            # The controller updates this live journal in place. An external
            # volume reader can briefly observe an incomplete update.
            state = {}
        report = {"packaged_runs": len(packaged), "pending": pending, "errors": failures[-20:],
                  "updated_unix": time.time(),
                  "zstd_version": subprocess.check_output(["zstd", "--version"], text=True).strip()}
        save(base / "execution" / "archive-progress.json", report)
        volume.commit()
        if len(state.get("returned", {})) == len(MODELS) and not pending:
            return report
        time.sleep(15)


@app.local_entrypoint()
def main():
    call = archive_completed_runs.spawn()
    root = Path(__file__).resolve().parents[1] / "Finetune-result" / "execution"
    save(root / "archive-call.json", {"call_id": call.object_id, "app_id": app.app_id,
                                      "run": RUN, "gpu": None})
    print("CPU artifact packaging call:", call.object_id, flush=True)
    result = call.get()
    save(root / "archive-returned.json", result)
    print(json.dumps(result), flush=True)
