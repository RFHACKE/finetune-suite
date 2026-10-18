"""Bundle finished-run metadata and a checked archive catalog on CPU only.

This avoids thousands of small remote reads during local retrieval. Original
training exports and per-run lossless archives are never rewritten.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import modal

RUN = os.environ.get("FYP_MODAL_RUN", "frw-20261018")
MODELS = ("llama3.2-1b", "qwen2.5-1.5b", "granite3.3-2b", "qwen2.5-3b")
app = modal.App("fyp-completed-delivery", include_source=False)
primary = modal.Volume.from_name("fyp-finetune-results", version=2)
vault = modal.Volume.from_name("fyp-finetune-archive")
image = (modal.Image.debian_slim(python_version="3.11").apt_install("zstd")
         .env({"PYTHONPATH": "/app", "FYP_MODAL_RUN": RUN})
         .add_local_file(__file__, "/app/modal_delivery.py", copy=True))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_tree(source, destination):
    for path in source.rglob("*"):
        if (not path.is_file() or path.name == "resume.pt"
                or path.name.endswith((".lock", ".tmp", ".pyc"))):
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


@app.function(image=image, volumes={"/results": primary.with_mount_options(read_only=True),
                                   "/vault": vault},
              cpu=(0.125, 2), memory=(256, 2048), timeout=900,
              single_use_containers=True, retries=0)
def prepare_delivery(inline_metadata=False):
    base = Path("/results") / RUN
    archive_base = Path("/vault") / RUN
    state_dir = base / "execution/unattended"
    if not (state_dir / "status.json").is_file():
        state_dir = base / "execution"
    status = json.loads((state_dir / "status.json").read_text())
    expected = json.loads((state_dir / "expected-grid.json").read_text())
    expected_ids = {row["run_id"] for row in expected}
    if (not expected or len(expected_ids) != len(expected)
            or status["completed_sequences"] != len(expected) or not status["training_finished"]):
        raise ValueError("Delivery catalog requires the finished requested cohort")
    metadata = Path("/work/metadata")
    metadata.mkdir(parents=True, exist_ok=True)
    copy_tree(base / "defaults", metadata)
    for folder in ("calibration", "benchmark", "execution", "overfitting"):
        copy_tree(base / folder, metadata / folder)
    # Normalize the preserved grid's location for both the original observer
    # cohort and the separately recorded overcomplete-factor extension.
    expected_path = metadata / "execution/unattended/expected-grid.json"
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    expected_path.write_text(json.dumps(expected, indent=2) + "\n")
    protocol, software, records = None, None, []
    for model in MODELS:
        source = base / "main" / model
        controls = json.loads((source / "protocol.json").read_text())
        packages = json.loads((source / "software.json").read_text())
        if protocol is not None and (controls != protocol or packages != software):
            raise ValueError("Worker protocol/software records differ")
        protocol, software = controls, packages
        for folder in ("data", "hardware", "invocations", "batches"):
            copy_tree(source / folder, metadata / folder)
        copy_tree(source / "logs", metadata / "logs" / model)
        for path in source.glob("failures*.jsonl"):
            shutil.copy2(path, metadata / f"failures-{model}-{path.name}")
        for name in ("worker-summary.json", "worker-error.json"):
            if (source / name).exists():
                shutil.copy2(source / name, metadata / "execution" / f"main-{model}-{name}")
        for path in sorted((source / "runs").glob("*/result.json")):
            result = json.loads(path.read_text())
            if result["status"] != "completed":
                raise ValueError("Incomplete run in finished cohort")
            target = metadata / "runs" / path.parent.name / "result.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            for name in ("train_log.jsonl", "epoch_diagnostics.jsonl", "overfitting.json"):
                if (path.parent / name).is_file():
                    shutil.copy2(path.parent / name, target.parent / name)
            copy_tree(path.parent / "diagnostics", target.parent / "diagnostics")
            manifest = json.loads((archive_base / "archives" / model / (path.parent.name + ".json")).read_text())
            archive = archive_base / "archives" / model / manifest["archive"]
            if (manifest["protocol_sha256"] != result["protocol_sha256"]
                    or archive.stat().st_size != manifest["archive_bytes"]):
                raise ValueError("Archive metadata mismatch")
            records.append({**manifest, "remote_path": str(archive.relative_to(Path("/vault")))})
    if len(records) != len(expected) or {r["run_id"] for r in records} != expected_ids:
        raise ValueError("Catalog does not cover the exact preserved requested cohort")
    for name, payload in (("protocol.json", protocol), ("software.json", software)):
        (metadata / name).write_text(json.dumps(payload, indent=2) + "\n")
    destination = archive_base / "archives" / "_delivery"
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "metadata.tar.zst"
    tar = subprocess.Popen(["tar", "-C", str(metadata), "-cf", "-", "."], stdout=subprocess.PIPE)
    compressor = subprocess.run(["zstd", "-T2", "-3", "--long=27", "--no-progress", "-q", "-f",
                                 "-o", str(archive)], stdin=tar.stdout, check=False)
    tar.stdout.close()
    if tar.wait() or compressor.returncode:
        raise RuntimeError("Metadata packaging failed")
    canonical = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    catalog = {"schema": "frw-completed-delivery-v1", "run": RUN, "training_finished": True,
               "expected_runs": len(expected), "protocol_sha256": canonical,
               "scientific_source_sha256": protocol["source_sha256"],
               "metadata": {"remote_path": str(archive.relative_to(Path("/vault"))),
                            "bytes": archive.stat().st_size, "sha256": digest(archive)},
               "archives": records}
    path = archive_base / "delivery/catalog.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, indent=2) + "\n")
    vault.commit()
    return {"catalog": catalog, "metadata_bytes": archive.read_bytes()} if inline_metadata else catalog


@app.local_entrypoint()
def main(target: str = "Finetune-result", inline_metadata: bool = False):
    call = prepare_delivery.spawn(inline_metadata)
    root = Path(target).resolve()
    destination = root / "execution"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "delivery-call.json").write_text(json.dumps({"call_id": call.object_id,
        "app_id": app.app_id, "gpu": None, "run": RUN}, indent=2) + "\n")
    returned = call.get()
    if inline_metadata:
        catalog, raw = returned["catalog"], returned["metadata_bytes"]
        if len(raw) != catalog["metadata"]["bytes"] or hashlib.sha256(raw).hexdigest() != catalog["metadata"]["sha256"]:
            raise ValueError("RPC-delivered metadata failed its independent cloud checksum")
        archive = destination / "metadata-inline.tar.zst"
        archive.write_bytes(raw)
        from tools.collect_delivery import extract_metadata
        extract_metadata(archive, root)
        (destination / "metadata-retrieval.json").write_text(json.dumps({
            "metadata_sha256": catalog["metadata"]["sha256"], "expected_runs": catalog["expected_runs"],
            "scientific_source_sha256": catalog["scientific_source_sha256"],
            "raw_training_exports_retrieved": False, "transport": "authenticated_modal_rpc"}, indent=2) + "\n")
    else:
        catalog = returned
    (destination / "delivery-catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    print(json.dumps({"runs": catalog["expected_runs"], "metadata_bytes": catalog["metadata"]["bytes"],
                      "total_archive_bytes": sum(r["archive_bytes"] for r in catalog["archives"])}))
