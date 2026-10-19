"""Explicit budgeted launch/retrieval for twelve continuous six-epoch fits.

Launch with `modal run --detach Epochs-test/modal_epoch_study.py --stage start`.
All training requires an explicit positive --budget. Other stages are CPU only.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import traceback

import modal

from tools.modal_experiments import ROOT, cache, copy_delta, hf_secret, image as base_image, outputs, save

HERE = ROOT / "Epochs-test"
sys.path.insert(0, str(HERE))
from epoch_study import MODELS, expected_grid, protocol  # noqa: E402

RUN = os.environ.get("FYP_EPOCH_RUN", "epochs-20261019")
if not RUN.replace("-", "").replace("_", "").isalnum():
    raise ValueError("FYP_EPOCH_RUN must be a single safe directory name")
RATE = 3600 * (0.000542 + 2 * 0.0000131 + 16 * 0.00000222)
LOCAL = HERE / "execution"
app = modal.App("fyp-epoch-trajectory-study", include_source=False)
image = base_image.uv_pip_install("python-docx==1.2.0", "python-pptx==1.0.2").env(
    {"PYTHONPATH": "/app:/app/tools:/app/Epochs-test", "FYP_EPOCH_RUN": RUN}
)
for name in (
    "epoch_study.py",
    "modal_epoch_study.py",
    "build_reports.py",
    "README.md",
    "requirements-report.txt",
):
    image = image.add_local_file(HERE / name, "/app/Epochs-test/" + name, copy=True)


@app.function(
    image=image,
    secrets=[hf_secret()],
    volumes={"/cache": cache, "/results": outputs},
    gpu="L40S",
    cpu=(2, 2),
    memory=(8192, 16384),
    timeout=10800,
    max_containers=4,
    single_use_containers=True,
    retries=0,
)
def train(model, scope, seconds, retry_failed=False):
    import torch
    from epoch_study import run_model

    torch.set_num_threads(4)
    local = Path("/work") / RUN / model
    remote = Path("/results") / RUN / "models" / model
    if remote.exists():
        copy_delta(remote, local)
    local.mkdir(parents=True, exist_ok=True)

    def flush():
        copy_delta(local, remote)
        outputs.commit()

    begin = time.monotonic()
    try:
        return run_model(model, scope, local, seconds, flush, retry_failed)
    except Exception as exc:
        failure = {
            "model": model,
            "status": "worker_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "worker_elapsed_s": time.monotonic() - begin,
        }
        save(local / "worker_failure.json", failure)
        return failure
    finally:
        flush()


@app.function(
    image=image,
    volumes={"/results": outputs},
    cpu=(0.125, 2),
    memory=(512, 4096),
    timeout=1200,
    single_use_containers=True,
    retries=0,
)
def report():
    from build_reports import build

    root = Path("/results") / RUN
    try:
        result = build(root)
        outputs.commit()
        return result
    except Exception as exc:
        failure = {"status": "report_failed", "error": repr(exc), "traceback": traceback.format_exc()}
        save(root / "execution" / "report_failure.json", failure)
        outputs.commit()
        return failure


def freeze_source(root):
    from epoch_study import study_source_digest

    destination = root / "execution" / "source"
    for name in (
        "rank_suite",
        "requirements.txt",
        "Epochs-test/epoch_study.py",
        "Epochs-test/modal_epoch_study.py",
        "Epochs-test/build_reports.py",
        "Epochs-test/README.md",
        "Epochs-test/requirements-report.txt",
        "tools/modal_experiments.py",
    ):
        source = Path("/app") / name
        target = destination / name
        if source.is_dir():
            shutil.copytree(
                source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
            )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    files = [
        {
            "path": str(p.relative_to(destination)),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted(destination.rglob("*"))
        if p.is_file()
    ]
    save(
        root / "execution" / "source_manifest.json",
        {"scientific_source_sha256": study_source_digest(), "files": files},
    )


@app.function(
    image=image,
    volumes={"/results": outputs},
    cpu=(0.125, 0.5),
    memory=(256, 2048),
    timeout=14400,
    single_use_containers=True,
    retries=0,
)
def coordinate(scope, budget, seconds):
    root = Path("/results") / RUN
    receipt = root / "execution" / "launch.json"
    if receipt.exists():
        raise ValueError(
            "This remote cohort was already launched; use status/collect, never silently duplicate it"
        )
    root.mkdir(parents=True, exist_ok=True)
    save(root / "protocol.json", protocol(scope))
    save(root / "expected_grid.json", expected_grid(scope))
    freeze_source(root)
    started = time.monotonic()
    # Includes cold-start/queue time, deliberately overestimating billable GPU time.
    calls = {model: train.with_options(timeout=seconds).spawn(model, scope, seconds) for model in MODELS}
    journal = {
        "run": RUN,
        "scope": scope,
        "approved_additional_usd": budget,
        "gpu_resource_ceiling_usd": budget - 0.20,
        "rate_ceiling_usd_per_worker_hour": RATE,
        "calls": {key: call.object_id for key, call in calls.items()},
        "started_unix": time.time(),
        "seconds_per_worker": seconds,
        "automatic_gpu_retries": False,
    }
    save(receipt, journal)
    outputs.commit()
    returned, finish = {}, {}
    while len(returned) < len(calls):
        for model, call in calls.items():
            if model in returned:
                continue
            try:
                returned[model] = call.get(timeout=0)
                finish[model] = time.monotonic()
            except (TimeoutError, modal.exception.TimeoutError):
                pass
            except Exception as exc:
                returned[model] = {"status": "worker_failed", "error": repr(exc)}
                finish[model] = time.monotonic()
        now = time.monotonic()
        conservative = sum((finish.get(model, now) - started) * RATE / 3600 for model in calls)
        if conservative >= budget - 0.30:
            for model, call in calls.items():
                if model not in returned:
                    call.cancel(terminate_containers=True)
                    returned[model] = {"status": "budget_stopped", "partial_metrics_preserved": True}
                    finish[model] = time.monotonic()
        journal.update(
            {
                "returned": returned,
                "conservative_gpu_cost_usd": conservative,
                "updated_unix": time.time(),
                "status": "training_finished" if len(returned) == len(calls) else "running",
            }
        )
        outputs.reload()
        save(root / "execution" / "status.json", journal)
        outputs.commit()
        if len(returned) < len(calls):
            time.sleep(10)
    # The detached parent stays alive until CPU reporting finishes.
    journal["report"] = report.remote()
    outputs.reload()
    journal["status"] = "completed" if journal["report"].get("complete") else "incomplete"
    save(root / "execution" / "status.json", journal)
    outputs.commit()
    return journal


@app.function(
    image=image,
    volumes={"/results": outputs},
    cpu=(0.125, 0.5),
    memory=(256, 2048),
    timeout=300,
    single_use_containers=True,
    retries=0,
)
def snapshot(make_archive=False):
    root = Path("/results") / RUN
    if not root.is_dir():
        return {"status": "not_launched", "run": RUN}
    read_issues = []

    def read(path, default=None):
        # A V2 reader may briefly observe an in-progress producer copy.
        for attempt in range(6):
            try:
                return json.loads(path.read_text())
            except FileNotFoundError:
                return default
            except (OSError, json.JSONDecodeError) as exc:
                if attempt == 5:
                    read_issues.append({"path": str(path.relative_to(root)), "error": type(exc).__name__})
                    return default
                time.sleep(0.1)

    status = read(root / "execution" / "status.json", {})
    progress = {}
    for model in MODELS:
        path = root / "models" / model / "progress.json"
        results = [read(p, {}) for p in (root / "models" / model / "runs").glob("*/result.json")]
        progress[model] = {
            "last": read(path),
            "completed": sum(r.get("status") == "completed" for r in results),
            "failed": [r["run_id"] for r in results if r.get("status") == "failed"],
        }
    result = {"run": RUN, "status": status, "progress": progress, "read_issues": read_issues}
    if make_archive:
        delivery = root / "execution" / "delivery.tar.gz"
        local = Path("/tmp/epoch-delivery.tar.gz")
        with tarfile.open(local, "w:gz") as archive:
            for path in sorted(root.rglob("*")):
                if (
                    path.is_file()
                    and path != delivery
                    and not path.name.endswith(".lock")
                    and "__pycache__" not in path.parts
                ):
                    archive.add(path, arcname=str(path.relative_to(root)))
        shutil.copy2(local, delivery)
        result["archive"] = {
            "bytes": delivery.stat().st_size,
            "sha256": hashlib.sha256(delivery.read_bytes()).hexdigest(),
        }
        outputs.commit()
    return result


@app.function(
    image=image,
    volumes={"/results": outputs.with_mount_options(read_only=True)},
    cpu=(0.125, 0.5),
    memory=(256, 1024),
    timeout=120,
    scaledown_window=10,
    max_containers=1,
    retries=0,
)
def read_chunk(offset, size=4 * 1024 * 1024):
    if offset < 0 or not 0 < size <= 4 * 1024 * 1024:
        raise ValueError("Invalid chunk bounds")
    with (Path("/results") / RUN / "execution" / "delivery.tar.gz").open("rb") as stream:
        stream.seek(offset)
        return stream.read(size)


def collect():
    result = snapshot.remote(True)
    if "archive" not in result:
        raise ValueError(result)
    target = HERE / "results"
    target.mkdir(parents=True, exist_ok=True)
    temporary = HERE / "execution" / "delivery.tar.gz.part"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    # New snapshots can change; small downloads restart, large original downloads are untouched.
    size = result["archive"]["bytes"]
    with temporary.open("wb") as stream:
        while stream.tell() < size:
            chunk = read_chunk.remote(stream.tell(), min(4 * 1024 * 1024, size - stream.tell()))
            if not chunk:
                raise IOError("Unexpected end of remote epoch archive")
            stream.write(chunk)
            print(f"Epoch study: {stream.tell() / 1e6:.1f}/{size / 1e6:.1f} MB", flush=True)
    if hashlib.sha256(temporary.read_bytes()).hexdigest() != result["archive"]["sha256"]:
        raise IOError("Epoch archive checksum mismatch")
    with tarfile.open(fileobj=io.BytesIO(temporary.read_bytes()), mode="r:gz") as archive:
        for member in archive.getmembers():
            destination = (target / member.name).resolve()
            if not destination.is_relative_to(target.resolve()) or not member.isfile():
                raise ValueError("Unsafe archive member")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, destination.open("wb") as sink:
                shutil.copyfileobj(source, sink)
    temporary.unlink()
    save(
        LOCAL / "collection.json",
        {"run": RUN, "archive": result["archive"], "completed_unix": time.time(), "output": str(target)},
    )
    for filename in ("epoch-study.docx", "epoch-study.pptx"):
        source = target / "reports" / filename
        if source.exists():
            shutil.copy2(source, HERE / filename)
    print(json.dumps({"output": str(target), "archive": result["archive"]}, indent=2))


@app.local_entrypoint()
def main(stage: str = "status", scope: str = "sst2", budget: float = 0, seconds: int = 3600):
    if scope != "sst2":
        raise ValueError("Only the authorized SST-2 scope is supported")
    LOCAL.mkdir(parents=True, exist_ok=True)
    if stage == "start":
        if not 1 <= budget <= 20 or not 300 <= seconds <= 10800:
            raise ValueError("Training requires an explicitly authorized budget $1–20 and 300–10800 seconds")
        receipt = LOCAL / "launch.json"
        if receipt.exists():
            raise ValueError("Local launch receipt exists; inspect status instead of duplicating training")
        call = coordinate.spawn(scope, budget, seconds)
        payload = {
            "run": RUN,
            "scope": scope,
            "budget": budget,
            "seconds": seconds,
            "call_id": call.object_id,
            "app_id": app.app_id,
            "started_unix": time.time(),
        }
        save(receipt, payload)
        print(json.dumps(payload, indent=2))
        # --detach keeps the app and its durable coordinator alive after a disconnect.
        print(json.dumps(call.get(), indent=2))
    elif stage == "collect":
        collect()
    elif stage == "status":
        print(json.dumps(snapshot.remote(), indent=2))
    elif stage == "report":
        print(json.dumps(report.remote(), indent=2))
    else:
        raise ValueError("stage must be start, status, collect, or report")
