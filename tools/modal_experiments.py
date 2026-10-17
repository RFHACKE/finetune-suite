"""Explicit, staged Modal execution; no cloud calls occur on import.

Usage: modal run --detach tools/modal_experiments.py --stage prefetch
Stages: prefetch, benchmark, calibrate, merge-defaults, train, consolidate.
GPU stages require a positive --seconds allowance per worker. Each worker has
its own result directory; Linux advisory locks and active training use local
disk, with incremental durable copies after task checkpoints and every run.
Only rank_suite/, requirements.txt and this script enter the image. HF_TOKEN
is injected as an ephemeral Modal secret, never copied as a file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
RUN = os.environ.get("FYP_MODAL_RUN", "frw-20261018")
CACHE_NAME = "fyp-pinned-model-cache"
RESULTS_NAME = "fyp-finetune-results"
RANKS = [4, 8, 16, 32, 64, 128, 256]
MODELS = ["llama3.2-1b", "qwen2.5-1.5b", "granite3.3-2b", "qwen2.5-3b"]
app = modal.App("fyp-finetune-suite", include_source=False)
cache = modal.Volume.from_name(CACHE_NAME, create_if_missing=True, version=2)
outputs = modal.Volume.from_name(RESULTS_NAME, create_if_missing=True, version=2)


def hf_secret():
    if not modal.is_local():
        return modal.Secret.from_dict({})
    token = os.environ.get("HF_TOKEN")
    if not token:
        for line in (ROOT / "hf.env").read_text().splitlines():
            key, sep, value = line.removeprefix("export ").partition("=")
            if sep and key.strip() == "HF_TOKEN":
                token = value.strip().strip("\"'")
    if not token:
        raise ValueError("HF_TOKEN is missing from environment/hf.env")
    return modal.Secret.from_dict({"HF_TOKEN": token})


requirements = [line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines()
                if line.strip() and not line.startswith("#") and not line.startswith("torch==")]
image = (modal.Image.debian_slim(python_version="3.11")
         .uv_pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
         .uv_pip_install(*requirements)
         .env({"PYTHONPATH": "/app:/app/tools", "HF_HOME": "/cache/huggingface",
               "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4",
               "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
               "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1",
               "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_DATASETS_DISABLE_PROGRESS_BARS": "1"})
         .add_local_dir(ROOT / "rank_suite", "/app/rank_suite", copy=True,
                        ignore=["__pycache__", "*.pyc"])
         .add_local_file(ROOT / "requirements.txt", "/app/requirements.txt", copy=True)
         .add_local_file(Path(__file__), "/app/tools/modal_experiments.py", copy=True))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def copy_delta(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if not path.is_file() or path.name.endswith(".lock"):
            continue
        target = destination / path.relative_to(source)
        if (target.exists() and target.stat().st_size == path.stat().st_size
                and target.stat().st_mtime_ns == path.stat().st_mtime_ns):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def spec_for(path, model=None, **kwargs):
    from rank_suite.config import ExperimentSpec
    return ExperimentSpec(models=[model] if model else MODELS, track="lora", ranks=RANKS,
                          output_dir=str(path), micro_batch=16, accum=1, **kwargs)


@app.function(image=image, secrets=[hf_secret()], volumes={"/cache": cache, "/results": outputs},
              cpu=(2, 2), memory=(8192, 16384), timeout=1800, single_use_containers=True,
              retries=0)
def prefetch():
    from huggingface_hub import snapshot_download
    from rank_suite import config
    from rank_suite.data import prepare_task_data
    from rank_suite.model_utils import load_tokenizer
    begin = time.time()
    root = Path("/results") / RUN / "execution"
    def download(key):
        m = config.MODELS[key]
        snapshot_download(m.hf_id, revision=m.revision, max_workers=4,
                          allow_patterns=["*.json", "*.safetensors", "*.model",
                                          "tokenizer*", "merges.txt", "vocab.*"])
        return {"model": key, "revision": m.revision, "status": "downloaded"}
    # snapshot_download already parallelizes files. Nested thread_map contexts
    # across models race over tqdm's class lock, even with progress disabled.
    weights = [download(key) for key in MODELS]
    for key in MODELS:
        tok = load_tokenizer(config.MODELS[key])
        for calibration in (True, False):
            prepare_task_data(tok, selection_dir=Path("/work/prefetch-data") / key /
                              ("calibration" if calibration else "main"), calibration=calibration)
    copy_delta("/work/prefetch-data", root / "prefetch-data")
    report = {"status": "complete", "elapsed_s": time.time() - begin, "models": weights,
              "data_revision": config.DATA_REVISION, "requested_ranks": RANKS,
              "requested_seeds": list(config.DEFAULT_SEEDS), "main_sequences": 140}
    save(root / "prefetch.json", report)
    cache.commit()
    outputs.commit()
    return report


def durable_training(local, remote):
    """Keep suite locking local and persist after task boundaries and run exits."""
    from rank_suite import train
    original_save, original_run = train.save_torch, train.run_single
    def flush():
        copy_delta(local, remote)
        outputs.commit()
    def checkpoint(path, payload):
        original_save(path, payload)
        if Path(path).name == "resume.pt":
            flush()
    def run(*args, **kwargs):
        try:
            return original_run(*args, **kwargs)
        finally:
            flush()
    train.save_torch, train.run_single = checkpoint, run
    return flush


@app.function(image=image, secrets=[hf_secret()], volumes={"/cache": cache, "/results": outputs},
              gpu="H100", cpu=(2, 2), memory=(16384, 24576), timeout=900,
              max_containers=4, single_use_containers=True, retries=0)
def gpu_work(stage, model="qwen2.5-3b", seconds=600, gpu="H100"):
    import torch
    from rank_suite import config
    begin = time.time()
    torch.set_num_threads(4)
    remote_base = Path("/results") / RUN
    local = Path("/work") / stage / model
    remote = remote_base / stage / model
    if remote.exists():
        copy_delta(remote, local)
    local.mkdir(parents=True, exist_ok=True)
    flush = durable_training(local, remote)
    try:
        if stage == "benchmark":
            from rank_suite.data import prepare_task_data
            from rank_suite.model_utils import load_tokenizer
            from rank_suite.train import run_single
            from rank_suite.protocol import digest
            tok = load_tokenizer(config.MODELS[model])
            data = prepare_task_data(tok, calibration=True)
            for value in data.values():
                value.train_dataset = value.train_dataset.select(range(192))
            spec = spec_for(local, model, train_samples=192, epochs=1, calibrated_defaults={})
            measurements = []
            for rank in (4, 256):
                run = config.RunSpec(model, "lora", rank, config.DEFAULT_LORA_LR[model], 42,
                                     mode="lr_test", epochs=1, train_samples=192)
                result = run_single(run, local / "runs" / run.run_id, data, tok,
                                    config.MODELS[model], experiment=spec,
                                    protocol_sha=digest({"benchmark": RUN, "rank": rank}))
                measurements.append({"rank": rank, "peak_vram_gb": result["peak_vram_gb"],
                                     "wall_s": result["train_time_s"],
                                     "tasks": result["train_summary"]})
            report = {"status": "complete", "model": model, "gpu": torch.cuda.get_device_name(0),
                      "measurements": measurements, "elapsed_s": time.time() - begin,
                      "scientific_results": False,
                      "note": "192 training examples, one epoch; performance-only, excluded from LR/main evidence."}
        elif stage == "calibration":
            from rank_suite.calibration import run_calibration
            spec = spec_for(local, model, max_hours=max(60, seconds - 90) / 3600,
                            hourly_rate=4.35, calibrated_defaults={})
            report = run_calibration(spec)
        elif stage == "main":
            from rank_suite.runner import run_all
            defaults = remote_base / "defaults"
            copy_delta(defaults, local)
            spec = spec_for(local, model, max_hours=max(60, seconds - 90) / 3600,
                            hourly_rate=4.35)
            report = run_all(spec)
        else:
            raise ValueError(f"Unknown GPU stage: {stage}")
        report["worker_elapsed_s"] = time.time() - begin
        save(local / "worker-summary.json", report)
        return report
    except Exception as exc:
        import traceback
        save(local / "worker-error.json", {"error": repr(exc), "traceback": traceback.format_exc(),
                                          "elapsed_s": time.time() - begin})
        raise
    finally:
        flush()


@app.function(image=image, volumes={"/results": outputs}, cpu=(2, 2),
              memory=(4096, 8192), timeout=900, single_use_containers=True, retries=0)
def combine(stage):
    from rank_suite import config, utils
    from rank_suite.lr_defaults import SCHEMA, calibration_context
    from rank_suite.protocol import digest, source_digest
    base = Path("/results") / RUN
    if stage == "merge-defaults":
        root = base / "defaults"
        selected, context = {}, None
        for model in MODELS:
            source = base / "calibration" / model
            payload = utils.load_json(source / "lr_defaults.json")
            if not payload or payload.get("status") != "complete":
                raise ValueError(f"Calibration incomplete for {model}")
            if context is not None and payload["context"] != context:
                raise ValueError("Calibration contexts differ")
            context = payload["context"]
            selected.update(payload["selected"])
            for item in (source / "lr_test" / "experiments").iterdir():
                copy_delta(item, root / "lr_test" / "experiments" / item.name)
        expected = calibration_context(spec_for(root, calibrated_defaults={}))
        if context != expected:
            raise ValueError("Calibration source/training controls changed")
        payload = {"schema": SCHEMA, "status": "complete", "context": context, "selected": selected}
        payload["sha256"] = digest(payload)
        save(root / "lr_defaults.json", payload)
        # Exercise the ordinary evidence/default validation before main training.
        validated = spec_for(root)
        if set(validated.calibrated_defaults) != set(MODELS):
            raise ValueError("Missing model defaults")
        report = {"status": "complete", "selected": selected}
    elif stage == "consolidate":
        root = base / "study"
        copy_delta(base / "defaults", root)
        protocol = None
        completed = []
        for model in MODELS:
            source = base / "main" / model
            controls = utils.load_json(source / "protocol.json")
            if not controls:
                raise ValueError(f"Main worker never started: {model}")
            if protocol is not None and controls != protocol:
                raise ValueError("Cannot pool different experimental controls")
            protocol = controls
            for name in ("runs", "base", "hardware", "data", "invocations", "batches"):
                if (source / name).exists():
                    copy_delta(source / name, root / name)
            for name in ("protocol.json", "software.json"):
                shutil.copy2(source / name, root / name)
            if (source / "logs").exists():
                copy_delta(source / "logs", root / "logs" / model)
            for path in (source / "runs").glob("*/result.json"):
                value = utils.load_json(path)
                if value.get("status") == "completed":
                    completed.append((value["model"], value["rank"], value["seed"]))
        expected = {(m, r, s) for m in MODELS for r in RANKS for s in config.DEFAULT_SEEDS}
        report = {"status": "complete" if set(completed) == expected and len(completed) == 140 else "partial",
                  "expected_sequences": 140, "completed_sequences": len(completed),
                  "missing": sorted(expected - set(completed)), "ranks": RANKS,
                  "seeds": list(config.DEFAULT_SEEDS), "source_sha256": source_digest(),
                  "scope": "Requested LoRA-only cohort; full-FT baseline-dependent feasibility is not certified."}
        save(root / "requested-completeness.json", report)
        shutil.copytree("/app/rank_suite", root / "source" / "rank_suite", dirs_exist_ok=True)
        shutil.copy2("/app/requirements.txt", root / "source" / "requirements.txt")
        shutil.copy2("/app/tools/modal_experiments.py", root / "source" / "modal_experiments.py")
    else:
        raise ValueError(stage)
    save(base / "execution" / f"{stage}.json", report)
    outputs.commit()
    return report


@app.function(image=image, volumes={"/results": outputs}, cpu=(0.125, 0.125),
              memory=(256, 1024), timeout=28800, single_use_containers=True, retries=0)
def coordinate(stage, models, seconds, gpu):
    """A durable CPU parent keeps every parallel child alive if the client exits."""
    calls = {model: gpu_work.with_options(gpu=gpu, timeout=seconds).spawn(
        stage, model=model, seconds=seconds, gpu=gpu) for model in models}
    journal = {"stage": stage, "seconds_per_worker": seconds, "gpu": gpu,
               "created_unix": time.time(),
               "calls": {model: call.object_id for model, call in calls.items()}}
    path = Path("/results") / RUN / "execution" / f"{stage}-workers.json"
    save(path, journal)
    outputs.commit()
    results = {}
    for model, call in calls.items():
        try:
            results[model] = call.get()
        except Exception as exc:
            results[model] = {"status": "worker_failed", "error": repr(exc)}
        journal["returned"] = results
        save(path, journal)
        outputs.commit()
    return results


@app.local_entrypoint()
def main(stage: str = "prefetch", models: str = "", seconds: int = 600, gpu: str = "H100"):
    state = ROOT / "Finetune-result" / "execution"
    state.mkdir(parents=True, exist_ok=True)
    selected_models = models.split(",") if models else MODELS
    if any(m not in MODELS for m in selected_models):
        raise ValueError("Use the fixed model registry keys")
    if stage == "prefetch":
        calls = [("prefetch", prefetch.spawn())]
    elif stage in ("benchmark", "calibration", "main"):
        if seconds < 120:
            raise ValueError("An explicit per-worker time allowance of at least 120 seconds is required")
        if stage == "benchmark" and not models:
            selected_models = ["qwen2.5-3b"]
        calls = [("coordinator", coordinate.spawn(stage, selected_models, seconds, gpu))]
    elif stage in ("merge-defaults", "consolidate"):
        calls = [(stage, combine.spawn(stage))]
    else:
        raise ValueError(stage)
    journal = {"run": RUN, "stage": stage, "app_id": app.app_id, "gpu": gpu,
               "seconds_per_worker": seconds, "created_unix": time.time(),
               "calls": {model: call.object_id for model, call in calls}, "results_volume": RESULTS_NAME}
    save(state / f"{stage}-calls.json", journal)
    print(json.dumps(journal, indent=2), flush=True)
    results = {model: call.get() for model, call in calls}
    save(state / f"{stage}-returned.json", results)
    print(json.dumps(results, indent=2), flush=True)
