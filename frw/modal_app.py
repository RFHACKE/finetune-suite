"""Modal app for frw-v4: one self-driving deployment per account.

Each account owns a queue (files on its own Volume), atomic job claims and a
cost ledger (a modal.Dict), GPU workers that run jobs until the queue, the
budget or the deadline is exhausted, and a cron watchdog that respawns workers,
advances LR selection into reported runs and enforces the spending cap. Nothing
depends on the local machine once deployed.

Deploy (the supervisor does this per account with that account's token):
    FRW_GPU=L4 modal deploy frw/modal_app.py
"""
from __future__ import annotations

import json
import os
import time
import traceback
import uuid
from pathlib import Path

import modal

APP = "frw-v4"
GPU = os.environ.get("FRW_GPU", "L4")
CPU_CORES = 2.0
MEM_MIB = 12288
MEM_CAP_MIB = 16384
# $/s list prices (modal billing rates, 2026-10-25). CPU/memory at the capped limits.
GPU_RATE = {"T4": 0.59, "L4": 0.80, "A10G": 1.10, "L40S": 1.95, "A100-40GB": 2.10,
            "A100-80GB": 2.50, "H100": 3.95}
CPU_RATE_S = 0.0000131
MEM_RATE_S = 0.00000222
WORKER_RATE_S = GPU_RATE[GPU] / 3600 + CPU_CORES * CPU_RATE_S + (MEM_CAP_MIB / 1024) * MEM_RATE_S
STARTUP_ALLOWANCE_S = 120            # image load + model load before the ledger starts
WORKER_TIMEOUT_S = 6 * 3600
STALE_S = 15 * 60                    # claim/heartbeat staleness

ROOT = Path(__file__).resolve().parents[1]
app = modal.App(APP)
results = modal.Volume.from_name("frw-v4-results", create_if_missing=True)
hfcache = modal.Volume.from_name("frw-hf-cache", create_if_missing=True)
state = modal.Dict.from_name("frw-v4-state", create_if_missing=True)

image = (modal.Image.debian_slim(python_version="3.11")
         .uv_pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
         .uv_pip_install("transformers==4.57.6", "peft==0.17.1", "datasets==4.4.1", "accelerate==1.10.1",
                         "safetensors==0.6.2", "numpy==2.2.6", "huggingface-hub==0.36.0",
                         "tokenizers==0.22.1", "sentencepiece==0.2.1", "protobuf==6.33.0",
                         "hf-xet==1.7.0", "scipy==1.15.3")
         .env({"HF_HOME": "/hf", "FRW_DATA_DIR": "/vol/frw-v4/data", "TOKENIZERS_PARALLELISM": "false",
               "PYTHONUNBUFFERED": "1",
               "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
               "OMP_NUM_THREADS": "2", "FRW_GPU": GPU})
         .add_local_python_source("frw"))
secrets = [modal.Secret.from_name("frw-hf")]
VOL = Path("/vol")
BASE = VOL / "frw-v4"


# ---------------------------------------------------------------------------
# helpers (run inside containers)
# ---------------------------------------------------------------------------
def _save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{uuid.uuid4().hex[:6]}")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def _load(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _config():
    return _load(BASE / "config.json", {})


def _queue():
    out = []
    qdir = BASE / "queue"
    if qdir.exists():
        for f in qdir.glob("*.json"):
            d = _load(f)
            if d:
                out.append(d)
    return out


def _done(job_id: str) -> bool:
    return (BASE / "runs" / job_id / "result.json").exists() or (BASE / "dead" / f"{job_id}.json").exists()


def _spend_estimate():
    """Conservative spend from the ledger (every container that ever ran here)."""
    total = 0.0
    for k, v in state.items():
        if isinstance(k, str) and k.startswith("ledger:"):
            total += (max(0.0, v["last"] - v["start"]) + STARTUP_ALLOWANCE_S) * v["rate"]
    total += float(state.get("overhead_usd", 0.0))
    # Real metered spend reported by the supervisor (billing lags, so it is only a floor),
    # and the persisted high-water mark (Dict entries expire after 7 idle days).
    floor = max(float(state.get("billing_floor", 0.0)), float(state.get("spend_high_water", 0.0)))
    est = max(total, floor)
    if est > float(state.get("spend_high_water", 0.0)):
        state.put("spend_high_water", est)
    return est


def _job_cost(job: dict, cfg: dict) -> float:
    est = cfg.get("job_seconds", {})
    key = f"{job['model']}|{job['split']}"
    base_s = est.get(key, 900.0)
    r = job.get("rank") or 0
    mult = 1.0 + (0.6 if r >= 1024 else 0.3 if r >= 512 else 0.12 if r >= 256 else 0.0)
    if job.get("kind") in ("main", "main_al"):
        mult *= 1.0 + cfg.get("probe_overhead", 0.25)
    kind = str(job.get("kind", ""))
    if kind.startswith("ep") and "_" in kind:
        mult *= 0.75 * (0.3 + 0.7 * int(kind[2:kind.index("_")]) / 3)   # no probes; scales with epochs
    if job.get("kind") == "stab":
        mult *= cfg.get("stab_factor", 0.75)          # official split without probes/adapters
    if str(job.get("kind", "")).endswith("_al"):
        mult *= 1.0 + cfg.get("all_linear_overhead", 0.6)
    return base_s * mult * WORKER_RATE_S


def _halted(cfg) -> str | None:
    if state.get("halt"):
        return "halt flag"
    deadline = cfg.get("deadline_unix")
    if deadline and time.time() > deadline:
        return "deadline"
    return None


def _claim(job_id: str, owner: str) -> int:
    """Atomically claim a job; returns the attempt number or -1 if someone else holds it."""
    for attempt in range(6):
        key = f"claim:{job_id}:{attempt}"
        if state.put(key, {"owner": owner, "ts": time.time()}, skip_if_exists=True):
            return attempt
        cur = state.get(key)
        if cur is None:
            continue
        hb = state.get(f"hb:{job_id}:{attempt}")
        last = max(cur["ts"], hb["ts"] if hb else 0.0)
        if time.time() - last < STALE_S:
            return -1                      # actively held by another worker
    return -1


def _held(job_id: str) -> bool:
    for attempt in range(6):
        cur = state.get(f"claim:{job_id}:{attempt}")
        if cur is None:
            return False
        hb = state.get(f"hb:{job_id}:{attempt}")
        if time.time() - max(cur["ts"], hb["ts"] if hb else 0.0) < STALE_S:
            return True
    return False


def advance(log=print):
    """Turn completed LR searches into reported jobs (idempotent; safe to run anywhere)."""
    from frw import config as C
    from frw import jobs as J
    cfg = _config()
    queue = {d["job_id"]: d for d in _queue()}
    created = 0
    for pair in cfg.get("pairs", []):
        model, rank = pair["model"], pair["rank"]
        suffix = pair.get("suffix", "")
        sel_path = BASE / "selections" / f"{model}__r{rank}{suffix}.json"
        if sel_path.exists():
            continue
        tunes = [d for d in queue.values() if d["spec"]["kind"] == "tune" + suffix and d["spec"]["model"] == model
                 and d["spec"]["rank"] == rank]
        by_lr = {}
        for d in tunes:
            res = _load(BASE / "runs" / d["job_id"] / "result.json")
            if res is None and (BASE / "dead" / f"{d['job_id']}.json").exists():
                res = {"status": "failed"}
            by_lr.setdefault(d["spec"]["lr"], [])
            if res is not None:
                by_lr[d["spec"]["lr"]].append(res)
        if not by_lr:
            continue
        status, lr, extra, table = J.select_lr(by_lr, len(C.TUNE_SEEDS), C.LORA_LR_MIN, C.LORA_LR_MAX)
        if status == "extend":
            for job in J.tune_jobs(model, ranks=[rank], lrs=[extra], suffix=suffix):
                if job.job_id not in queue:
                    _enqueue(job)
                    created += 1
        elif status == "selected":
            new = J.main_jobs(model, rank, lr, suffix=suffix) + (
                J.order_jobs(model, rank, lr) if cfg.get("orders", True) and not suffix else [])
            for job in new:
                if job.job_id not in queue:
                    _enqueue(job)
                    created += 1
            _save(sel_path, {"model": model, "rank": rank, "suffix": suffix, "selected_lr": lr,
                             "table": {str(k): v for k, v in table.items()},
                             "rule": "argmax mean final-row holdout accuracy over tuning seeds; ties->lower LR;"
                                     " edge winners extended by factor 2", "ts": time.time()})
            log(f"selected {model} r={rank}: lr={lr} table={table}")
    return created


def _enqueue(job):
    _save(BASE / "queue" / f"{job.job_id}.json", {"job_id": job.job_id, "spec": job.spec,
                                                   "priority": job.priority})


def _candidates(current_model=None):
    rows = [d for d in _queue() if not _done(d["job_id"])]
    rows.sort(key=lambda d: (d["priority"] // 100, d["spec"]["model"] != current_model, d["priority"]))
    return rows


def _runnable(cfg, current_model=None):
    return [d for d in _candidates(current_model) if not _held(d["job_id"])]


# ---------------------------------------------------------------------------
# GPU worker
# ---------------------------------------------------------------------------
@app.function(image=image, gpu=GPU, cpu=(CPU_CORES, CPU_CORES), memory=(MEM_MIB, MEM_CAP_MIB),
              timeout=WORKER_TIMEOUT_S, volumes={str(VOL): results, "/hf": hfcache}, secrets=secrets,
              retries=0, max_containers=10)
def worker(slot: int):
    import threading

    import torch

    from frw import train as T
    from frw.jobs import Job

    t_start = time.time()
    uid = f"{slot}-{uuid.uuid4().hex[:8]}"
    led_key = f"ledger:{uid}"
    state.put(led_key, {"slot": slot, "start": t_start, "last": t_start, "rate": WORKER_RATE_S, "gpu": GPU})
    stop = threading.Event()
    current = {"job": None, "attempt": -1}

    def beat():
        while not stop.is_set():
            now = time.time()
            try:
                state.put(led_key, {"slot": slot, "start": t_start, "last": now, "rate": WORKER_RATE_S, "gpu": GPU})
                state.put(f"slot:{slot}", {"uid": uid, "ts": now, "job": current["job"]})
                if current["job"]:
                    state.put(f"hb:{current['job']}:{current['attempt']}", {"ts": now})
            except Exception:  # noqa: BLE001 - heartbeat must never kill training
                pass
            stop.wait(60)
    threading.Thread(target=beat, daemon=True).start()

    ctx = None
    done_jobs = 0
    log_path = BASE / "logs" / f"worker-{uid}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [{uid}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    try:
        while True:
            cfg = _config()
            reason = _halted(cfg)
            if reason:
                log(f"stop: {reason}")
                break
            if time.time() - t_start > WORKER_TIMEOUT_S - 50 * 60:
                log("stop: worker time limit")
                break
            results.reload()
            d, attempt = None, -1
            for cand in _candidates(ctx.m.key if ctx else None):
                if _held(cand["job_id"]):
                    continue
                d = cand
                break
            if d is None:
                if advance(log):
                    results.commit()
                    continue
                log("stop: no runnable jobs")
                break
            spend = _spend_estimate()
            busy = sum(1 for k, v in state.items() if isinstance(k, str) and k.startswith("slot:")
                       and v.get("job") and v.get("uid") != uid and time.time() - v["ts"] < 300)
            projected = spend + _job_cost(d["spec"], cfg) * (1.25 + busy)
            if projected > cfg.get("cap_usd", 0):
                log(f"stop: budget (spent~{spend:.2f}, projected {projected:.2f} > cap {cfg.get('cap_usd')})")
                state.put("budget_exhausted", {"ts": time.time(), "spend": spend})
                break
            attempt = _claim(d["job_id"], uid)
            if attempt < 0:
                continue
            current.update(job=d["job_id"], attempt=attempt)
            job = Job.from_spec(d["spec"], d["priority"])
            log(f"start {job.job_id} (attempt {attempt}, spent~{spend:.2f})")
            t_job = time.time()
            try:
                if ctx is None or ctx.m.key != job.model:
                    ctx = None
                    torch.cuda.empty_cache()
                    ctx = T.Context(job.model, torch.device("cuda"), ref_dir=BASE / "refs")
                out = BASE / "runs" / job.job_id
                res = T.run(job, ctx, out, probes=job.kind in ("main", "main_al"),
                            save_adapter=job.kind in ("main", "main_al"),
                            log=log)
                res["worker"] = {"uid": uid, "attempt": attempt, "gpu_type": GPU,
                                 "job_wall_s": time.time() - t_job}
                _save(out / "result.json", res)
                done_jobs += 1
                log(f"done {job.job_id} status={res['status']} wall={time.time() - t_job:.0f}s "
                    f"acc={res['metrics']['acc_final'] if res['metrics'] else None}")
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                log(f"FAILED {job.job_id}: {exc}")
                _save(BASE / "failures" / f"{job.job_id}__{attempt}.json",
                      {"job_id": job.job_id, "attempt": attempt, "error": str(exc), "traceback": tb,
                       "ts": time.time(), "uid": uid})
                if attempt >= 1:
                    _save(BASE / "dead" / f"{job.job_id}.json", {"job_id": job.job_id, "error": str(exc)})
                ctx = None
                torch.cuda.empty_cache()
            finally:
                current.update(job=None, attempt=-1)
                results.commit()
    finally:
        stop.set()
        now = time.time()
        state.put(led_key, {"slot": slot, "start": t_start, "last": now, "rate": WORKER_RATE_S, "gpu": GPU})
        state.put(f"slot:{slot}", {"uid": uid, "ts": 0, "job": None, "ended": now})
        results.commit()
    return {"uid": uid, "jobs": done_jobs, "elapsed_s": time.time() - t_start}


# ---------------------------------------------------------------------------
# Watchdog (cron) and utilities
# ---------------------------------------------------------------------------
WATCHDOG_RATE_S = 0.25 * CPU_RATE_S + 0.5 * MEM_RATE_S


def _real_billing():
    """(metered USD this calendar month, error) using the container's own workspace identity."""
    try:
        s = modal.Workspace.from_context().billing.summary()
        return float(s.metered_cost), None
    except Exception as exc:  # noqa: BLE001
        return None, repr(exc)[:300]


def _stop_self(reason: str):
    """Halt workers, cancel their calls and stop this app (cron included). Best effort, no stored tokens."""
    out = {"reason": reason, "ts": time.time(), "cancelled": [], "app_stop": None}
    state.put("halt", True)
    for s in range(10):
        v = state.get(f"slot:{s}")
        if v and v.get("call_id"):
            try:
                modal.FunctionCall.from_id(v["call_id"]).cancel(terminate_containers=True)
                out["cancelled"].append(s)
            except Exception as exc:  # noqa: BLE001
                out.setdefault("cancel_errors", []).append(repr(exc)[:120])
    _save(BASE / "stopped.json", out)
    results.commit()
    try:
        import asyncio

        from modal.client import _Client
        from modal_proto import api_pb2
        app_id = modal.App.lookup(APP).app_id

        async def _stop():
            client = await _Client.from_env()
            await client.stub.AppStop(api_pb2.AppStopRequest(
                app_id=app_id, source=api_pb2.APP_STOP_SOURCE_PYTHON_CLIENT))
        asyncio.run(_stop())
        out["app_stop"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["app_stop"] = repr(exc)[:300]
    return out


@app.function(image=image, cpu=0.25, memory=512, timeout=600, volumes={str(VOL): results},
              schedule=modal.Period(minutes=10), retries=0)
def watchdog():
    t0 = time.time()
    results.reload()
    cfg = _config()
    if not cfg:
        return {"status": "unconfigured"}
    created = advance()
    metered, billing_error = _real_billing()
    if metered is not None:
        state.put("billing_floor", metered)
        state.put("billing_floor_ts", time.time())
    spend = _spend_estimate()
    hard = cfg.get("hard_cap_usd", 29.0)
    if spend >= hard or (metered is not None and metered >= hard):
        state.put("halt", True)
        _save(BASE / "status.json", {"ts": time.time(), "halt_reason": "hard cap", "spend_estimate_usd": spend,
                                     "metered_usd": metered})
        rc = _stop_self(f"hard cap reached: estimate {spend:.2f}, metered {metered}")
        return {"status": "hard cap: app stop", "rc": rc}
    reason = _halted(cfg)
    runnable = _runnable(cfg)
    alive = []
    for s in range(cfg.get("workers", 2)):
        v = state.get(f"slot:{s}")
        if v and v.get("ts") and time.time() - v["ts"] < 8 * 60:
            alive.append(s)
        elif v and not v.get("ts") and time.time() - v.get("spawned", 0) < 10 * 60:
            alive.append(s)              # spawned recently, still starting
    spawned = []
    if not reason and runnable and not state.get("budget_exhausted"):
        for s in range(cfg.get("workers", 2)):
            if s in alive or len(alive) + len(spawned) >= min(cfg.get("workers", 2), len(runnable)):
                continue
            call = worker.spawn(s)
            state.put(f"slot:{s}", {"uid": None, "ts": 0, "job": None, "spawned": time.time(),
                                    "call_id": call.object_id})
            spawned.append(s)
    q = _queue()
    done = sum(1 for d in q if (BASE / "runs" / d["job_id"] / "result.json").exists())
    dead = sum(1 for d in q if (BASE / "dead" / f"{d['job_id']}.json").exists())
    status = {"ts": time.time(), "spend_estimate_usd": round(spend, 4), "metered_usd": metered,
              "billing_error": billing_error, "cap_usd": cfg.get("cap_usd"),
              "queued": len(q), "done": done, "dead": dead, "runnable": len(runnable),
              "alive_slots": alive, "spawned": spawned, "halt_reason": reason, "created": created,
              "budget_exhausted": state.get("budget_exhausted")}
    _save(BASE / "status.json", status)
    state.put("overhead_usd", float(state.get("overhead_usd", 0.0)) + (time.time() - t0 + 15) * WATCHDOG_RATE_S)
    results.commit()
    idle = not alive and not spawned and (not runnable or reason or state.get("budget_exhausted"))
    if idle and q and not cfg.get("smoke"):
        quiet = state.get("idle_since") or time.time()
        state.put("idle_since", quiet)
        if time.time() - quiet > 25 * 60:      # three quiet checks in a row: nothing left to do here
            status["self_stop"] = _stop_self(f"finished or stopped ({reason or 'no runnable jobs'})")
    elif state.contains("idle_since"):
        state.pop("idle_since")
    return status


@app.function(image=image, cpu=2.0, memory=8192, timeout=3600, volumes={str(VOL): results, "/hf": hfcache},
              secrets=secrets, retries=1)
def prefetch(models: list):
    from huggingface_hub import snapshot_download

    from frw import config as C
    from frw.data import export_raw, load_task
    from frw.model import load_tokenizer
    t0 = time.time()
    out = {}
    data_dir = BASE / "data"
    if not all((data_dir / f"{t}.json").exists() for t in C.TASKS):
        out["data_sha256"] = export_raw(data_dir)
        results.commit()
    os.environ["FRW_DATA_DIR"] = str(data_dir)
    for key in models:
        m = C.MODELS[key]
        snapshot_download(m.hf_id, revision=m.revision,
                          allow_patterns=["*.json", "*.safetensors", "tokenizer*", "merges.txt", "vocab.*"])
        tok = load_tokenizer(m)
        for split in ("holdout", "official"):
            for t in C.TASKS:
                td = load_task(t, tok, split)
                out[f"{key}/{t}/{split}"] = td.provenance
    hfcache.commit()
    _save(BASE / "prefetch.json", {"elapsed_s": time.time() - t0, "data": out})
    state.put("overhead_usd", float(state.get("overhead_usd", 0.0)) + (time.time() - t0 + 30) * (
        2 * CPU_RATE_S + 8 * MEM_RATE_S))
    results.commit()
    return {"elapsed_s": time.time() - t0, "n": len(out)}


@app.function(image=image, gpu=GPU, cpu=(CPU_CORES, CPU_CORES), memory=(MEM_MIB, MEM_CAP_MIB), timeout=1800,
              volumes={str(VOL): results, "/hf": hfcache}, secrets=secrets, retries=0)
def bench(model: str, ranks: list, steps: int = 60):
    """Throughput probe: training steps/s at given ranks plus evaluation examples/s."""

    import torch
    import torch.nn.functional as F

    from frw.data import batches
    from frw.model import features, make_heads, trainable, transformer_of, wrap_lora
    from frw.train import Context, autocast, evaluate
    t0 = time.time()
    ctx = Context(model, torch.device("cuda"))
    base = ctx.base_model()
    t_load = time.time() - t0
    out = {"gpu": torch.cuda.get_device_name(0), "load_s": t_load, "ranks": {}}
    for task in ("rte", "mnli"):
        ctx.task(task, "official")
    for r in ranks:
        pm = wrap_lora(base, r).cuda()
        tr = transformer_of(pm)
        heads = make_heads(("rte", "mnli", "sst2"), ctx.m.d_model, 1).cuda()
        params = trainable(pm) + list(heads[0].parameters())
        opt = torch.optim.AdamW(params, lr=1e-4)
        torch.cuda.synchronize()
        n = 0
        ts = None
        for ids, mask, labels, _ in batches(ctx.task("rte", "official").train, 16, ctx.tok.pad_token_id,
                                            shuffle_seed=1):
            if n == 5:
                torch.cuda.synchronize()
                ts = time.time()
            with autocast(torch.device("cuda")):
                f = features(tr, ids.cuda(), mask.cuda())
            loss = F.cross_entropy(heads[0](f), labels.cuda())
            loss.backward()
            opt.step()
            opt.zero_grad()
            n += 1
            if n >= steps + 5:
                break
        torch.cuda.synchronize()
        step_s = (time.time() - ts) / steps
        te = time.time()
        ev = evaluate(tr, heads[1], ctx.task("mnli", "official").eval, ctx.tok.pad_token_id, torch.device("cuda"))
        eval_s = time.time() - te
        out["ranks"][r] = {"rte_step_s": step_s, "mnli_eval_s": eval_s,
                           "mnli_eval_per_ex_ms": 1000 * eval_s / len(ctx.task("mnli", "official").eval),
                           "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "acc_untrained": ev["accuracy"]}
        ctx.base = pm.unload()
        base = ctx.base
        if hasattr(base, "peft_config"):
            del base.peft_config
        del opt, heads
        torch.cuda.empty_cache()
    out["wall_s"] = time.time() - t0
    out["rate_usd_s"] = WORKER_RATE_S
    return out


@app.function(image=image, cpu=0.25, memory=512, timeout=300, volumes={str(VOL): results})
def control(action: str, payload: dict | None = None):
    """Small remote control surface used by the local supervisor."""
    results.reload()
    if action == "configure":
        _save(BASE / "config.json", payload)
    elif action == "enqueue":
        from frw.jobs import Job
        for d in payload["jobs"]:
            job = Job.from_spec(d["spec"], d["priority"])
            if not (BASE / "queue" / f"{job.job_id}.json").exists():
                _enqueue(job)
    elif action == "dequeue":
        for job_id in payload["job_ids"]:
            p = BASE / "queue" / f"{job_id}.json"
            if p.exists() and not _done(job_id):
                p.unlink()
    elif action == "halt":
        state.put("halt", bool(payload.get("halt", True)))
    elif action == "reset_budget_flag":
        if state.contains("budget_exhausted"):
            state.pop("budget_exhausted")
    elif action == "advance":
        advance()
    elif action == "pending":
        rows = []
        for d in _candidates():
            if (d["spec"]["kind"] in ("main", "order", "stab", "main_al") or d["spec"]["kind"].startswith("ep")) \
                    and not _held(d["job_id"]):
                rows.append({"job_id": d["job_id"], "spec": d["spec"], "priority": d["priority"],
                             "est_usd": _job_cost(d["spec"], _config())})
        return {"pending": rows, "spend_estimate_usd": _spend_estimate(),
                "budget_exhausted": state.get("budget_exhausted"), "halt": state.get("halt")}
    elif action == "unhalt":
        state.put("halt", False)
        if state.contains("idle_since"):
            state.pop("idle_since")
    elif action == "selftest":
        metered, err = _real_billing()
        try:
            app_id = modal.App.lookup(APP).app_id
        except Exception as exc:  # noqa: BLE001
            app_id = repr(exc)[:200]
        return {"metered": metered, "billing_error": err, "app_id": app_id}
    elif action == "billing_floor":
        if payload and payload.get("usd") is not None:
            state.put("billing_floor", float(payload["usd"]))
            state.put("billing_floor_ts", time.time())
    results.commit()
    ledger = {k: v for k, v in state.items() if isinstance(k, str) and k.startswith(("ledger:", "slot:"))}
    return {"config": _config(), "status": _load(BASE / "status.json"), "spend_estimate_usd": _spend_estimate(),
            "halt": state.get("halt"), "budget_exhausted": state.get("budget_exhausted"),
            "n_queue": len(_queue()), "ledger": ledger}
