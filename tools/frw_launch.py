"""Deploy, prefetch, configure and start the frw-v4 study on the six-account pool.

    python3 -m tools.frw_launch --plan            # print the assignment and cost forecast only
    python3 -m tools.frw_launch --go --accounts 1,2,3,4,5,6

Account plan (LoRA only, L40S): Granite and Qwen2.5-3B are split over two accounts each by rank;
Qwen2.5-1.5B and Llama-3.2-1B get one account each. Each (model, rank) pair stays inside one
account so its LR search, selection and reported runs never cross accounts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frw import config as C  # noqa: E402
from frw import jobs as J  # noqa: E402
from tools.frw_supervisor import EXEC, control, deploy, fn, log, save  # noqa: E402

GPU = "L40S"
WORKERS = 5
CAP_USD = 26.0
HARD_CAP_USD = 29.0
DEADLINE = dt.datetime(2026, 10, 31, 22, 0, tzinfo=dt.timezone.utc)   # keep all spend in October's credit
SPLIT_A = (1, 16, 256, 2, 32, 512)
SPLIT_B = (4, 64, 1024, 8, 128)
PLAN = {
    1: ("granite3.3-2b", SPLIT_A),
    2: ("granite3.3-2b", SPLIT_B),
    3: ("qwen2.5-3b", SPLIT_A),
    4: ("qwen2.5-3b", SPLIT_B),
    5: ("qwen2.5-1.5b", C.RANKS),
    6: ("llama3.2-1b", C.RANKS),
}
# Seconds per job on L40S (smoke/benchmark measured; refined by the ledger as jobs finish).
JOB_SECONDS = {
    "llama3.2-1b|holdout": 100, "llama3.2-1b|official": 160,
    "qwen2.5-1.5b|holdout": 150, "qwen2.5-1.5b|official": 250,
    "granite3.3-2b|holdout": 290, "granite3.3-2b|official": 470,
    "qwen2.5-3b|holdout": 270, "qwen2.5-3b|official": 450,
}
RATE_S = (1.95 / 3600) + 2 * 0.0000131 + 16 * 0.00000222


def forecast():
    rows, total = [], 0.0
    for n, (model, ranks) in PLAN.items():
        secs = 0.0
        for r in ranks:
            mult = 1.0 + (0.6 if r >= 1024 else 0.3 if r >= 512 else 0.12 if r >= 256 else 0.0)
            tune = 8.8 * JOB_SECONDS[f"{model}|holdout"] * mult
            main = 5 * JOB_SECONDS[f"{model}|official"] * mult * 1.0
            order = (6 * JOB_SECONDS[f"{model}|official"] * 0.85 * mult) if r in C.ORDER_RANKS else 0.0
            secs += tune + main + order
        usd = secs * RATE_S
        total += usd
        rows.append({"account": n, "model": model, "ranks": list(ranks), "gpu_hours": secs / 3600, "usd": usd})
    return rows, total


def configure(n, model, ranks, dry=False):
    cfg = {"pairs": [{"model": model, "rank": r} for r in ranks], "cap_usd": CAP_USD,
           "hard_cap_usd": HARD_CAP_USD, "workers": WORKERS, "deadline_unix": DEADLINE.timestamp(),
           "job_seconds": JOB_SECONDS, "orders": True, "gpu": GPU, "protocol": C.PROTOCOL,
           "configured": dt.datetime.now(dt.timezone.utc).isoformat()}
    jobs = J.tune_jobs(model, ranks=tuple(ranks))
    payload = {"jobs": [{"spec": j.spec, "priority": j.priority} for j in jobs]}
    if dry:
        return cfg, payload
    control(n, "configure", cfg)
    st = control(n, "enqueue", payload)
    log(f"account {n}: configured {model} ranks={list(ranks)}; queue={st['n_queue']}")
    return cfg, payload


def snapshot_source():
    """Copy the exact scientific/orchestration source deployed for this cohort, with SHA-256."""
    import hashlib
    import shutil
    dst = EXEC / "source"
    files = sorted(list((ROOT / "frw").glob("*.py")) + list((ROOT / "tools").glob("frw_*.py"))
                   + [ROOT / "docs" / "PROTOCOL_V4.md", ROOT / "tests" / "test_frw.py"])
    manifest = {}
    for f in files:
        rel = f.relative_to(ROOT)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst / rel)
        manifest[str(rel)] = hashlib.sha256(f.read_bytes()).hexdigest()
    save(dst / "SHA256SUMS.json", manifest)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--accounts", default="1,2,3,4,5,6")
    ap.add_argument("--skip-deploy", action="store_true")
    ap.add_argument("--skip-prefetch", action="store_true")
    a = ap.parse_args(argv)
    rows, total = forecast()
    for r in rows:
        print(f"acct {r['account']}: {r['model']:14} ranks={r['ranks']} ~{r['gpu_hours']:.1f} GPU-h ~${r['usd']:.2f}")
    print(f"forecast total ~${total:.2f} (admission cap ${CAP_USD}/account, hard stop ${HARD_CAP_USD})")
    if not a.go:
        return
    EXEC.mkdir(parents=True, exist_ok=True)
    snapshot_source()
    save(EXEC / "launch-plan.json", {"plan": {n: {"model": m, "ranks": list(rk)} for n, (m, rk) in PLAN.items()},
                                      "forecast": rows, "gpu": GPU, "workers": WORKERS, "cap_usd": CAP_USD,
                                      "hard_cap_usd": HARD_CAP_USD, "deadline": DEADLINE.isoformat(),
                                      "job_seconds": JOB_SECONDS, "ts": time.time()})
    for n in [int(x) for x in a.accounts.split(",")]:
        model, ranks = PLAN[n]
        if not a.skip_deploy:
            deploy(n, GPU)
        if not a.skip_prefetch:
            log(f"account {n}: prefetch {model}")
            log(f"account {n}: prefetch -> {fn(n, 'prefetch').remote([model])}")
        configure(n, model, ranks)
        st = fn(n, "watchdog").remote()
        log(f"account {n}: watchdog kick -> {json.dumps(st)[:300]}")


if __name__ == "__main__":
    main()
