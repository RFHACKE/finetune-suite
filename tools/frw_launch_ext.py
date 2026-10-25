"""Launch the pre-registered extension cohorts (docs/PROTOCOL_V4_EXT.md) on the remaining October credit.

    python3 -m tools.frw_launch_ext --plan
    python3 -m tools.frw_launch_ext --go
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frw import config as C  # noqa: E402
from frw import jobs as J  # noqa: E402
from tools.frw_launch import DEADLINE, JOB_SECONDS, RATE_S, snapshot_source  # noqa: E402
from tools.frw_supervisor import EXEC, OUT, billing, control, deploy, fn, log, save  # noqa: E402

GPU = "L40S"
WORKERS = 5
CAP_USD = 27.0
HARD_CAP_USD = 29.0
STAB = {1: ("granite3.3-2b", (512, 1024)), 2: ("granite3.3-2b", (16, 64, 128)), 5: ("granite3.3-2b", (256,)),
        3: ("qwen2.5-3b", (16, 64, 128, 256)), 6: ("qwen2.5-3b", (512, 1024))}
AL = {4: "qwen2.5-1.5b", 6: "llama3.2-1b"}


def selected_lrs():
    out = {}
    for f in (OUT / "accounts").glob("*/selections/*.json"):
        d = json.loads(f.read_text())
        if not d.get("suffix"):
            out[(d["model"], d["rank"])] = d["selected_lr"]
    return out


def plan():
    sel = selected_lrs()
    per = {}
    for n, (model, ranks) in STAB.items():
        for r in ranks:
            per.setdefault(n, {"jobs": [], "pairs": [], "models": set()})
            per[n]["jobs"] += J.stab_jobs(model, r, sel[(model, r)])
            per[n]["models"].add(model)
    for n, model in AL.items():
        per.setdefault(n, {"jobs": [], "pairs": [], "models": set()})
        per[n]["jobs"] += J.tune_jobs(model, ranks=C.AL_RANKS, suffix="_al")
        per[n]["pairs"] += [{"model": model, "rank": r, "suffix": "_al"} for r in C.AL_RANKS]
        per[n]["models"].add(model)
    return per


def estimate(n, p):
    secs = 0.0
    for j in p["jobs"]:
        s = JOB_SECONDS[f"{j.model}|{j.split}"] * (0.85 if j.kind == "stab" else 1.6)
        secs += s * (1.15 if (j.rank or 0) >= 512 else 1.0)
    for pair in p["pairs"]:                      # reported runs created after selection
        secs += 5 * JOB_SECONDS[f"{pair['model']}|official"] * 1.6 + 1.0 * JOB_SECONDS[f"{pair['model']}|holdout"] * 1.6
    return secs * RATE_S * 1.05


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--accounts", default="")
    a = ap.parse_args(argv)
    per = plan()
    if a.accounts:
        keep = {int(x) for x in a.accounts.split(",")}
        per = {n: p for n, p in per.items() if n in keep}
    rows = []
    for n, p in sorted(per.items()):
        spent = billing(n)["metered_usd"]
        est = estimate(n, p)
        rows.append({"account": n, "models": sorted(p["models"]), "queued_now": len(p["jobs"]),
                     "pairs": len(p["pairs"]), "spent": spent, "estimate": est, "after": spent + est})
        print(f"acct {n}: {sorted(p['models'])} queued={len(p['jobs'])} pairs={len(p['pairs'])} "
              f"spent ${spent:.2f} + est ${est:.2f} = ${spent + est:.2f} (cap ${CAP_USD})")
    if not a.go:
        return
    snapshot_source()
    save(EXEC / "launch-ext-plan.json", {"rows": rows, "protocol": "docs/PROTOCOL_V4_EXT.md",
                                          "ts": dt.datetime.now(dt.timezone.utc).isoformat()})
    for n, p in sorted(per.items()):
        deploy(n, GPU)
        log(f"account {n}: prefetch -> {fn(n, 'prefetch').remote(sorted(p['models']))}")
        cfg = {"pairs": p["pairs"], "cap_usd": CAP_USD, "hard_cap_usd": HARD_CAP_USD, "workers": WORKERS,
               "deadline_unix": DEADLINE.timestamp(), "job_seconds": JOB_SECONDS, "orders": False, "gpu": GPU,
               "protocol": C.PROTOCOL + "-ext", "configured": dt.datetime.now(dt.timezone.utc).isoformat()}
        control(n, "configure", cfg)
        control(n, "unhalt")
        control(n, "reset_budget_flag")
        st = control(n, "enqueue", {"jobs": [{"spec": j.spec, "priority": j.priority} for j in p["jobs"]]})
        log(f"account {n}: extension configured; queue={st['n_queue']}")
        log(f"account {n}: watchdog kick -> {json.dumps(fn(n, 'watchdog').remote())[:300]}")


if __name__ == "__main__":
    main()
