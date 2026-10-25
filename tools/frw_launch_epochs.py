"""Launch E3 (epoch budget, docs/PROTOCOL_V4_EXT.md) on the last October credit."""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frw import config as C  # noqa: E402
from frw import jobs as J  # noqa: E402
from tools.frw_launch import DEADLINE, JOB_SECONDS  # noqa: E402
from tools.frw_supervisor import EXEC, control, deploy, fn, log, save  # noqa: E402

S = C.EPOCH_SEEDS
PLAN = {1: ("llama3.2-1b", (1, 2, 3, 5), S[:5]), 2: ("llama3.2-1b", (1, 2, 3, 5), S[5:]),
        3: ("qwen2.5-1.5b", (1, 3, 5), S[:4]), 4: ("qwen2.5-1.5b", (1, 3, 5), S[4:8]),
        5: ("qwen2.5-1.5b", (1, 3, 5), S[8:])}
LR = {"llama3.2-1b": 1e-4, "qwen2.5-1.5b": 1e-4}

if __name__ == "__main__":
    save(EXEC / "launch-epochs-plan.json", {n: {"model": m, "epochs": e, "seeds": list(s)} for n, (m, e, s) in PLAN.items()})
    for n, (model, epochs, seeds) in PLAN.items():
        jobs = J.epoch_jobs(model, 16, LR[model], epochs, seeds)
        deploy(n, "L40S")
        log(f"account {n}: prefetch -> {fn(n, 'prefetch').remote([model])}")
        cfg = {"pairs": [], "cap_usd": 28.0, "hard_cap_usd": 29.0, "workers": 3, "deadline_unix": DEADLINE.timestamp(),
               "job_seconds": JOB_SECONDS, "orders": False, "gpu": "L40S", "protocol": C.PROTOCOL + "-ext-e3",
               "configured": dt.datetime.now(dt.timezone.utc).isoformat()}
        control(n, "configure", cfg)
        control(n, "unhalt")
        control(n, "reset_budget_flag")
        st = control(n, "enqueue", {"jobs": [{"spec": j.spec, "priority": j.priority} for j in jobs]})
        log(f"account {n}: E3 configured ({len(jobs)} jobs); queue={st['n_queue']}")
        log(f"account {n}: watchdog kick -> {json.dumps(fn(n, 'watchdog').remote())[:250]}")
