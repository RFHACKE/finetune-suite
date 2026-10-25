"""Local supervisor for the six-account frw-v4 deployment.

The remote side is autonomous (cron watchdog + workers + in-app spend ledger per
account). This process adds what one account cannot do alone: real billing
checks with a hard stop, result collection into one local tree, cross-account
rebalancing and the final analysis. It is safe to stop and restart at any time.

    python3 -m tools.frw_supervisor status
    python3 -m tools.frw_supervisor collect
    python3 -m tools.frw_supervisor halt --account 3      # or --account all
    python3 -m tools.frw_supervisor daemon --interval 900
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.frw_accounts import accounts, client, env_for  # noqa: E402

APP = "frw-v4"
OUT = ROOT / "results-v4"
EXEC = OUT / "execution"
HARD_STOP_USD = 29.0          # real metered spend at which an account is halted and its app stopped
ACCOUNT_LIMIT_USD = 30.0


def now_iso():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    EXEC.mkdir(parents=True, exist_ok=True)
    line = f"{now_iso()} {msg}"
    print(line, flush=True)
    with (EXEC / "supervisor.log").open("a") as fh:
        fh.write(line + "\n")


def save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def selected_accounts(arg):
    if arg in (None, "all"):
        return list(accounts())
    return [int(x) for x in str(arg).split(",")]


# ---------------------------------------------------------------------------
# remote helpers
# ---------------------------------------------------------------------------
def fn(n, name):
    import modal
    return modal.Function.from_name(APP, name, client=client(n))


def control(n, action, payload=None):
    return fn(n, "control").remote(action, payload)


def billing(n) -> dict:
    import modal
    ws = modal.Workspace.from_context(client=client(n))
    s = ws.billing.summary()
    return {"metered_usd": float(s.metered_cost), "billed_usd": float(s.billed_cost),
            "adjustments": {k: float(v) for k, v in s.adjustments.items()},
            "breakdown": {k: float(v) for k, v in s.metered_cost_breakdown.items()}}


def deploy(n, gpu, attempts=4):
    env = env_for(n)
    env.update(FRW_GPU=gpu, PYTHONPATH=str(ROOT))
    for attempt in range(1, attempts + 1):
        r = subprocess.run(["modal", "deploy", "-m", "frw.modal_app"], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=1800)
        if r.returncode == 0:
            log(f"account {n}: deployed {APP} gpu={gpu}")
            return
        log(f"account {n}: deploy attempt {attempt} failed: {(r.stdout + r.stderr)[-300:]!r}")
        time.sleep(30 * attempt)       # transient image-builder shutdowns happen; retry
    raise RuntimeError(f"deploy account {n} failed after {attempts} attempts")


def stop_app(n):
    r = subprocess.run(["modal", "app", "stop", APP, "--yes"], cwd=ROOT, env=env_for(n),
                       capture_output=True, text=True, timeout=300)
    log(f"account {n}: app stop rc={r.returncode} {r.stdout.strip()[-200:]} {r.stderr.strip()[-200:]}")


def halt(n, stop=True):
    try:
        control(n, "halt", {"halt": True})
    except Exception as exc:  # noqa: BLE001
        log(f"account {n}: halt flag failed: {exc}")
    if stop:
        stop_app(n)


# ---------------------------------------------------------------------------
# status / billing guard
# ---------------------------------------------------------------------------
def status(accts, guard=True) -> dict:
    out = {"ts": now_iso(), "accounts": {}}
    for n in accts:
        row = {}
        try:
            b = billing(n)
            row["billing"] = b
        except Exception as exc:  # noqa: BLE001
            row["billing_error"] = repr(exc)[:300]
            b = None
        try:
            c = control(n, "billing_floor", {"usd": b["metered_usd"]} if b else None)
            row["remote"] = {k: c[k] for k in ("status", "spend_estimate_usd", "halt", "budget_exhausted", "n_queue")}
            row["config_cap"] = (c.get("config") or {}).get("cap_usd")
        except Exception as exc:  # noqa: BLE001
            row["remote_error"] = repr(exc)[:300]
            row["remote"] = volume_state(n)       # app stopped/not deployed: read the volume directly
        if guard and b and b["metered_usd"] >= HARD_STOP_USD:
            log(f"account {n}: metered ${b['metered_usd']:.2f} >= hard stop ${HARD_STOP_USD}; halting")
            halt(n, stop=True)
            row["halted_by_supervisor"] = True
        out["accounts"][n] = row
    save(EXEC / "status-latest.json", out)
    with (EXEC / "status-history.jsonl").open("a") as fh:
        fh.write(json.dumps(out, default=str) + "\n")
    return out


def volume_state(n) -> dict:
    import modal
    vol = modal.Volume.from_name("frw-v4-results", client=client(n))
    out = {"status": None, "stopped": None, "halt": None, "budget_exhausted": None}
    for key, path in (("status", "frw-v4/status.json"), ("stopped", "frw-v4/stopped.json")):
        try:
            out[key] = json.loads(_read(vol, path))
        except Exception:  # noqa: BLE001
            pass
    if out["stopped"]:
        out["halt"] = out["stopped"].get("reason")
        if any(w in str(out["halt"]) for w in ("budget", "hard cap")):
            out["budget_exhausted"] = True
    return out


def volume_pending(n) -> list:
    """Reported jobs still pending on a stopped account, read from its volume (no app needed)."""
    import modal
    from tools.frw_launch import JOB_SECONDS, RATE_S
    vol = modal.Volume.from_name("frw-v4-results", client=client(n))
    done = {p.parent.name for p in (OUT / "runs").glob("*/result.json")}
    dead = {p.stem for p in (OUT / "accounts" / str(n) / "dead").glob("*.json")}
    rows = []
    for e in vol.listdir("frw-v4/queue"):
        job_id = Path(e.path).stem
        if job_id in done or job_id in dead:
            continue
        d = json.loads(_read(vol, e.path))
        spec = d["spec"]
        if spec["kind"] not in ("main", "order"):
            continue
        secs = JOB_SECONDS.get(f"{spec['model']}|{spec['split']}", 900) * 1.6
        rows.append({"job_id": job_id, "spec": spec, "priority": d["priority"], "est_usd": secs * RATE_S})
    return rows


def print_status(st):
    print(f"{'acct':>4} {'metered$':>9} {'est$':>7} {'queue':>6} {'done':>5} {'dead':>5} {'runnable':>8} "
          f"{'alive':>6} halt")
    tot_m = tot_d = tot_q = 0
    for n, row in st["accounts"].items():
        b = row.get("billing", {})
        r = row.get("remote", {})
        s = r.get("status") or {}
        tot_m += b.get("metered_usd", 0)
        tot_d += s.get("done", 0)
        tot_q += s.get("queued", 0)
        print(f"{n:>4} {b.get('metered_usd', float('nan')):>9.2f} {r.get('spend_estimate_usd', float('nan')):>7.2f} "
              f"{s.get('queued', '-'):>6} {s.get('done', '-'):>5} {s.get('dead', '-'):>5} {s.get('runnable', '-'):>8} "
              f"{str(s.get('alive_slots', '-')):>6} {r.get('halt') or s.get('halt_reason') or ''}")
    print(f"total metered ${tot_m:.2f}; jobs done {tot_d}/{tot_q}")


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------
SMALL = ("result.json", "train_log.jsonl.gz", "spectral.json.gz")


def _retry(fn, *args, tries=6):
    for i in range(tries):
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001
            if "rate limit" not in str(exc).lower() and "ResourceExhausted" not in type(exc).__name__:
                raise
            time.sleep(5 * (i + 1))
    return fn(*args)


def _read(vol, path) -> bytes:
    return _retry(lambda p: b"".join(vol.read_file(p)), path)


def _write_atomic(path: Path, data: bytes):
    import threading
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def collect(accts, adapters=False):
    import modal
    from concurrent.futures import ThreadPoolExecutor
    total_new = 0
    for n in accts:
        vol = modal.Volume.from_name("frw-v4-results", client=client(n))
        acct_dir = OUT / "accounts" / str(n)
        acct_dir.mkdir(parents=True, exist_ok=True)
        try:
            try:
                entries = _retry(lambda p: vol.listdir(p, recursive=True), "frw-v4")
            except Exception as exc:  # noqa: BLE001
                log(f"account {n}: listing failed {exc!r}")
                continue
            files = [e.path for e in entries if getattr(e, "type", None) is None or "FILE" in str(e.type)]
            runs, wanted = {}, []
            for path in files:
                parts = path.split("/")
                if len(parts) == 4 and parts[1] == "runs":
                    runs.setdefault(parts[2], {})[parts[3]] = path
                elif len(parts) == 3 and parts[1] in ("selections", "failures", "dead", "logs"):
                    dst = acct_dir / parts[1] / parts[2]
                    if parts[1] == "logs" or not dst.exists():
                        wanted.append((path, dst))
                elif len(parts) == 2 and parts[1] in ("config.json", "status.json", "prefetch.json", "stopped.json"):
                    wanted.append((path, acct_dir / parts[1]))
                elif len(parts) == 3 and parts[1] == "data" and not (OUT / "data" / parts[2]).exists():
                    wanted.append((path, OUT / "data" / parts[2]))
            jobs = []
            for job_id, names in runs.items():
                dst = OUT / "runs" / job_id
                if "result.json" not in names or ((dst / "result.json").exists() and not adapters):
                    continue
                jobs.append((job_id, names))

            def fetch_run(item):
                job_id, names = item
                dst = OUT / "runs" / job_id
                dst.mkdir(parents=True, exist_ok=True)
                for name, path in names.items():
                    if name == "result.json":
                        continue
                    if name.endswith((".npz", ".json", ".gz")) or (adapters and name.endswith((".safetensors", ".pt"))):
                        if not (dst / name).exists():
                            _write_atomic(dst / name, _read(vol, path))
                (dst / "account.txt").write_text(str(n))
                _write_atomic(dst / "result.json", _read(vol, names["result.json"]))   # last = complete marker
                return 1

            def fetch_file(item):
                path, dst = item
                dst.parent.mkdir(parents=True, exist_ok=True)
                _write_atomic(dst, _read(vol, path))
                return 1
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(fetch_file, wanted))
                new = sum(pool.map(fetch_run, jobs))
            total_new += new
            log(f"account {n}: collected {new} new runs ({len(runs)} run dirs on volume)")
        except Exception as exc:  # noqa: BLE001
            log(f"account {n}: collect error {exc!r}")
    return total_new


# ---------------------------------------------------------------------------
# rebalancing: reported jobs (selection already frozen) move from an account that
# hit its admission cap to the account with the most headroom
# ---------------------------------------------------------------------------
def rebalance(st, cap=26.0, margin=1.5):
    info = {}
    for n in st["accounts"]:
        try:
            info[n] = control(n, "pending")
        except Exception as exc:  # noqa: BLE001
            vs = (st["accounts"][n].get("remote") or {})
            if vs.get("stopped") and not vs.get("budget_exhausted"):
                info[n] = {"pending": [], "halt": True, "stopped": True,
                           "spend_estimate_usd": (vs.get("status") or {}).get("spend_estimate_usd", 0.0)}
            elif vs.get("stopped") and vs.get("budget_exhausted"):
                info[n] = {"pending": volume_pending(n), "budget_exhausted": True, "stopped": True}
            else:
                info[n] = {"error": repr(exc)[:200]}
    donors = [n for n, v in info.items() if v.get("budget_exhausted") and v.get("pending")]
    if not donors:
        return 0
    head = {}
    for n, v in info.items():
        if "error" in v or v.get("budget_exhausted"):
            continue
        b = (st["accounts"][n].get("billing") or {}).get("metered_usd", 0.0)
        own = sum(j["est_usd"] for j in v.get("pending", [])) * 1.25
        head[n] = cap - margin - max(v.get("spend_estimate_usd", 0.0), b) - own
    moved = 0
    for d in donors:
        for job in sorted(info[d]["pending"], key=lambda j: j["priority"]):
            if not head:
                break
            r = max(head, key=head.get)
            need = job["est_usd"] * 1.5
            if head[r] < need:
                log(f"rebalance: no headroom for {job['job_id']} (best account {r}: ${head[r]:.2f})")
                break
            try:
                if (info[r].get("halt")):
                    deploy(r, "L40S")
                    control(r, "unhalt")
                    info[r]["halt"] = False
                control(r, "enqueue", {"jobs": [{"spec": job["spec"], "priority": job["priority"]}]})
                if info[d].get("stopped"):
                    import modal
                    modal.Volume.from_name("frw-v4-results", client=client(d)).remove_file(
                        f"frw-v4/queue/{job['job_id']}.json")
                else:
                    control(d, "dequeue", {"job_ids": [job["job_id"]]})
                head[r] -= need
                moved += 1
                log(f"rebalance: moved {job['job_id']} from account {d} to {r}")
            except Exception as exc:  # noqa: BLE001
                log(f"rebalance error {exc!r}")
                break
    if moved:
        for r in head:
            try:
                fn(r, "watchdog").spawn()
            except Exception:  # noqa: BLE001
                pass
    return moved


# ---------------------------------------------------------------------------
# daemon
# ---------------------------------------------------------------------------
def all_finished(st) -> bool:
    for row in st["accounts"].values():
        s = (row.get("remote") or {}).get("status") or {}
        r = row.get("remote") or {}
        if r.get("stopped") or r.get("halt") or r.get("budget_exhausted") or s.get("halt_reason"):
            continue
        if not s or s.get("runnable", 1) > 0 or s.get("alive_slots"):
            return False
        if s.get("queued", 0) > s.get("done", 0) + s.get("dead", 0):
            return False
    return True


def run_analysis():
    py = ROOT / ".venv" / "bin" / "python"
    r = subprocess.run([str(py if py.exists() else sys.executable), "-m", "frw.analysis", "--root", str(OUT)],
                       cwd=ROOT, capture_output=True, text=True, timeout=7200)
    log(f"analysis rc={r.returncode} {r.stdout[-500:]} {r.stderr[-1500:]}")


def write_status_md(st):
    lines = [f"# frw-v4 live status ({now_iso()})", "",
             "| account | model | metered $ | cap $ | done / queued | runnable | workers | note |",
             "|---|---|---|---|---|---|---|---|"]
    plan = {}
    try:
        plan = json.loads((EXEC / "launch-plan.json").read_text())["plan"]
    except Exception:  # noqa: BLE001
        pass
    tot = 0.0
    for n, row in st["accounts"].items():
        b = (row.get("billing") or {}).get("metered_usd")
        tot += b or 0.0
        r = row.get("remote") or {}
        s_ = r.get("status") or {}
        note = r.get("halt") or s_.get("halt_reason") or ("budget cap reached" if r.get("budget_exhausted") else "")
        lines.append(f"| {n} | {plan.get(str(n), {}).get('model', '?')} | {b if b is None else round(b, 2)} | "
                     f"{row.get('config_cap')} | {s_.get('done', '?')} / {s_.get('queued', '?')} | "
                     f"{s_.get('runnable', '?')} | {len(s_.get('alive_slots') or [])} | {note or ''} |")
    lines += ["", f"Total metered across the pool: **${tot:.2f}** (limit $30 per account).", ""]
    sels = sorted((OUT / "accounts").glob("*/selections/*.json"))
    if sels:
        lines += ["## LR selections so far (training holdout, pre-registered rule)", "",
                  "| model | rank | selected LR | mean holdout ACC by LR |", "|---|---|---|---|"]
        for f in sels:
            d = json.loads(f.read_text())
            tab = ", ".join(f"{float(k):.1e}: {100 * v:.1f}" for k, v in sorted(d["table"].items(), key=lambda kv: float(kv[0])))
            lines.append(f"| {d['model']} | {d['rank']} | {d['selected_lr']:.1e} | {tab} |")
    done = len(list((OUT / "runs").glob("*/result.json")))
    lines += ["", f"Runs collected locally: {done}. Analysis: `results-v4/analysis/report.html` "
              "(refreshed every 2 h and at the end)."]
    (OUT / "STATUS.md").write_text("\n".join(lines) + "\n")


def daemon(interval, analyze_every):
    log(f"daemon start interval={interval}s")
    last_analysis = 0.0
    while True:
        try:
            st = status(list(accounts()))
            collect(list(accounts()))
            rebalance(st)
            write_status_md(st)
            if time.time() - last_analysis > analyze_every:
                run_analysis()
                last_analysis = time.time()
            if all_finished(st):
                collect(list(accounts()))
                run_analysis()
                save(EXEC / "finished.json", {"ts": now_iso(), "status": st})
                log("all accounts finished or stopped; final collection and analysis written")
                if not Path(EXEC / "keep-daemon").exists():
                    return
        except Exception as exc:  # noqa: BLE001
            log(f"daemon cycle error {exc!r}")
        time.sleep(interval)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status")
    s.add_argument("--account", default="all")
    s.add_argument("--no-guard", action="store_true")
    c = sub.add_parser("collect")
    c.add_argument("--account", default="all")
    c.add_argument("--adapters", action="store_true", help="also download final adapters/heads (large)")
    h = sub.add_parser("halt")
    h.add_argument("--account", required=True)
    h.add_argument("--no-stop", action="store_true", help="only set the halt flag; keep the app deployed")
    d = sub.add_parser("daemon")
    d.add_argument("--interval", type=int, default=900)
    d.add_argument("--analyze-every", type=int, default=3 * 3600)
    sub.add_parser("analyze")
    a = parser_args = p.parse_args(argv)
    if a.cmd == "status":
        print_status(status(selected_accounts(a.account), guard=not a.no_guard))
    elif a.cmd == "collect":
        collect(selected_accounts(a.account), adapters=a.adapters)
    elif a.cmd == "halt":
        for n in selected_accounts(a.account):
            halt(n, stop=not a.no_stop)
    elif a.cmd == "daemon":
        daemon(a.interval, a.analyze_every)
    elif a.cmd == "analyze":
        run_analysis()
    return parser_args


if __name__ == "__main__":
    main()
