---
name: modal
description: Run, deploy, monitor and cost-control GPU jobs on Modal (modal.com) for this project, including the six-account credit pool in modal.env. Use for any Modal task - deploying apps, launching or checking fine-tuning runs, reading volumes/logs, checking spend/billing, stopping apps, or debugging Modal errors.
---

# Modal for the feasible-rank-window project

Facts verified on 2026-10-25 against Modal SDK/CLI **1.6.2** (`/usr/local/bin/modal`, system `python3`)
and modal.com docs/pricing. Re-check `modal billing rates` before quoting prices.

## Accounts and credentials (never print tokens)

- `modal.env` holds six workspaces as `acN-token-id=` / `acN-token-secret=` (N=1..6). It is git-ignored.
  Workspaces: 1 abdullah-qureshi2503, 2 fasiraja099, 3 rajafasi098, 4 syedaliabbasnaq14,
  5 syedshahg551214, 6 foziasharif098.
- `~/.modal.toml` profile `rajafasi01010` is the user's **own paid account**, not part of the pool.
  Never launch work there unless the user explicitly asks.
- Per-account CLI: export the pair as `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` for that one command
  (`tools/frw_accounts.py:env_for(n)`); unset `MODAL_PROFILE`. In Python use
  `tools.frw_accounts.client(n)` (= `modal.Client.from_credentials(id, secret)`) and pass `client=` to
  `Function.from_name`, `Volume.from_name`, `Dict.from_name`, `Workspace.from_context`, etc.
- Hugging Face token: `hf.env`; it is installed in each workspace as the named secret `frw-hf`
  (`modal.Secret.objects.create("frw-hf", {"HF_TOKEN": ...}, client=c, allow_existing=True)`).
  Llama-3.2 is gated and needs it.

## Plan limits and money (Starter plan)

- $30 compute credit per workspace per **calendar month (UTC)**; usage beyond it is billed
  (pay-as-you-go, no automatic stop). The user's rule: **never exceed $30 per account**.
- 10 concurrent GPUs, 100 containers, 5 deployed crons, 1-day log retention, 1 TiB/month free volume
  storage and egress.
- Prices ($/h): T4 0.59, L4 0.80, A10(G) 1.10, L40S 1.95, A100-40GB 2.10, A100-80GB 2.50,
  RTX-PRO-6000 3.03, H100 3.95, H200 4.54, B200 6.25. CPU $0.0473/physical-core-h, memory
  $0.008/GiB-h (billed on max(request, usage); give `cpu=(req, limit)` / `memory=(req, limit)` to cap).
- Spend checks:
  `modal billing summary --json` (this month: `metered_cost`, `credits`, `billed_cost`),
  `modal billing report --for today -r h --json` (hourly, full intervals only, so it lags up to ~1 h),
  SDK: `modal.Workspace.from_context(client=c).billing.summary()` / `.report(start=..., resolution="h")`.
  There is **no API to set a workspace budget**; tell the user to set one in the dashboard
  (Settings -> Usage/Billing) if they want a provider-enforced cap.
- Because billing lags, guard spend in-app: every container writes a heartbeat ledger
  (start, last, $/s) to a `modal.Dict`; admit a new job only if
  `ledger + in-flight reserve + 1.25 x job estimate <= cap`. Keep a margin (cap $27, hard stop $29).

## Core API patterns (1.6.2)

```python
app = modal.App("name")
vol = modal.Volume.from_name("vol", create_if_missing=True)          # v1: <=5 concurrent writers
d = modal.Dict.from_name("state", create_if_missing=True)            # entries expire after 7 days idle
image = (modal.Image.debian_slim(python_version="3.11")
         .uv_pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
         .uv_pip_install("transformers==4.57.6", ...)
         .add_local_python_source("frw"))                             # deploy with `-m pkg.module`

@app.function(image=image, gpu="L4", cpu=(2, 2), memory=(12288, 16384), timeout=6*3600,
              volumes={"/vol": vol}, secrets=[modal.Secret.from_name("frw-hf")], retries=0)
def worker(slot: int): ...

@app.function(schedule=modal.Period(minutes=10), cpu=0.25, memory=512)
def watchdog(): worker.spawn(0)                                       # functions may spawn others
```

- Deploy: `cd repo && PYTHONPATH=. modal deploy -m frw.modal_app` (module mode so local packages
  resolve). Deployed apps and spawned calls keep running when the local machine is off.
- Call a deployed function: `modal.Function.from_name(app, fn, client=c)`; `.remote()` blocks,
  `.spawn()` returns a `FunctionCall` (`FunctionCall.from_id(id).get(timeout=0)` raises
  `TimeoutError` while running). `fn.with_options(gpu="H100")` overrides resources per call.
- `modal run --detach file.py` keeps an ephemeral app alive after the client exits.
- Atomic claim: `d.put(key, value, skip_if_exists=True)` returns `True` only for the first writer.
- Volumes: writes become visible elsewhere only after `vol.commit()` (also automatic in background
  and at exit); readers call `vol.reload()` (fails if the container holds open files on it).
  CLI: `modal volume ls|get|put|rm <vol> <path>`; `modal volume get` streams large files.
- Timeouts: default 300 s, max 24 h per attempt; `startup_timeout` separate. GPU functions are
  **preemptible** (restarted on the same input) and cannot be `nonpreemptible`; make jobs idempotent.
- GPU strings: T4, L4, A10 (A10G alias), L40S, A100-40GB, A100-80GB, RTX-PRO-6000, H100 (may run
  on H200 at H100 price; `H100!` pins), H200, B200, B300. Fallback lists: `gpu=["L40S", "A100-40GB"]`.

## Operations

- List/stop: `modal app list`, `modal app stop <app> --yes`, `modal app logs <app>`
  (live only; logs are kept 1 day on Starter, so the project also writes logs to the volume).
- Containers: `modal container list`, `modal container stop <id>`.
- Secrets/volumes/dicts: `modal secret list`, `modal volume list`, `modal dict list`.
- Stopping an app ends running containers immediately (in-flight jobs are lost; restart is safe).

## This project's orchestration (frw-v4)

- `frw/modal_app.py`: per-account app `frw-v4` (volume `frw-v4-results`, HF cache `frw-hf-cache`,
  dict `frw-v4-state`). Functions: `worker(slot)` GPU loop, `watchdog` (10-min cron: advance LR
  selection, respawn workers, enforce cap), `prefetch(models)`, `bench(model, ranks)`,
  `control(action, payload)` (`configure|enqueue|dequeue|halt|advance|reset_budget_flag`).
- `tools/frw_supervisor.py`: local multi-account supervisor (setup, status, billing, collect,
  rebalance, analysis). See `docs/PROTOCOL_V4.md` and `docs/RUNBOOK_V4.md`.
- Emergency stop for one account: `python3 -m tools.frw_supervisor halt --account N`
  (sets the halt flag, then `modal app stop frw-v4`). All accounts: `--account all`.

## Pitfalls seen in this project

- Billing summary is per calendar month; credits reset on the 1st (UTC). Track cumulative spend across
  the boundary yourself if work spans months.
- `modal deploy file.py` puts the file's directory on `sys.path`; use `-m package.module` from the
  repo root for `add_local_python_source`.
- Do not let several containers write the same Volume file; give each job its own directory.
- `datasets` cache locks on shared volumes can deadlock; freeze small datasets to JSON on the volume.
- The user's own account still has the old `fyp-unattended-preservation` deployment and ~90 GB of
  volumes from the v3 cohort; leave them unless the user asks.
