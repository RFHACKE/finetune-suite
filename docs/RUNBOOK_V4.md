# frw-v4 runbook (six-account Modal pool, LoRA only)

The design is in [PROTOCOL_V4.md](PROTOCOL_V4.md). This page is about operating it.

## What runs where

| Account | Workspace | Model | Ranks |
|---|---|---|---|
| 1 | abdullah-qureshi2503 | Granite-3.3-2B | 1, 2, 16, 32, 256, 512 |
| 2 | fasiraja099 | Granite-3.3-2B | 4, 8, 64, 128, 1024 |
| 3 | rajafasi098 | Qwen2.5-3B | 1, 2, 16, 32, 256, 512 |
| 4 | syedaliabbasnaq14 | Qwen2.5-3B | 4, 8, 64, 128, 1024 |
| 5 | syedshahg551214 | Qwen2.5-1.5B | all 11 |
| 6 | foziasharif098 | Llama-3.2-1B | all 11 |

Each workspace runs the deployed app `frw-v4` on L40S GPUs with five workers. A cron
`watchdog` runs every 10 minutes. It turns finished learning-rate searches into reported jobs,
respawns workers and enforces the spending cap, so **nothing depends on this computer**.

Money guards, per account. All of them run **inside the account**, so they hold with the PC off:

1. In-app ledger: a new job is admitted only if `estimated spend + in-flight reserve +
   1.25 × job estimate ≤ $26` (lowered from $27 at 21:34 UTC).
2. Real billing: every 10 minutes the watchdog reads the workspace's actual `metered_cost`. It
   uses the container's own identity; no tokens are stored anywhere. The value becomes a floor
   for the ledger. At **$29** the watchdog sets the halt flag, cancels the running workers and
   stops the app. The local supervisor repeats the same check whenever the PC is on.
3. A deadline of 2026-10-31 22:00 UTC keeps all spend inside October's credit, so November's
   $30 per account is left for you.
4. When an account has had nothing to run for ~25 minutes (finished, capped or past the
   deadline), its watchdog stops the app, so no idle cron keeps billing.

## If Claude Code stops (usage limit) or the PC sleeps or shuts down

- Training, LR selection, worker respawns, budget enforcement and self-shutdown all continue in
  the six Modal workspaces. Results are stored durably on each workspace's volume.
- The local `frw-supervisor` service only collects results, rebalances work between accounts
  and builds the analysis. After a sleep it continues where it left off. After a reboot, WSL must
  be started (open any Ubuntu/WSL terminal once); systemd then starts the service automatically.
  To catch up manually: `python3 -m tools.frw_supervisor collect` followed by
  `.venv/bin/python -m frw.analysis --root results-v4`.
- If the PC stays off until the end, nothing is lost. The final collection and analysis happen the
  next time the service runs.
- A new Claude Code session can resume from `docs/RUNBOOK_V4.md`, `results-v4/execution/`
  (status history, supervisor log) and `python3 -m tools.frw_supervisor status`.

For a provider-enforced stop, set a workspace budget of $30 in each dashboard
(Settings → Usage & Billing). The API cannot set this.

## Commands (run from the repository root)

```bash
python3 -m tools.frw_supervisor status                  # spend, queue, done, workers per account
python3 -m tools.frw_supervisor collect                 # pull finished runs into results-v4/runs/
python3 -m tools.frw_supervisor collect --adapters      # also pull final adapters (large)
python3 -m tools.frw_supervisor halt --account 3        # stop one account (flag + app stop)
python3 -m tools.frw_supervisor halt --account all      # emergency stop everything
.venv/bin/python -m frw.analysis --root results-v4      # regenerate JSON + figures + report
systemctl status frw-supervisor                         # local daemon (collect/billing/analysis loop)
journalctl -u frw-supervisor -f
```

To resume a halted account, redeploy it (`python3 -m tools.frw_launch --go --accounts N
--skip-prefetch`) and reset the flag with `control(N, "halt", {"halt": False})`. Queued work is
idempotent, so finished runs are never repeated.

## Outputs

- `results-v4/runs/<job_id>/`: `result.json` (accuracy/NLL/probe matrices, metrics, spectral
  summaries, timing, GPU), `train_log.jsonl.gz` (every optimizer step), `spectral.json.gz`
  (per-matrix diagnostics at each boundary), `pred_b<i>_<task>.npz` (predictions), and for main
  runs the final adapter and heads when collected with `--adapters`.
- `results-v4/accounts/<n>/`: config, status, LR selections, failures, worker logs.
- `results-v4/analysis/`: `runs.csv/json`, `lr_surface.json`, `selected_lr.json`,
  `main_summary.json`, `windows.json`, `paired_tests.json`, `trends.json`,
  `order_robustness.json`, `spectral_association.json`, `report.md`, and `figures/*.png`, each
  with a same-name `.json` holding the plotted numbers.
- `results-v4/execution/`: launch plan, supervisor log, status history, billing snapshots.

## Reproducing a single run anywhere

```python
from frw import train as T, jobs as J
import torch
job = J.Job("main", "qwen2.5-1.5b", "lora", 16, 2e-4, 42, "O1", "official")
ctx = T.Context(job.model, torch.device("cuda"))
T.run(job, ctx, Path("out") / job.job_id, probes=True, save_adapter=True)
```

Set `FRW_DATA_DIR=results-v4/data` to use the frozen rows (identical to the Hub pipeline).
Package versions: torch 2.8.0 (cu128), transformers 4.57.6, peft 0.17.1, datasets 4.4.1.
