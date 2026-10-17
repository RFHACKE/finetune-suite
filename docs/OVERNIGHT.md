# Unattended Modal execution and morning recovery

The running cohort is `frw-20261018` in Modal workspace `rajafasi01010`. It contains four fixed models × seven LoRA ranks (4, 8, 16, 32, 64, 128, 256) × five seeds (42, 123, 456, 789, 1024): **140 sequences**. The 48-case LR/head calibration has already finished. Scientific training controls and the selected defaults remain frozen.

Training runs under a detached remote CPU coordinator. Closing this conversation, sleeping the local computer, losing the local connection, or exhausting Codex quota does not stop those remote workers. GPU containers exit when their model's assigned runs finish. The existing coordinator retains the $28 aggregate main-stage guard and six-hour worker timeouts; calibration had a separate $6 allowance. The approved overall spending envelope is $35, including preparation and CPU services. These guards are conservative resource estimates, rather than an account-wide billing lock. Billing delays, resource prices and continuing storage charges still matter.

CPU spectral analysis and final scientific analysis are **deferred until the user returns**. Previously computed local spectral caches are retained. There are no automatic training retries and no external messages. The temporary archive download endpoint was stopped before leaving the experiments unattended.

## Durable remote data

| Modal volume | Directory | Contents |
| --- | --- | --- |
| `fyp-finetune-results` | `/frw-20261018/main/<model>/` | Completed runs, partial task-boundary recovery checkpoints, per-step logs, failure tracebacks, protocol/software/hardware/data records, pretrained q/v projection cache. |
| `fyp-finetune-results` | `/frw-20261018/defaults/` | Activated LR/head defaults, their calibration reports and charts. |
| `fyp-finetune-results` | `/frw-20261018/calibration/<model>/` | Original calibration cases, step logs, selections and charts. |
| `fyp-finetune-results` | `/frw-20261018/execution/main-workers.json` | Training call IDs, worker returns, estimated main spending and any budget stop reason. |
| `fyp-finetune-results` | `/frw-20261018/execution/unattended/` | Remote heartbeat, exact 140-run inventory, failure records, retry candidates, frozen source and checksummed source manifest. |
| `fyp-finetune-archive` | `/frw-20261018/archives/<model>/` | Lossless completed-run archives and independent SHA-256 manifests; the first archive per model includes its pretrained projections. |
| `fyp-pinned-model-cache` | `/huggingface/` | Pinned model/tokenizer weights and cached data for later recovery. |

At each completed task boundary and run exit, the training wrapper copies its private working directory to durable storage and commits it. A separate CPU-only packager checks exported sizes and projection SHA-256 values, then saves lossless archives on the archive volume. A sudden GPU/container failure can lose the current unfinished task; the latest committed task boundary remains available. Incomplete exports are identified and retained for later recovery.

Raw exports include the final adapter and classification heads at **each task boundary**, projection factors, all saved validation predictions, accuracy/NLL matrices, optimizer-step logs and selected defaults. Together with the saved pretrained projections and frozen source, these are sufficient for the suite's later CPU spectral, update-rank, subspace and aggregate analyses. Full model weights need not be downloaded to perform those analyses.

The archive is a second copy within the same Modal account, rather than an independent off-platform backup. Volumes persist after compute stops, but deleting the volumes or losing account access loses that protection. Storage can continue to incur charges; keep both volumes until a verified local copy is complete.

## Read the overnight report first

Use the system Python containing Modal, with the same authenticated profile:

```bash
python3 tools/modal_unattended.py status
```

This downloads only small status files to `Finetune-result/execution/unattended/`; it starts no cloud compute. Check:

- `status.json`: heartbeat time, completed/archived counts, model counts, worker outcomes and cost guard evidence.
- `runs.csv`: all 140 model/rank/seed combinations, state, last saved optimizer step and recovery checkpoint availability.
- `failures.json`: per-run exceptions with tracebacks, fatal worker errors and artifact-preservation issues.
- `retry-plan.json`: missing, failed or invalid identities to review later. It does not execute retries.
- `source-manifest.json` and `source/`: checksummed frozen scientific code, requirements, tools and the original main runner snapshot when available.

`running` means workers or packaging remain active. `complete` means all 140 sequences have their expected exports/step logs and packaged copies; it does **not** assert that deferred numerical/spectral audits have passed. `finished_with_issues` lists any incomplete runs or missing archives explicitly. `observer_timeout` means the observer reached its own time limit; it does not cancel training. A stale heartbeat or `observer-exit.json` warrants checking `main-workers.json` and the Modal app logs directly. Raw GPU exports remain independent of the observer.

The observer also polls the existing GPU call IDs so a worker that exits before writing its own error log remains identifiable. Its loop records its own errors and retries **observation only**. It never launches, retunes or retries a training experiment. Pending runs are distinguished from failed runs while workers remain active. A recorded task checkpoint should be checked against its corresponding export files before attempting recovery.

## Retrieve results after returning

```bash
# Requires Modal authentication and the local zstd CLI; starts no cloud compute.
python3 tools/collect_modal_results.py --prefer-archives --workers 4
# If training/packaging is still running, use --watch to keep collecting:
python3 tools/collect_modal_results.py --prefer-archives --workers 4 --watch
```

The collector verifies the downloaded archive SHA-256 and saved projection hashes before marking runs ready. It preserves local completed spectral results and brings down failure logs, provenance and the overnight reports too. Interrupted authenticated HTTP downloads from the previous session remain as ignored `.download` files; normal SDK collection can replace them safely.

Ordinary authenticated SDK downloads require no public endpoint. If their throughput is inadequate, the previously authorized temporary token-protected stream can be redeployed using [MODAL_EXECUTION.md](MODAL_EXECUTION.md). Stop it again after retrieval. Starting a stream is optional and is not part of the unattended observer.

After verified retrieval, use the suite CPU environment:

```bash
./setup.sh --cpu
./run.sh intruders --output Finetune-result --threads 4
python -m tools.audit_requested_results --root Finetune-result
# Only after all 140 runs and their CPU spectra pass verification:
python -m tools.finalize_collected_results --root Finetune-result
```

Use the same activated Python environment for the last two commands, for example `.venv/bin/python`. Spectral analysis resumes its existing per-matrix cache. The finalizer refuses to produce a complete final delivery when the exact requested cohort, exported evidence, source checks or spectra are incomplete. It then generates aggregate tables, charts, a report and a complete local artifact checksum manifest. The requested LoRA-only cohort cannot establish full-FT-baseline-dependent feasible windows.

If any sequences failed, retain all successful results and review `retry-plan.json` and the original worker logs before allocating more GPU time. Recovery must use the same source, protocol, model/data revisions and activated LR/head policy. Completed identities should be skipped and the last valid task checkpoint reused where possible. No recovery run is launched overnight.

## Observer administration and every CLI option

The observer is deployed as a private SDK function, with no HTTP endpoint, GPU or Hugging Face secret. The following setup is already performed for this execution; **do not start a duplicate**:

```bash
modal deploy tools/modal_unattended.py
python3 tools/modal_unattended.py start
```

The `start` process saves `Finetune-result/execution/unattended-call.json` and exits immediately. The deployed function continues independently and stops its CPU container after training/packaging finishes or its own time limit expires. The idle deployment has no warm containers or scheduled invocations. Its bounded, small CPU cost remains part of the $35 envelope.

| Argument/option | Default | Meaning |
| --- | --- | --- |
| `start` | Required command choice | Spawn the already-deployed CPU observer once; refuses a duplicate recorded for this cohort. Never starts GPU work. |
| `status` | Required command choice | Retrieve small durable status/failure/CSV files and print a summary; no cloud compute allocation. |
| `--run` | `FYP_MODAL_RUN`, otherwise `frw-20261018` | Single remote cohort directory name, used by both commands. |
| `--root` | `Finetune-result` | Local results root for call records/status downloads and original launch provenance. |
| `--interval` | `60` | `start`: seconds between remote observations; minimum 15. |
| `--max-seconds` | `25200` | `start`: observer lifetime, seven hours by default; allowed range 60–28200. Training has its own independent limits. |
| `--archive-grace` | `600` | `start`: maximum additional seconds waiting for missing archives after all workers return. Raw exports stay preserved if packaging fails. |
| `-h`, `--help` | Off | Show all commands/options and exit. |

The same flags are parsed for both commands; timing flags affect only `start`. Environment variable `FYP_MODAL_RUN` also supplies the default directory for the training/packaging tools. The deployed observer receives its actual directory as an explicit call argument. Do not delete the saved call record simply to bypass duplicate protection; check the existing remote call/status first.
