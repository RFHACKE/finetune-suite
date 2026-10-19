# Requested Modal execution

The requested cohort fixes the four methodology checkpoints and expands the LoRA ranks to **4, 8, 16, 32, 64, 128, 256**. Five seeds (42, 123, 456, 789, 1024) give **140 complete RTE → MNLI → SST-2 sequences**. Each main task retains 2,490 training examples, three epochs and its complete official validation split. There is no full-FT baseline in this requested execution; baseline-dependent feasible windows cannot be certified from this cohort alone.

Its completed LoRA calibration selected LRs `4.00e-4 / 1.85e-4 / 3.20e-4 / 3.00e-4` with head multipliers `5 / 1 / 5 / 5` in registry order. These are now [project defaults for fresh outputs](LORA_DEFAULTS.md). Existing cloud images and result fingerprints retain the original frozen source; follow [OVERNIGHT.md](OVERNIGHT.md) to use its pinned checkout for this cohort's analysis or recovery. Do not redeploy or resume this cohort from the changed scientific source.

The [Modal runner](../tools/modal_experiments.py) uses the pinned package environment with CUDA 12.8, SDPA and bf16 computation. Trainable adapter/head parameters and AdamW moments remain fp32. A micro-batch of 16 and accumulation of one preserve effective batch 16, with the partition recorded consistently in calibration and main results. No quantization, reduced main validation, reduced task data, shortened epochs or substituted checkpoints are used to save credit.

The training-only LR/head calibration is the procedure in [PROTOCOL_V3.md](PROTOCOL_V3.md), restricted to LoRA: at most 48 cases and 30,000 optimizer updates. Its endpoint checks follow the supplied rank range, here 4 and 256. The default CLI's original range still checks 4 and 64. The selected rate is the best eligible tested policy under the declared selection objective, rather than a claim of a universal optimum for every rank or task.

CPU preparation downloads all pinned weights and tokenizes the training/validation data before GPU allocation. A separate performance benchmark uses 192 training examples and one epoch at ranks 4 and 256; it is excluded from LR selection and main scientific results. GPU type and time allowances are chosen after measuring that benchmark and checking current Modal rates. The user approved **$35 total**, including preparation, after a measured $31–$35 forecast. The existing aggregate stage caps remain $6 for calibration and $28 for the main sweep, leaving the balance for preparation and CPU packaging. Worker timeouts and whole-run admission limits also bound spending. Incomplete work is retained and reported explicitly if that envelope cannot cover the requested design.

The selected GPU is L40S: the measured rank-256 benchmark was slower than H100 but cheaper per complete sequence. Four workers run concurrently, one per fixed model. Main workers request two CPU cores and 8 GiB host RAM, capped at two cores and 16 GiB; calibration was launched with a 16 GiB request and 24 GiB cap. These resource limits and their respective conservative billing rates are recorded separately in the execution evidence. They do not alter the frozen numerical training controls.

Only `rank_suite/`, `requirements.txt` and the Modal runner enter the image. `hf.env` stays local; its token is supplied through an ephemeral Modal secret. `notify.env` is not used. No messaging service is invoked. Model cache and scientific results live on separate durable Modal volumes. Every worker has a private local directory for Linux locks; incremental copies preserve task-boundary recovery checkpoints and completed results on the output volume.

Staged commands, run from the repository root with Modal already authenticated:

```bash
modal run --detach tools/modal_experiments.py --stage prefetch
modal run --detach tools/modal_experiments.py --stage benchmark --seconds 600
# Supply measured, budgeted seconds per model; four workers run concurrently.
modal run --detach tools/modal_experiments.py --stage calibration --gpu L40S --seconds 3600 --stage-budget 6
modal run --detach tools/modal_experiments.py --stage merge-defaults
modal run --detach tools/modal_experiments.py --stage main --gpu L40S --seconds 21600 --stage-budget 28
modal run --detach tools/modal_experiments.py --stage consolidate
```

`--models` accepts a comma-separated subset of fixed registry keys; `--gpu` defaults to H100 and the budgeted choices are H100/L40S. `--seconds` bounds each GPU input; `--stage-budget` is the aggregate USD allowance across all workers and is required for calibration/main, with a hard maximum of $28 per stage. A small CPU coordinator polls completion and conservatively counts queue/startup/execution time at the hard-capped resource rate, terminating pending work before its stage allowance is exhausted. Review cumulative costs between stages to stay within the total $35 authorization. `--stage monitor-calibration --stage-budget 6` attaches a new CPU monitor to saved calibration call IDs without starting GPU work. `FYP_MODAL_RUN` selects the remote run directory and defaults to `frw-20261018`. Stages do not automatically advance; their completed evidence and measured costs must be reviewed before admitting the next stage. Local call identifiers and runner snapshots are saved under `Finetune-result/execution/`. The Modal SDK `FunctionCall.from_id(id).get(timeout=0)` can check them after a client interruption; a pending call raises built-in `TimeoutError`.

Merged defaults retain the original activated calibration reports for all four models. All main workers load that same policy before freezing their protocol. Consolidation verifies common experimental controls and writes a separate `requested-completeness.json` for the 140-sequence requested scope; the original suite audit still reports the absent full-FT design honestly. Retrieve the run volume into `Finetune-result`, retaining raw predictions, step logs, adapters, heads, projection factors, calibration JSON and charts. CPU spectral analysis and aggregate figures can then use the saved tensors without renting a GPU.

For the current unattended execution, follow [OVERNIGHT.md](OVERNIGHT.md): durable training and packaging continue remotely; local collection, CPU spectral work and final analysis are deferred until the user returns. A private CPU observer preserves an exact run inventory, failure tracebacks, retry candidates and the frozen source remotely. No training retries are automatic.

The CPU packager can run during the sweep. Collection and spectral analysis below are manual steps for when the user returns:

```bash
modal run --detach tools/modal_archive.py
python3 tools/collect_modal_results.py --watch --prefer-archives
# In a separate process for each model, using the suite's Python environment:
python -m tools.analyze_collected_runs --model llama3.2-1b --threads 2 --watch
```

The packager reads completed checkpoints from `fyp-finetune-results` and writes lossless `tar.zst` files to the separate `fyp-finetune-archive` volume. It verifies exported file sizes and projection checksums before packaging. The collector checks the compressed archive SHA-256, extracts every saved scientific export, and verifies the pretrained and adapted projection checksums before marking a run ready for local CPU analysis. Compression removes repeated bytes without changing the exported tensors. Obsolete recovery-only `resume.pt` files and advisory locks stay in the cloud; final adapters, heads, projections, predictions and logs are retrieved. Downloaded archives are removed after successful extraction to avoid storing a second full copy.

Collector options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--target` | `Finetune-result` | Local unified result directory. |
| `--run` | `frw-20261018` | Remote experiment directory. Use a new local target for a different cohort. |
| `--volume` | `fyp-finetune-results` | Primary scientific result volume. |
| `--workers` | `8` | Concurrent downloads; must be positive. |
| `--watch` | Off | Repeat until all training workers return and completed results are verified. |
| `--interval` | `60` | Seconds between collection cycles; minimum five. |
| `--prefer-archives` | Off | Retrieve CPU-packaged lossless archives instead of individual main-run files; requires the local `zstd` executable. |
| `--archive-volume` | `fyp-finetune-archive` | Volume containing the packaged archives. |
| `--archive-url` | Empty | Optional explicitly authorized HTTPS Modal endpoint for streaming archives; requires `--prefer-archives` and curl. Omit it to use normal authenticated SDK downloads. |
| `--access-file` | `.modal-state/stream-token` | Local credential file for that optional endpoint; its value is never printed or passed in process arguments. |

Each completed archive is extracted and published for local analysis immediately, even while another download is still pending. Every archive must match the cloud packager's size and SHA-256 before extraction.

For this execution, the user explicitly approved a temporary, externally reachable, token-protected Modal streaming endpoint after ordinary downloads stalled. Deploy [modal_stream.py](../tools/modal_stream.py) only with comparable authorization. It mounts only the archive volume, serves only this run's `archives/` files, checks a random bearer token before reading any file, and supports HTTP range requests for resumption. It has no GPU, no Hugging Face credential, and no directory listing. The token is generated in ignored `.modal-state/stream-token`, supplied as an ephemeral Modal secret, and placed in private temporary curl configuration files during downloads. Authentication rejection and a full cloud archive checksum are tested before use.

```bash
modal deploy tools/modal_stream.py
FYP_ARCHIVE_URL=$(python3 -c 'import modal; print(modal.Function.from_name("fyp-archive-stream", "stream_archives").get_web_url())')
python3 tools/collect_modal_results.py --watch --prefer-archives --archive-url "$FYP_ARCHIVE_URL"
# Stop the temporary endpoint after successful collection:
modal app stop --yes fyp-archive-stream
```

The optional stream changes transport only. It retains the independent cloud archive SHA-256 and all projection checks. Cumulative CPU transfer charges remain part of the approved total. Normal SDK collection remains available without deploying an endpoint.

The local analysis helper accepts `--root` (default `Finetune-result`), required `--model` (one fixed registry key), `--threads` (default 2), `--watch` (off by default), and `--interval` (default 60 seconds, minimum effective five). It processes only verified runs that still lack completed spectral diagnostics. Its temporary directory contains symlinks to the actual saved exports; the suite's unchanged analysis code writes results into the real run directories. Run one helper per model to overlap CPU analysis with GPU work. Neither collection nor local analysis starts cloud compute. The CPU packager has no CLI options; `FYP_MODAL_RUN` selects its run directory.

After retrieval and spectral analysis finish, run `python -m tools.audit_requested_results --root Finetune-result`. Its only option is `--root`, with that same default. The audit writes `requested-completeness.json`, checking the exact 140 model/rank/seed combinations, shared controls, selected rates, data cardinalities, full optimizer logs, checkpoint sizes/checksums, saved predictions against reported accuracies, and completed spectral matrix counts. This requested-scope audit complements the suite's original full-FT-dependent completeness audit.

## Finished-cohort segmented delivery

For the finished sweep, `tools/modal_delivery.py` checks the exact preserved
grid, all worker protocols/software and all archive sizes before producing one
metadata bundle and `execution/delivery-catalog.json`. This avoids thousands
of small remote reads. It accepts `--target` (default `Finetune-result`) for
the local destination and `--inline-metadata` (off) to return and extract the
checked metadata through authenticated Modal RPC. `FYP_MODAL_RUN` selects the
remote cohort; use a separate target for each cohort. Common Modal `--profile`,
`-h` and `--help` options are also available. It starts only CPU work.

```bash
modal run --detach tools/modal_delivery.py
python3 -m tools.collect_delivery --archive-url "$FYP_ARCHIVE_URL"
```

The segmented collector requires `zstd`, `aria2c`, a finished catalog, and the
previously authorized authenticated endpoint. Install the native utilities
with the distribution's package manager, or pass the workspace-local aria2
executable. Its complete options are:

| Option | Default | Meaning |
|---|---|---|
| `--catalog` | Saved path, otherwise `TARGET/execution/delivery-catalog.json` | Checked finished-cohort archive catalog. |
| `--target` | `Finetune-result` | Local cohort root; incompatible protocols are rejected. |
| `--archive-volume` | Saved value, otherwise `fyp-finetune-archive` | Named archive volume used by the shared verification helper. |
| `--archive-url` | Saved value | Explicitly authorized HTTPS bearer-protected stream endpoint; supply once for a fresh transfer. Completed targets need no endpoint. |
| `--access-file` | Saved path, otherwise `.modal-state/stream-token` | Private token file; credentials never appear in process arguments. |
| `--aria2` | Saved path, otherwise `.bootstrap/aria2/usr/bin/aria2c` | Native executable; use an absolute system path if installed elsewhere. |
| `--workers` | Saved value, otherwise `16` | Positive concurrent archive count. |
| `--connections` | Saved value, otherwise `16` | HTTP range connections per archive, from 1 through 16. |
| `-h`, `--help` | — | Show help. |

The US streaming function permits up to eight CPU containers with four concurrent
requests each. Each container requests 1 GiB RAM with a 2 GiB cap and a 24-hour
function timeout. The previous 256 MiB request/1 GiB cap with 16 concurrent
requests suffered repeated memory-related container terminations on the larger
archives. Native `.aria2` control files and partial data support resumption.
Resumption is the default. Keep the entire result directory, especially
`execution/retrieval.json`, `execution/incoming/*.download`, and their `.aria2`
files. Verified runs are skipped, complete saved archives are checksum-checked
and reused, and partial archives continue from saved segments. Native controls
are saved every ten seconds. Ctrl+C or SIGTERM asks active native downloads to
save their controls and stop; queued downloads are not started. A per-target
lock and active-process check reject duplicate downloaders.

Endpoint settings and credential **paths**, not credential values, are saved
in `execution/download-settings.json`. From the repository root, repeat:

```bash
python3 -m tools.collect_delivery --target Finetune-result/large-ranks
```

This reuses the saved catalog, endpoint and transfer options while appending
to the existing progress log. Explicit options override the saved values.
The endpoint must still be deployed. For the 512/1024 cohort, the preferred
restart command also redeploys its endpoint when needed, monitors spending,
starts local CPU analysis as verified data arrives, and stops the endpoint:

```bash
./start-download.sh
```

It uses `Finetune-result/large-ranks` unless `--root` is supplied, reuses the
existing catalog and metadata, and starts no GPU work. Both commands report
an existing active downloader rather than starting a second transfer.
The shell launcher selects Python with the Modal SDK installed and invokes
`tools.watch_large_ranks --resume-download`. `MODAL_PYTHON` overrides its
interpreter selection; additional arguments are forwarded to the watcher.
When systemd is available, the launcher starts a background service that survives
the shell and Codex session. Root uses a system service; other accounts use their
user manager when available. The default service is `fyp-result-download`.
Unexpected supervisor failures restart after 30 seconds, with at most three
starts per hour. It is a transient service: rerun the launcher after a machine
reboot. `--foreground` runs in the current terminal instead; hosts without a
working systemd manager also use foreground mode. `--help` documents this switch.

For the default system service, inspect or stop it with:

```bash
systemctl status fyp-result-download
systemctl stop fyp-result-download
```

Use `systemctl --user` for a user service. Custom `--root` targets receive a
service-name suffix printed by the launcher. `execution/download-supervisor.log`
retains stdout and tracebacks; `execution/download-supervisor.json` records the
PID, phase, update time, child PID, attempt, bytes and last progress. Its heartbeat
updates approximately every 30 seconds. A dead PID or stale heartbeat signals a
failed supervisor even if the progress log's last entry showed normal transfer.
The supervisor uses four archive workers and 8/4/2 connections per archive across
its three transport attempts. After fifteen minutes without new archive bytes,
it saves partial controls, stops that attempt and retries. The account spending
guard and endpoint cleanup apply to every attempt.
Billing reads time out after thirty seconds. A temporary failure may reuse a
checked reading for up to fifteen minutes if more than $0.50 remains below the
approved ceiling, retrying billing every thirty seconds. Stale or insufficient
headroom pauses collection; a live reading retains the $0.25 stop margin.
`execution/download-billing-status.json` and `execution/download-billing-errors.jsonl`
record those checks. Exhausting three transport attempts returns an error,
allowing the existing bounded systemd restart policy to resume saved partials.
An already completed target exits without any network downloads and retains
its completed state. If an interrupted transfer's token file or endpoint is
unavailable, restore that access before resuming; preserve the partial files.

The segmented downloader automatically appends timestamped progress to
`TARGET/execution/download-progress.log` at startup, every ten minutes, and on
successful completion. For example:

```text
2026-10-18T06:00:00Z  7.80 GB / 21.64 GB  (36.0%)  verified=80/140
```

Timestamps use UTC (`Z`), and GB means decimal gigabytes. Progress counts the
catalog's compressed experiment archives, including extracted/deleted archives
and an estimate of partial transfers from allocated filesystem blocks. Sparse
files' final logical lengths do not count as downloaded bytes. Metadata and
unrelated files are excluded; verified archive counts are shown separately.
Each cohort writes its own log. Entries append across restarts; the final
completion entry can occur before the next ten-minute interval. A normally
interrupted or failed collector also writes an immediate `status=stopped` entry.

To attach the same logger to an already running download without restarting it:

```bash
python3 -m tools.download_progress --target Finetune-result
```

This local observer accepts `--target` (default `Finetune-result`), `--catalog`
(default `TARGET/execution/delivery-catalog.json`), and `-h`/`--help`. It logs
immediately, then every 600 seconds, and exits after retrieval finishes. A file
lock prevents duplicate writers. It uses no cloud compute and requires the
computer to stay awake. Logger read errors go to
`TARGET/execution/download-progress-errors.log`.
Failed native transfers retain their diagnostics beside the partial archive
as `*.download.error.log`, with bearer tokens redacted. Native progress bars
are disabled; errors remain enabled so a transfer failure is distinguishable
from normal partial progress. A run enters the `verified` count only after
its complete archive and the required local reference/adapted projections
pass their checksum checks.

Each compressed archive must pass its independent cloud size and SHA-256;
extracted projections and pretrained caches must also pass their checksums.
An archive that arrives before its model's base cache waits for that dependency.
Only verified runs are eligible for CPU spectral analysis. Metadata alone does
not count as retrieved raw training data. `retrieval_finished` becomes true
only after every requested archive is verified; the CPU watch helper waits
for that condition rather than stopping when GPU training ends.

CPU analysis must import the source recorded in the cohort's protocol. The
original sweep used scientific source hash `ff3d682687b8ca5db927a383539cf816691c06a647c96c7b2750048f14d3fb16`,
preserved in `Finetune-result/execution/frozen-checkout`. From the repository:

```bash
PYTHONPATH="$PWD/Finetune-result/execution/frozen-checkout" \
  .venv/bin/python tools/analyze_collected_runs.py \
  --model llama3.2-1b --threads 2 --watch
```

The separate [512/1024 extension](LARGE_RANK_EXTENSION.md) has its own source,
protocol, expected grid, epoch diagnostics and local directory. Do not merge
its files into the original cohort or apply the original 140-run audit to it.
For isolated retrieval services, `FYP_STREAM_APP` selects the streaming app
name (default `fyp-archive-stream`), while `FYP_MODAL_RUN` scopes the archives
it may serve. A distinct app prevents one cohort's finalizer from stopping
another cohort's active transfer. These environment settings change transport,
not training or scientific controls.
