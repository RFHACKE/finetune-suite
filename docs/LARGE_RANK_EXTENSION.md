# LoRA factor-width extension: 512 and 1024

The additional experiment is a separate cohort, `frw-20261018-r512-1024`.
It contains 40 sequences: four fixed checkpoint revisions × factor widths
512/1024 × seeds 42/123/456/789/1024. The original 140-sequence cohort and its
frozen source are preserved unchanged. Local extension data belongs under
`Finetune-result/large-ranks/`, rather than the original cohort's `runs/`.

The selected LoRA rates are Llama 4.00e-4, Qwen 1.5B 1.85e-4, Granite 3.20e-4,
and Qwen 3B 3.00e-4. Classifier-head multipliers remain 5, 1, 5 and 5. These are
transferred best-tested defaults; the original calibration did not test these
new widths. Its evidence is copied to `execution/calibration-origin/`, and the
lineage is recorded explicitly rather than claiming a new calibration.

All models, checkpoint revisions, data selections, task order, q/v targets,
three epochs, full task-boundary validation, alpha=2r, dropout=0, weight decay,
optimizer precision, and batch partition (micro 16, accumulation 1, evaluation
32) match the original cloud experiment. Existing weight decay, cosine decay
and clipping remain active. No early stopping or per-run LR adjustment is
introduced: changing these in response to the main validation curves would
confound rank comparisons and turn those measurements into selection data.

## Factor width and effective rank

`--allow-overcomplete-ranks` explicitly permits factor widths above a matrix
dimension. Effective update rank is still bounded by the projection shape.

| Model | Maximum q update rank | Maximum v update rank |
|---|---:|---:|
| Llama 3.2 1B | 2048 | 512 |
| Qwen 2.5 1.5B | 1536 | 256 |
| Granite 3.3 2B | 2048 | 512 |
| Qwen 2.5 3B | 2048 | 256 |

Consequently, width 1024 cannot create a rank-1024 v update. Larger factors
increase trainable parameter counts and can change optimization despite the
same effective-rank ceiling. The protocol and each result record these
semantics; q/v targets are not changed to avoid this ceiling. PEFT's vanilla
LoRA factor shapes and alpha/r scaling remain in use, as described in the
[PEFT LoRA documentation](https://huggingface.co/docs/peft/en/package_reference/lora).

## Overfitting observations

`--epoch-diagnostics` evaluates fixed stratified probes of up to 256 training
examples and 256 official validation examples for the current task after every
epoch. `--diagnostic-samples N` changes that probe size and therefore belongs
in a different output cohort. Probes share DATA_SEED=42 across models, ranks
and optimization seeds; tokenization-specific provenance and exact positions
are saved. The training probe is in-sample, not a held-out selection set.

The diagnostic evaluation restores training modes and CPU/CUDA RNG state.
It does not change optimizer state, shuffling, LR, epoch count or checkpoint
selection. An automated test verifies identical weights and RNG states with
monitoring enabled and disabled on a stochastic model.

`epoch_diagnostics.jsonl` stores training/validation NLL, accuracy, their gaps,
online training loss, epoch, optimizer step and diagnostic runtime.
`diagnostics/` retains probe positions plus per-example labels, predictions,
probabilities and NLL. These support later paired analyses and calibration
plots. All normal full-validation predictions and task-boundary exports remain.

A descriptive warning marks an adjacent epoch where validation probe NLL rises
by more than 0.02 while training probe NLL falls by more than 0.02. This is not
a statistical test: sampling noise and distribution differences can also
produce gaps. `overfitting.json` records per-task warnings; it never selects
the lowest-probe epoch. Fixed probes may miss overfitting outside their sample.

After GPU workers finish, a CPU task writes `overfitting/summary.json`,
`overfitting/epoch-metrics.csv` and PNG/SVG plots for each model/width. Plots
show training/validation NLL and accuracy versus epoch, with mean ± sample
standard deviation over the available seeds. Any failed or missing seeds
remain explicit in the remote inventory.

## Cloud execution and retrieval

```bash
modal run --detach tools/modal_large_ranks.py --seconds 14400 --stage-budget 13.5
```

The runner uses one 48 GB L40S per model, four workers concurrently. CLI options:

| Option | Default | Meaning |
|---|---|---|
| `--seconds` | `14400` | Per-model GPU timeout; 120–18000 seconds allowed. |
| `--stage-budget` | `13.5` | Aggregate conservative GPU-worker allowance in USD; maximum 13.50. |
| `--archive-only` | Off | Recover the CPU archiver and chart generation using the saved cohort journal; starts no GPU experiments. A saved recovery call prevents accidental duplicate starts. |
| `--profile` | Active Modal profile | Modal's common CLI profile override. |
| `--recover-model` | Empty | One explicit recovery of an identified hardware-provenance interruption, using the saved cohort. Keeps the same GPU name/capability, records small VRAM-capacity differences per attempt, and guards the combined original/recovery GPU cost at $13.50. |
| `--help` | — | Modal-generated entrypoint help. |

The approved additional operational cap is $15; $1.50 is reserved outside
the GPU stage for CPU coordination, archiving and chart generation. Prices
use [Modal's published rates](https://modal.com/pricing). The coordinator
conservatively includes GPU queue/startup time and cancels unfinished workers
before its aggregate stage allowance is spent. No automatic GPU retries run.

Remote storage uses the existing private `fyp-finetune-results` and
`fyp-finetune-archive` volumes, under the new cohort directory. Raw tensors,
partial task-boundary recovery checkpoints and logs persist remotely. A durable
CPU parent keeps child calls running independently of the launching client.
Lossless archive packaging and charts run on CPU. The local call identifier is
saved to `Finetune-result/large-ranks/execution/main-calls.json`.

Inspect remote `execution/status.json`, `failures.json`, `retry-plan.json`,
`main-workers.json`, `runs.csv`, and `archive-progress.json`. These record the
exact expected 40 identities, completed/failed/missing runs and committed
recovery boundaries. `execution/source/` and its checksum manifest preserve
the source used by this cohort. Full numerical and CPU spectral audits remain
required after retrieval; training completion alone is not final scientific
verification. Retain both remote copies until local checksums pass.

The local watcher retrieves completed epoch logs and generates provisional
charts while GPU training continues. After both GPU work and CPU archive
recovery finish, it can retrieve the complete metadata/charts bundle through
authenticated Modal RPC, independently of the much larger raw tensor transfer:

```bash
python3 -m tools.watch_large_ranks --fetch-metadata --fetch-raw
```

| Watcher option | Default | Meaning |
|---|---|---|
| `--root` | `Finetune-result/large-ranks` | Local cohort root with saved main and archive recovery call IDs. |
| `--interval` | `60` | Polling seconds; minimum 15. Durable status snapshots are fetched at most every 120 seconds while running. |
| `--max-seconds` | `21600` | Training/archiving observation lifetime; minimum 60. Reaching it does not cancel cloud work. Subsequent raw collection has its separate spending guard. |
| `--fetch-metadata` | Off | After the exact 40-run cohort finishes and is archived, retrieve its checked metadata and charts using a CPU job. |
| `--fetch-raw` | Off | Requires `--fetch-metadata`. Wait for the original cohort's downloads, then retrieve all 40 tensor archives with checksums and native resumption. Monitor the saved overall spending ceiling and retain remote data if that ceiling stops transfer. |
| `--resume-download` | Off | Use by itself after an interruption. Reuse the existing delivery catalog/metadata and resume partial archives, manage the CPU endpoint and spending guard, and append to the same progress log. Rejects an already active downloader and starts no GPU training. |
| `-h`, `--help` | — | Show help. |

For download resumption, `./start-download.sh` is the executable shortcut for
`python3 -m tools.watch_large_ranks --resume-download`. It selects Python with
the Modal SDK installed; `MODAL_PYTHON=/path/to/python` overrides that choice.
Pass `--root PATH` to select another local extension directory. It reports an
already active or completed transfer without starting a second downloader.
When systemd is available it runs as a background service with restart-on-failure;
`--foreground` keeps it in the terminal instead. See the
[delivery reference](MODAL_EXECUTION.md#finished-cohort-segmented-delivery) for
service status/stop commands, heartbeat files and the restart limits.

`execution/progress.json` explicitly distinguishes retrieved diagnostic
records from raw scientific exports. The watcher uses the local frozen source
for chart generation. It never retries or launches GPU training. Its local
process requires the laptop to remain awake; cloud training, raw data and
remote charts persist independently if that process stops.

Raw collection uses a separate, authenticated CPU endpoint named
`fyp-large-rank-stream`, scoped only to this cohort and stopped after transfer
or a spending stop. It reuses the previously authorized random-token access
mechanism. The original endpoint is unaffected. The spending guard reads
`Finetune-result/execution/active-budget.json` and checks account usage every
five minutes, retaining a $0.25 margin. Brief billing API failures can reuse a
checked reading for at most fifteen minutes, with more than $0.50 of headroom,
while retrying every thirty seconds; otherwise collection pauses with partials
retained. Billing checks and failures have separate `download-billing-*` logs.
Each collection cycle has three transport attempts; exhaustion returns an
error so the bounded systemd restart policy can recover it. There are no GPU
retries. `execution/raw-delivery-*.json` distinguishes a
complete verified download, a budget stop and a transport failure. Raw
collection alone does not claim completed CPU spectral analysis or final audit.
The CPU endpoint reserves 1 GiB RAM per container, caps it at 2 GiB and limits
per-container concurrency to four requests. Four local archive workers use eight,
four and two connections across successive attempts. Fifteen minutes without
new archive bytes triggers a resumable restart. The supervisor's 30-second
heartbeat and exception tracebacks are saved independently of the archive worker
in `execution/download-supervisor.json` and `execution/download-supervisor.log`.
As verified raw exports arrive, the watcher starts local CPU spectral helpers
with two threads per model and this cohort's frozen source. They save matrix
diagnostics and caches independently of the downloader; their PIDs and logs
are under `execution/cpu-workers.json` and `execution/cpu-<model>.log`.
Only checksum-verified runs are analyzed. Full numerical/prediction auditing
and final comparative analysis remain separate checks after collection.
