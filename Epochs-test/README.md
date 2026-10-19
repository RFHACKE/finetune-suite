# Separate epoch trajectory study

The authorized study uses **SST-2**, all four fixed models, LoRA **rank 16**,
and **three seeds (42, 123, 456)**. There are **12 continuous six-epoch runs**
and **72 epoch observations**. Each run starts fresh from the pinned pretrained
model, then trains epoch 1, evaluates, trains epoch 2, evaluates, and continues
to epoch 6. Optimizer and scheduler state continue across all six epochs.

Every observation evaluates **all 2,490 fixed training examples and all 872
validation examples**. Evaluation restores RNG state and never changes training.
The original rank sweeps and the running downloader remain separate.

| Model | Adapter LR | Head LR multiplier |
|---|---:|---:|
| Llama-3.2-1B | 0.000400 | 5 |
| Qwen2.5-1.5B | 0.000185 | 1 |
| Granite-3.3-2B | 0.000320 | 5 |
| Qwen2.5-3B | 0.000300 | 5 |

These are the best previously tested LR/head policies, not global optima.
Controls include AdamW, **weight decay 0**, **dropout 0**, effective/micro batch
16, evaluation batch 32, maximum length 256, BF16 compute, FP32 trainable weights
and Adam moments, q/v adapters, alpha 32, clipping 1, and 3% warmup followed by
cosine decay to zero over six epochs (936 optimizer steps per run).

The smaller study is expected to cost approximately **$1.50–3** on four parallel
L40S workers. Its operational ceiling is **$4**, below the approved $8 ceiling.
Conservative aggregate resource accounting, worker timeouts and no automatic
paid retries bound the work. Spending less than the ceiling is expected.

The completed cohort contains all **12 runs and 72 observations**, with no
failed or deferred runs. Training finished in about **10 minutes**. The worker
runtime resource estimate is **$0.992**; the observed account-meter increase
through initial reporting/collection was about **$1.30**, including startup
overhead and the continuing original download. It is below the $4 ceiling.

## Interpretation

The primary metric is full validation negative log-likelihood (NLL), with
accuracy, training NLL, generalization gaps, Brier score, and confidence on wrong
predictions retained. Epochs 4–6 are compared with epoch 3 within each seed.
A descriptive overfitting pattern requires falling training NLL and rising
validation NLL. A consistency label requires this direction in all three seeds
and a mean validation NLL rise over 0.02. This is not a significance test.
Plots show individual seeds, means and sample SD (not confidence intervals).

The study tests whether epoch 3 is best; it must report contrary results.
No epochs or seeds are discarded to support a desired conclusion. Observations
within a run are correlated, not independent replications. Previously inspected
official validation data is reused; this is not an untouched test-set estimate.
One task, one rank and three seeds do not establish a universal optimum.

**Schedule caveat:** epoch 3 inside the six-epoch cosine schedule differs from
epoch 3 in the original three-epoch schedule. These curves reveal learning and
possible overfitting within six-epoch trajectories. They do not directly compare
independently optimized epoch budgets or prove the original three-epoch policy
optimal. The project's global epoch default is unchanged.

The observed lowest mean validation NLL occurs at epoch **1** for Llama-1B,
Qwen-1.5B and Granite-2B, and epoch **3** for Qwen-3B. Highest mean validation
accuracy occurs at epoch **6** for Llama, Granite and Qwen-3B, and epoch **5**
for Qwen-1.5B. Thus, the data does not establish epoch 3 as universally best.
Mean NLL and accuracy move in different directions, so both must be considered.

CPU analysis revision 2 adds explicitly **exploratory** adjacent-epoch changes
and an analysis of the same validation examples misclassified at both epochs
3 and 6. These persistent-error confidence/NLL changes help explain worsening
probability loss despite improving accuracy. They do not alter the predeclared
comparisons, hypothesis verdict, training protocol or raw data. JSON and CSV
include the diagnostics and the matched example positions.

## Commands

Use the existing configured Modal CLI/account and `hf.env`. Models and data are
pinned and read from the existing shared cache. Tokens enter the container as an
ephemeral secret; no credential file is copied into an image or result archive.

```bash
# CPU-only protocol/grid preparation; no cloud call.
PYTHONPATH=. .venv/bin/python Epochs-test/epoch_study.py \
  --protocol-only --scope sst2 --output Epochs-test/execution/plan

# Paid training: launch only once, with authorized spending.
modal run --detach Epochs-test/modal_epoch_study.py \
  --stage start --scope sst2 --budget 4 --seconds 3600

# CPU-only remote status, collection, or report regeneration.
modal run Epochs-test/modal_epoch_study.py --stage status
modal run Epochs-test/modal_epoch_study.py --stage collect
modal run Epochs-test/modal_epoch_study.py --stage report

# CPU-only local report regeneration after collection.
.venv/bin/python -m pip install -r Epochs-test/requirements-report.txt
PYTHONPATH=. .venv/bin/python Epochs-test/build_reports.py --root Epochs-test/results
```

| Script | Option | Meaning/default |
|---|---|---|
| `epoch_study.py` | `--scope {sst2}` | Fixed study scope, default `sst2` |
| | `--model KEY` | One of the four fixed keys; required for direct GPU training |
| | `--output PATH` | Model result root or protocol-only directory; default `Epochs-test/results` |
| | `--seconds N` | Worker time allowance; default 3000 |
| | `--protocol-only` | Write protocol and 12-run grid without training/network |
| | `--retry-failed` | Explicitly retry failed/incomplete runs from pretrained weights; preserve prior attempts |
| `modal_epoch_study.py` | `--stage {start,status,collect,report}` | Default `status`; only `start` launches GPUs |
| | `--scope {sst2}` | Fixed study scope |
| | `--budget USD` | Explicit authorized additional ceiling for `start`; default 0 prevents accidental launches |
| | `--seconds N` | Hard timeout per worker; default 3600; aggregate budget may stop workers sooner |
| `build_reports.py` | `--root PATH` | Retrieved cohort root; default `Epochs-test/results` |
| `watch_results.py` | `--poll-seconds N` | Check the existing coordinator every 60 seconds by default (minimum 10) |
| | `--timeout-seconds N` | Default 10800; after this deadline collect available evidence, without starting training |
| All | `--help` | CLI help |

`FYP_EPOCH_RUN` optionally names a different remote cohort; default is
`epochs-20261019`. Local/remote launch receipts prevent accidental duplicates.
A new cohort still requires authorized spending.

## Stored evidence

Remote results persist in Modal Volume `fyp-finetune-results` under
`epochs-20261019/`. Observations, logs, completed runs, failed attempts and source
provenance are committed after each epoch and run. A detached coordinator keeps
training and CPU report generation alive without the local client. No automatic
GPU retries occur. Incomplete runs can later be restarted explicitly from the
pretrained model; no training checkpoints are saved for within-run recovery.

Local results live in `Epochs-test/results` and operational receipts in
`Epochs-test/execution`. Final **`epoch-study.docx`** and **`epoch-study.pptx`**
are copied here after retrieval. JSON/CSV, PNG/PDF figures and all per-epoch
labels, probabilities and example NLL are retained for CPU analysis.
Model weights, adapter weights, optimizer states and projection tensors are
never exported. Generated results/documents are ignored by Git; source code
and this protocol description are versioned independently.

The local `fyp-epoch-results.service` waits for this existing cloud coordinator
and retrieves the small archive automatically, independently of Codex or the
terminal. Its timestamped log is `execution/collection-progress.log`; CLI output
is in `execution/collection-cli.log`. CPU-only collection can retry up to three
times; this service never launches or retries GPU training. If it is stopped,
the remote evidence remains available through `--stage collect`. Inspect it with
`systemctl status fyp-epoch-results.service`. The original downloader runs in its
own separate service.

## Research sources

- Hu et al., LoRA: https://arxiv.org/abs/2106.09685
- Wang et al., GLUE: https://arxiv.org/abs/1804.07461
- Dodge et al., initialization, data order and early stopping: https://arxiv.org/abs/2002.06305
- Prechelt, Early Stopping—But When?: https://doi.org/10.1007/3-540-49430-8_3
- Cosine schedule parameters: https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.get_cosine_schedule_with_warmup
- Modal resource pricing, checked 2026-10-19: https://modal.com/pricing
