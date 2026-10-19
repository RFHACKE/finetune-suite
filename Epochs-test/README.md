# Independent epoch-budget study

This separate exploratory SST-2 study compares **48 fresh fits**:
**four fixed models × six epoch budgets (1–6) × two seeds (42, 123)**.
LoRA rank is 16. Each fit starts from the pinned base model, with fresh adapter,
heads, optimizer, shuffle generator and **budget-matched cosine schedule**.
Matching initialization hashes are checked across budgets within each seed.
No fit continues from a shorter-budget fit. Earlier continuous six-epoch
trajectory results are not reused or mixed into this cohort.

After every epoch we evaluate the complete **2,490-example fixed training set**
and **872-example official SST-2 validation set**. This produces **168 epoch
observations**, **48 endpoints** and **26,208 optimizer steps**. Only endpoints
enter the budget comparison; intermediate observations supply loss curves.
Evaluation restores Torch CPU/CUDA RNG state and never changes training.

| Model | Adapter peak LR | Head multiplier | Head peak LR |
|---|---:|---:|---:|
| Llama-3.2-1B | 4e-4 | 5 | 2e-3 |
| Qwen2.5-1.5B | 1.85e-4 | 1 | 1.85e-4 |
| Granite3.3-2B | 3.2e-4 | 5 | 1.6e-3 |
| Qwen2.5-3B | 3e-4 | 5 | 1.5e-3 |

These are the project's previously selected rates, held fixed here. Other
controls: q/v adapters, alpha32, AdamW/FP32 moments and trainables, BF16 compute,
batch/microbatch16, eval batch32, max length256, weight decay0, dropout0,
clip1. Dataset/model revisions, data selection and original SST-2 task-head
seed position are preserved. There are 156 updates per epoch. Warmup is
ceil(3% of planned steps): **5/10/15/19/24/29 steps** for budgets1–6, followed
by cosine decay over each fit's horizon. This compares duration plus its matched
schedule, rather than duration with a fixed learning-rate path.

The primary endpoint is mean validation NLL. Accuracy, full-training NLL,
generalization gaps, Brier score, wrong-prediction counts/confidence and
individual seed results are retained. Budgets4–6 are paired with budget3.
The predeclared descriptive overfitting screen requires falling train NLL and
rising validation NLL in **both seeds**, plus mean validation NLL rise >0.02.
That practical threshold is a project screening choice, not a significance
test. The hypothesis that three epochs are best is evaluated, never imposed.
No early stopping, adaptive tuning, checkpoint selection or seed removal occurs.

Two seeds give limited descriptive evidence. SD bars are not confidence
intervals. Previously inspected validation data is reused, so this is not an
untouched test-set confirmation. Results apply to this SST-2 subset/rank and do
not establish optimal epochs for other tasks, ranks or continual retention.
The original three-epoch suite default and original result download are separate.

## Execution and cost

The measured forecast is roughly **$3–$4 / 35–50 minutes**, with an operational
**$5 additional ceiling**, four parallel L40S workers and no automatic paid
retries. A conservative resource-time guard reserves $0.60 for startup and CPU
analysis/delivery and cancels remaining workers if its estimate reaches $4.40.
This is an application guard, not a provider-enforced invoice cap; billing can
lag and includes startup/cleanup time. The existing download's account allowance
is preserved separately. Actual receipts/timing are saved under `execution/`.

Use the existing Modal CLI/account and ignored `hf.env`. The token enters cloud
workers as an ephemeral secret, never as an image/archive file. Cached pinned
models/data are shared with earlier runs.

```bash
# CPU-only preparation: no network, no GPU launch.
PYTHONPATH=. .venv/bin/python Epochs-test/epoch_study.py \
  --protocol-only --scope sst2 --output Epochs-test/execution/plan

# Paid launch ONCE, with authorized spending.
modal run --detach Epochs-test/modal_epoch_study.py \
  --stage start --scope sst2 --budget 5 --seconds 3600

# CPU-only status, collection or report regeneration.
modal run Epochs-test/modal_epoch_study.py --stage status
modal run Epochs-test/modal_epoch_study.py --stage collect
modal run Epochs-test/modal_epoch_study.py --stage report

# Local CPU reanalysis after retrieval.
.venv/bin/python -m pip install -r Epochs-test/requirements-report.txt
PYTHONPATH=. .venv/bin/python Epochs-test/build_reports.py --root Epochs-test/results
```

| Script | Option | Meaning/default |
|---|---|---|
| `epoch_study.py` | `--scope {sst2}` | Fixed study task; default `sst2` |
| | `--model KEY` | One fixed model key; required for direct CUDA training |
| | `--output PATH` | Model result root or plan directory; default `Epochs-test/results` |
| | `--seconds N` | Worker allowance; default3000; stop before deadline and preserve evidence |
| | `--protocol-only` | Write protocol and 48-run grid without network/training |
| | `--retry-failed` | Explicit new fresh fit for a failed/incomplete attempt; preserve old attempts; not used automatically |
| `modal_epoch_study.py` | `--stage {start,status,collect,report}` | Default `status`; only `start` launches GPUs |
| | `--scope {sst2}` | Fixed authorized scope |
| | `--budget USD` | Explicit additional ceiling for launch; default0 blocks training; valid1–20 |
| | `--seconds N` | Per-worker timeout; default3600; valid300–10800; aggregate guard may stop earlier |
| `build_reports.py` | `--root PATH` | Cohort root; default `Epochs-test/results` |
| `watch_results.py` | `--poll-seconds N` | Wait for coordinator, default60, minimum10 |
| | `--timeout-seconds N` | Default10800, minimum60; then collect committed evidence without training |
| All | `--help` | Show CLI help |

`FYP_EPOCH_RUN` optionally selects a different remote cohort. Default:
**`epochs-independent-20261019`**, deliberately different from the prior
continuous study. Local/remote receipts prevent accidental duplicate launches.

## Durable evidence and deliverables

Remote data persists on Modal Volume **`fyp-finetune-results`**, under
**`epochs-independent-20261019/`**. Atomic incremental copies and volume commits
publish logs, predictions and failures after every epoch and run. The detached
coordinator runs CPU analysis after all workers return. Failures are explicit,
with run identifiers and tracebacks; there are no automatic GPU retries.
No weight checkpoints are saved, so a failed fit requires restarting that fit
from the base model when explicitly authorized.

Local evidence is collected into `Epochs-test/results/`, with operational
receipts in `Epochs-test/execution/`. Generated deliverables are:

- **`epoch-study.docx`**: full methods, endpoint tables, graphs and interpretation.
- **`epoch-study.pptx`**: presentation of design, comparisons, curves and limits.
- **`epoch-comparison.docx`**: concise per-model endpoint tables.
- JSON/CSV endpoint/paired/within-fit data, PNG/PDF figures and artifact hashes.
- Full-set per-example probabilities/labels/NLL, step losses/LRs/gradients,
  source/data/hardware/environment records, initialization hashes and failures.

Model weights, adapter weights, optimizer states and projection tensors are not
exported. Results and documents are ignored by Git; reproducible source is tracked.

The `fyp-epoch-independent-results.service` collector can retrieve results without
Codex or a terminal. It polls the existing coordinator, then performs CPU-only
collection (up to three delivery attempts). It never starts or retries GPU fits.
Its timestamped progress is in `execution/collection-progress.log`, with CLI
output in `execution/collection-cli.log`. Inspect with:

```bash
systemctl status fyp-epoch-independent-results.service
```

The original downloader has its own service and remains running independently.

## Completed findings (2026-10-19)

All **48 fits / 168 observations / 26,208 optimizer steps** passed validation.
Training took approximately **31 minutes** with four parallel L40S workers.
The conservative GPU/CPU/RAM resource estimate is **$3.05**; provider billing
can lag, and CPU delivery plus the separately running download incur small
additional charges. Receipts retain the observed account readings and estimates.
The controls, raw measurements and primary comparison are unchanged by CPU
analysis revision 2. Descriptive preferences among the six tested budgets are:

| Model | Lowest mean validation NLL | Highest mean validation accuracy |
|---|---:|---:|
| Llama-3.2-1B | 2 epochs | 4 epochs |
| Qwen2.5-1.5B | 1 epoch | 5 epochs |
| Granite3.3-2B | 2 epochs | 4 epochs |
| Qwen2.5-3B | 1 epoch | 4 epochs |

The full hypothesis that three epochs are best for every model and all larger
budgets consistently overfit is **not supported**. Differences are descriptive,
with only two seeds and reused validation data. Probability loss and accuracy
prefer different budgets; close minima should not be overinterpreted.

Revision 2 adds an explicitly **post-hoc** individual-seed chart and explanation
of accuracy regressions of at least two percentage points versus budget3. This
screen does not change the predeclared analysis or remove any fit. Granite's
five-epoch seed42 fit falls from **95.99% to 88.88%**; Qwen-3B's six-epoch seed42
fit falls from **95.41% to 79.93%**. Their training NLL rises versus budget3,
so these cases cannot both be labeled simple improving-train/worsening-validation
overfitting. Their mechanisms are not established by this study.

The initial reports remain under `results/cpu-analysis-v1/reports/`; the final
reports are under `results/reports/`. Revision2 source is saved remotely under
`execution/analysis-v2/`, alongside the original frozen training/analysis source.
No additional GPU training is used for these diagnostics.

## Primary sources

- Hu et al., LoRA: https://arxiv.org/abs/2106.09685
- Wang et al., GLUE: https://arxiv.org/abs/1804.07461
- Dodge et al., initialization/data order/early stopping: https://arxiv.org/abs/2002.06305
- Prechelt, Early Stopping—But When?: https://doi.org/10.1007/3-540-49430-8_3
- Cosine schedule: https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.get_cosine_schedule_with_warmup
- Modal pricing, checked2026-10-19: https://modal.com/pricing
