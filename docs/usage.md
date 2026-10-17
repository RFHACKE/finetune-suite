# Using `run.sh`

`run.sh` launches the Feasible Rank Window suite. It changes to the repository directory, activates `.venv` when present, and forwards arguments to `rank_suite.cli`. With no arguments it prints help. Training starts when you select `run`.

```bash
./run.sh --help
./run.sh run --help
```

## 1. Set up a GPU machine

Use a Linux host with a working NVIDIA driver, Python 3 available for bootstrap, and enough disk space for models and results. The CUDA installer requires NVIDIA driver version 570 or newer. PyTorch supplies the CUDA runtime; setup does not install a host driver or CUDA compiler.

```bash
git clone https://github.com/RFHACKE/finetune-suite.git fyp-suite
cd fyp-suite
./setup.sh
./run.sh doctor
./run.sh list-models
```

Authenticate Git with an account or deploy key that can read this private repository. Llama also requires Hugging Face access approval and authentication before downloading its weights:

```bash
.venv/bin/hf auth login
```

`HF_TOKEN` is supported through the environment. Keep tokens out of scripts and Git. Optionally download pinned models and prepare data before training:

```bash
./run.sh prepare --output results
```

The largest full-FT baseline has a static memory requirement above 24 GB. Validate it with a pilot on a 32–48 GB GPU; an RTX 4090 is not an all-model full-FT assumption. Actual fit and throughput require the pilot. See [the cloud runbook](CLOUD_RUNBOOK.md) for memory and storage planning.

## 2. Plan, check, and run

These commands print the selected workload without downloading models or starting training:

```bash
./run.sh plan
./run.sh run --dry-run
```

The default main design has **120 three-task sequences**: four models × (five LoRA ranks + one full-FT baseline) × five seeds. Each sequence learns RTE → MNLI Matched → SST-2. Planning needs only Python's standard library; the launcher still needs an available `python`, normally supplied by `.venv` after setup.

Start with a small software check, then measure production-size pilots:

```bash
# Tiny diagnostic: both tracks, reduced data, isolated results.
./run.sh run --mode smoke --models qwen2.5-1.5b --output results

# Largest model: full-length r=16 and full-FT configurations.
./run.sh run --mode pilot --models qwen2.5-3b --output results

# Both tracks for every model; completed pilots are skipped.
./run.sh run --mode pilot --output results

# Complete the remaining main grid.
./run.sh run --output results
```

| Mode | Selection and purpose | Output location |
|---|---|---|
| `main` (default) | Selected models/tracks/ranks/seeds; defaults to the complete study. | `<output>/` |
| `pilot` | Rank 16 and/or full FT, according to `--track`, at seed 42; full main training and evaluation. | `<output>/` |
| `smoke` | Rank 16 and/or full FT at seed 42; 64 training examples/task, one epoch, at most 32 evaluation examples/task. | `<output>/smoke/` |
| `sanity` | Preset RTE LR checks on a training-only 80/20 split; official validation remains separate. | `<output>/sanity/` |

Completed pilots are reused by `main` with the same output and controls. Smoke and sanity never count as main results. With all models and both tracks selected, sanity has **15 unique single-task runs**. Its preset rate comparisons override custom LR lists:

```bash
./run.sh run --mode sanity --output results
```

## 3. Select models, tracks, ranks, and seeds

| Model key | Exact checkpoint | Nominal LoRA LR | Nominal full-FT LR |
|---|---|---:|---:|
| `llama3.2-1b` | `meta-llama/Llama-3.2-1B` | `2.00e-4` | `2.00e-5` |
| `qwen2.5-1.5b` | `Qwen/Qwen2.5-1.5B` | `1.85e-4` | `1.85e-5` |
| `granite3.3-2b` | `ibm-granite/granite-3.3-2b-base` | `1.60e-4` | `1.60e-5` |
| `qwen2.5-3b` | `Qwen/Qwen2.5-3B` | `1.50e-4` | `1.50e-5` |

`--models` accepts these keys or exact checkpoint IDs, separated by spaces. Other checkpoints are rejected. `--track` accepts `lora`, `full`, or `both` (default).

```bash
# One model's complete rank sweep and baseline.
./run.sh run --models qwen2.5-1.5b --track both --output results

# Schedule only the LoRA portion of two models.
./run.sh run --models llama3.2-1b qwen2.5-1.5b --track lora --output results

# Explicit complete rank and seed selection.
./run.sh plan --ranks 4 8 16 32 64 --seeds 42 123 456 789 1024

# Inspect a small subset without training.
./run.sh run --models qwen2.5-1.5b --ranks 4 16 --seeds 42 --dry-run
```

Main defaults are ranks `4 8 16 32 64` and seeds `42 123 456 789 1024`. Subsets help scheduling, but a feasible-window decision requires all five prescribed ranks and seeds plus the nominal five-seed full-FT baseline for that model.

## 4. Numerical settings and learning rates

| Option | Default | Behavior |
|---|---|---|
| `--micro-batch` | `4` | Must be a positive divisor of 16. |
| `--accum` | Inferred (`4` by default) | Micro-batch × accumulation must equal the fixed effective batch of 16. |
| `--eval-batch` | `32` | Controls batch size; full validation is still evaluated. |
| `--optimizer` | `standard` | LoRA uses fp32 AdamW moments; full FT uses 8-bit matrix moments with fp32-state heads/vectors. |
| `--optimizer adamw` | Optional separate cohort | fp32 AdamW moments in both tracks; needs more VRAM. |
| `--deterministic` | Off | Requests strict deterministic algorithms; keep the policy fixed within a cohort. |
| `--save-full-weights` | Off | Also retains full-FT backbone weights at every boundary; substantial disk use. Exact q/v snapshots are always saved. |
| `--lora-lrs` / `--full-lrs` | Model defaults above | Explicit sensitivity rates, either common or model-specific. |

Plan a different batch partition before starting its own cohort:

```bash
./run.sh plan --micro-batch 8 --accum 2 --output results-batch8
./run.sh run --mode pilot --micro-batch 8 --accum 2 --output results-batch8
```

An OOM does not silently change a configuration's batch. Changed batch partitions, evaluation batch, optimizer policy, deterministic policy, scientific controls, or recorded software require a new `--output` directory. Training source hashes are recorded. Compatible model/seed subsets, sharding, sensitivity LR lists, and scheduling prices can be extended within a cohort.

LR overrides accept spaces or comma-separated rates. Scoped lists use model keys:

```bash
# Three LoRA sensitivity rates for one model; planning only.
./run.sh plan --models qwen2.5-1.5b --track lora \
  --lora-lrs qwen2.5-1.5b=9.25e-5,1.85e-4,3.7e-4

# Common full-FT sensitivity rates for selected models; planning only.
./run.sh plan --track full --full-lrs 1e-5 2e-5
```

Keep LR fixed across ranks for a primary comparison. Each rate has its own run identity and summary. Feasible-window analysis requires the **nominal full-FT LR**, so replacing it entirely with a custom list leaves the required baseline missing. Sanity mode uses its own preset rates.

Nominal rates are declared starting values, not proven optima. Epochs, training sample counts, sequence length, task order, alpha, and dropout are fixed protocol controls rather than exposed run options; see [the protocol](PROTOCOL_V2.md).

## 5. Budget and time controls

Enter your actual instance price. `0.40` below is illustrative, not a current Vast.ai quote:

```bash
./run.sh plan --hourly-rate 0.40 --output results

# This invocation: $25 allowance minus $5 reserve = $20 for run admission.
./run.sh run --hourly-rate 0.40 --budget 25 --reserve 5 --output results

# Alternatively, limit elapsed time before admitting another run.
./run.sh run --max-hours 2 --output results
```

| Option | Meaning |
|---|---|
| `--hourly-rate` | USD/hour for the instance serving this worker. |
| `--budget` | Per-invocation allowance; requires an hourly rate and must exceed the reserve. |
| `--reserve` | Amount excluded from the training-time allowance; default `0`. |
| `--max-hours` | Per-invocation elapsed-time allowance. With a budget, the smaller allowance applies. |

Limits finish whole runs and may overrun by one unmeasured run. They **do not stop or destroy the rented instance**. Setup, downloads, storage, transfer, CPU analysis, idle time, and earlier invocations are not automatically deducted. Supply the remaining allowance on each resumed launch.

Cost estimates stay unknown until measured runs cover the selected model/track/optimizer groups. Pilots provide those timings. The 1.25–2× planning range is an allowance, not a confidence interval or a price guarantee.

## 6. Resume and inspect training

Repeat the command with the same output and controls:

```bash
./run.sh run --output results
```

Completed runs are skipped before model loading. Incomplete sequences recover after the last completed task; an interrupted task restarts from that boundary. There is no within-task optimizer recovery.

Logs live in `<output>/logs/worker-0.log` and each run's `train_log.jsonl`. Failures stop the batch by default. `--keep-going` continues after failures for diagnostic work. `--force` reruns selected completed configurations and resets their completion/resume state; use it only when replacement is intended.

Each model's runs are locked to one GPU type within a cohort. Preserve the output tree and environment when moving or resuming the study.

## 7. One or multiple GPUs

One process uses one visible GPU. Select the physical GPU before launching:

```bash
CUDA_VISIBLE_DEVICES=1 ./run.sh run --models qwen2.5-1.5b --output results
```

On a host with identical GPUs, the parallel launcher starts independent workers and keeps each model/seed's ranks and baseline together. It requires Bash 5.1 or newer:

```bash
./run_parallel.sh 0,1 --mode pilot --output results
./run_parallel.sh 0,1 --output results
```

Logs default to `logs/parallel/worker-0.log`, `worker-1.log`, etc. Set `WORKER_LOG_DIR` to change that location. Remaining arguments are forwarded to `run`; do not include another `run` word.

For manual sharding, start these in separate terminals/processes:

```bash
CUDA_VISIBLE_DEVICES=0 ./run.sh run --num-shards 2 --shard-index 0 --output results
CUDA_VISIBLE_DEVICES=1 ./run.sh run --num-shards 2 --shard-index 1 --output results
```

Indices start at zero. Workers must share controls and the output tree. Budgets are per worker, not shared. Calculate a multi-GPU instance bill as total hourly price × instance wall time; summing worker elapsed-cost estimates can count overlapping rental time more than once.

## 8. Finish on CPU

Copy the **complete** results directory to persistent storage before ending the GPU rental, including `cache/projections/` and all boundary artifacts. On a CPU machine, clone the same code revision, restore the output tree, and run:

```bash
./setup.sh --cpu
./run.sh doctor --cpu
./run.sh intruders --output results --threads 4
./run.sh analyze --output results
```

Spectra use saved projections, require no GPU, and resume per matrix. The default solver is `eigh`. For a direct-SVD cross-check:

```bash
./run.sh intruders --output results --solver svd --threads 4
```

Solvers have separate caches, but a pass replaces each run's active spectral summary. Finish a consistent pass before interpreting correlations; analysis rejects mixed solver/numerical-stack summaries. `--models` limits an intruder pass; `--threads` controls CPU threads.

For the isolated smoke diagnostic, select its directory explicitly:

```bash
./run.sh intruders --output results/smoke --models qwen2.5-1.5b --threads 4
./run.sh analyze --output results/smoke --mode smoke
```

Read `results/aggregates/completeness.json` and `results/analysis/report.md` first. Outputs include CSV/JSON tables, figures, accuracy matrices, BWT, peak forgetting, seed variability, paired comparisons, feasible-rank sets, and intruder summaries. Partial grids remain descriptive and cannot certify a complete window.

## 9. Artifacts and command reference

```text
results/
  protocol.json, software.json   # immutable scientific/software identity
  hardware/<model>.json          # GPU type per model
  invocations/*.json             # environment and launch provenance
  batches/*.json                 # worker outcomes and elapsed-cost estimates
  data/main/                    # sample selection and tokenizer provenance
  logs/                         # worker logs
  runs/<run_id>/
    result.json                 # completion marker, metrics, provenance
    train_log.jsonl             # step loss, LR, gradient norm
    resume.pt                   # only while a sequence is incomplete
    checkpoints/task*/
      projections.pt, heads.pt
      predictions_*.npz         # predicted classes and labels, not logits
      adapter_*                 # LoRA boundaries
      full_weights.pt           # only with --save-full-weights
      spectra/<analysis-id>/matrices.json
  cache/projections/            # pretrained q/v weights needed on CPU
  cache/spectra-v2/              # cached pretrained singular vectors
  aggregates/, analysis/        # tables, audit, figures, report
  smoke/, sanity/               # isolated diagnostic cohorts
```

| Command | Purpose | GPU training? |
|---|---|---|
| `./run.sh plan [run options]` | Grid, revisions, work counts, cost coverage. | No |
| `./run.sh prepare [--models ...] [--output ...]` | Download pinned models/tokenizers and prepare data. | No |
| `./run.sh run [run options]` | Train, evaluate, export. | Yes |
| `./run.sh run --dry-run [run options]` | Print the plan without running it. | No |
| `./run.sh intruders [--models ...] [--output ...] [--threads 4] [--solver eigh\|svd]` | Resumable CPU spectra. | No |
| `./run.sh analyze [--output ...] [--mode main\|smoke]` | Aggregates, figures, completeness audit, report. | No |
| `./run.sh doctor [--optimizer adamw8bit\|adamw]` | Imports and actual CUDA/bf16/optimizer checks. | Kernel check only |
| `./run.sh doctor --cpu` | Preparation/analysis dependency checks. | No |
| `./run.sh list-models` | Fixed registry and nominal rates. | No |

The launcher defaults `CUDA_VISIBLE_DEVICES` to `0` and `OMP_NUM_THREADS` to `4` if unset, disables tokenizer parallelism, and sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Export GPU visibility or CPU thread count before launching to override those defaults. Strict deterministic algorithms still require `--deterministic`.

| Exit code | Meaning |
|---:|---|
| `0` | Selected command completed successfully. |
| `1` | One or more training runs failed. |
| `2` | Invalid arguments/configuration or a handled environment/runtime error. |
| `3` | Selected work was deferred, for example by an admission limit. |
| `130` | User interruption. |

## 10. Troubleshooting

| Symptom | Action |
|---|---|
| `python` missing or packages unavailable | Run setup and confirm `.venv/bin/python` exists; the launcher activates it. |
| CUDA unavailable / driver rejected | Select a usable NVIDIA host meeting setup's driver policy, or use CPU setup for analysis. |
| Llama download denied | Accept its access conditions and authenticate with an approved Hugging Face account. |
| GPU out of memory | Choose sufficient VRAM and validate a pilot. A changed batch partition needs a new output cohort. |
| Protocol/software mismatch | Restore the original code/environment or use a new output directory; do not edit manifests to merge cohorts. |
| GPU-type mismatch | Resume that model on its recorded GPU type or create a separate cohort. |
| Unknown cost | Finish full-length pilots for the selected model/track groups, then rerun `plan`. |
| Missing baseline/seeds or empty window | Check completeness and finish missing prescribed runs. A complete study can legitimately yield no feasible rank. |
| CPU analysis missing projection files | Restore the full output tree, including `cache/projections/`, from the GPU machine. |

See [README](../README.md), [the cloud runbook](CLOUD_RUNBOOK.md), [the protocol](PROTOCOL_V2.md), and [the validation record](VALIDATION.md). Legacy `notify.py` and `run_notify.sh` utilities are optional and separately configured; the core workflow does not invoke them.
