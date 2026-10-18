# Using `run.sh`

`run.sh` launches the Feasible Rank Window suite. It changes to the repository directory, activates `.venv` when present, and forwards arguments to `rank_suite.cli`. With no arguments it prints help. `run` starts the selected workload; `run --test-lr` performs calibration only and then exits. The shortcut `./run.sh --test-lr ...` is also accepted.

Current experiments use protocol `frw-v3`. Use a fresh results directory for this amendment; v2 training results and controls cannot be pooled with v3. The four checkpoint identities remain fixed.

For every accepted flag, including defaults and command-specific differences, see the [complete CLI option reference](#11-complete-cli-option-reference). The earlier sections provide the end-to-end workflow.

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

Start with a small software check, calibrate defaults, then measure production-size pilots. Keep pre-calibration smoke results in their own root:

```bash
# Tiny diagnostic: both tracks, reduced data, isolated results.
./run.sh run --mode smoke --models qwen2.5-1.5b --output results-check

# Inspect calibration cost in work units before starting it.
./run.sh plan --test-lr --output results

# Select and save per-model/track LR and head defaults; exits after calibration.
./run.sh --test-lr --output results

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

| Model key | Exact checkpoint | Default LoRA LR | LoRA head multiplier | Full-FT prior |
|---|---|---:|---:|---:|
| `llama3.2-1b` | `meta-llama/Llama-3.2-1B` | `4.00e-4` | `5` | `2.00e-5` |
| `qwen2.5-1.5b` | `Qwen/Qwen2.5-1.5B` | `1.85e-4` | `1` | `1.85e-5` |
| `granite3.3-2b` | `ibm-granite/granite-3.3-2b-base` | `3.20e-4` | `5` | `1.60e-5` |
| `qwen2.5-3b` | `Qwen/Qwen2.5-3B` | `3.00e-4` | `5` | `1.50e-5` |

`--models` accepts these keys or exact checkpoint IDs, separated by spaces. Other checkpoints are rejected. `--track` accepts `lora`, `full`, or `both` (default).

The LoRA LR/head pairs are project defaults promoted from the completed 48-case Modal calibration. They are the best tested training-holdout settings, with upper-grid-boundary selections for Llama, Granite and Qwen3B. The measured partition was micro-batch 16/accumulation 1; the ordinary CLI remains 4/4 unless those flags are supplied. Full-FT values remain uncalibrated priors with head multiplier 1. See [LORA_DEFAULTS.md](LORA_DEFAULTS.md) and its checksummed evidence summary.

After a completed `--test-lr`, `plan` and `run` use valid selected rates/head ratios in `<output>/lr_defaults.json` ahead of these project defaults. Explicit CLI rates/head ratios take precedence. `list-models` shows registry rates and LoRA head ratios. Main analysis anchors full-FT comparisons to the frozen selected baseline, without selecting its LR from main validation outcomes. Existing Modal results use their [frozen source checkout](OVERNIGHT.md); this defaults promotion does not rewrite old calibration fingerprints.

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
| `--head-lr-multiplier` | Cohort value, otherwise Section 3's LoRA ratio; full FT uses `1` | Positive finite head/backbone LR ratio. An explicit value overrides the selected head ratio and must be fixed before main training. |

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
./run.sh intruders --output results-check/smoke --models qwen2.5-1.5b --threads 4
./run.sh analyze --output results-check/smoke --mode smoke
```

Read `results/aggregates/completeness.json` and `results/analysis/report.md` first. Outputs include CSV/JSON tables, figures, accuracy matrices, BWT, peak forgetting, seed variability, paired comparisons, feasible-rank sets, and intruder summaries. Partial grids remain descriptive and cannot certify a complete window.

## 9. Artifacts and command reference

```text
results/
  lr_defaults.json              # activated calibration choices and evidence identity
  lr_test/                      # isolated calibration runs, plans, JSON reports and charts
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
  cache/spectra-v3/              # cached pretrained vectors and singular values
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

## 11. Complete CLI option reference

Syntax: `./run.sh COMMAND [OPTIONS]`. Put command options **after** the command, for example `./run.sh run --mode pilot`. Boolean switches take no value: use `--deterministic`, not `--deterministic true`. An omitted switch is off. `VALUE ...` means one or more values; do not type the ellipsis.

This reference covers the current parser in `rank_suite/cli.py`. Options are command-specific: in particular, `doctor --optimizer` has different choices from `run --optimizer`, and `analyze --mode` has different choices from `run --mode`.

### Help: global and every command

<!-- cli-options: global run plan prepare intruders analyze doctor list-models -->

| Flag | Value/default | Effect and example |
|---|---|---|
| `-h`, `--help` | No value; help is not printed unless requested or no command is supplied. | Print help and exit successfully. `./run.sh --help` lists commands; `./run.sh intruders --help` lists that command's options. Both spellings work globally and after every command. |

### `run` and `plan`: all shared options

<!-- cli-options: run plan -->

Every option in this table is accepted by both commands. `plan` validates configuration and prints the selected workload; it does not train, export weights, enforce an admission limit, or replace results. Thus execution switches such as `--force` and `--keep-going` describe the intended run policy when supplied to `plan`.

| Option and value | Effective default | Meaning, accepted values, and constraints |
|---|---|---|
| `--test-lr` | Off | Select LR/head defaults on training-only holdouts, save JSON/charts, and exit without main training. `plan --test-lr` prints the calibration plan. Shortcut: `./run.sh --test-lr`. See Section 13 for the selection rule and option interactions. |
| `--head-lr-multiplier MULTIPLIER` | Cohort ratio, otherwise Section 3's model-specific LoRA ratio; full FT uses `1`. | Finite positive classifier-head/backbone LR ratio. Explicit values override all selected tracks/models. Examples: `--head-lr-multiplier 5`; `run --test-lr --head-lr-multiplier 1` searches LRs with the head ratio fixed instead of searching `{1,5}`. |
| `--models MODEL ...` | All four fixed models. | Registry keys or exact checkpoint IDs from Section 3, separated by spaces. Duplicates are removed. Other checkpoints are rejected. Example: `--models llama3.2-1b qwen2.5-3b`. |
| `--output PATH` | `results` | Experiment directory. Relative paths resolve from the repository directory because the launcher changes there. Absolute paths are accepted, for example `--output /workspace/fyp-results`. Main/pilot use it directly; smoke/sanity append their own subdirectory. |
| `--track TRACK` | `both` | Choose `lora`, `full`, or `both`. `full` means all non-embedding backbone weights plus the current task head; token embeddings stay frozen. Ranks apply to LoRA only. |
| `--mode MODE` | `main` | Choose `main`, `pilot`, `sanity`, or `smoke`. Pilot/smoke force rank 16 and seed 42. Sanity additionally forces its preset LR comparisons and RTE training holdout. See Section 2 for data, epoch, and output differences. |
| `--ranks RANK ...` | `4 8 16 32 64` | Positive integers, separated by spaces or commas; duplicates are removed. By default each factor width is bounded by the narrowest selected `v_proj`: 256 with either Qwen, 512 with Llama/Granite only. Explicit larger-width extensions require `--allow-overcomplete-ranks` and a separate output cohort; effective update rank remains bounded by matrix dimensions. Other ranks are sensitivity experiments; the main window requires the prescribed grid. Pilot/smoke/sanity use rank 16 instead. |
| `--allow-overcomplete-ranks` | Off. | Explicitly permit LoRA factor widths above projection dimensions. Requires a separate output cohort and records the changed rank semantics in the protocol. The four models and q/v targets stay fixed. See [the 512/1024 extension](LARGE_RANK_EXTENSION.md). |
| `--epoch-diagnostics` | Off. | Evaluate fixed stratified training/validation probes after each epoch. Save NLL, accuracy, gaps and per-example predictions/probabilities; preserve training RNG and checkpoint policy. This observation-only protocol requires a separate output cohort. |
| `--diagnostic-samples N` | `256` | Positive number of examples per diagnostic split, capped by its size; used with `--epoch-diagnostics`. Probe identities are fixed by DATA_SEED=42, recorded on disk, and independent of optimization seed. Does not reduce the full task-boundary evaluation. |
| `--seeds SEED ...` | `42 123 456 789 1024` | Integers from `0` through `4294967295`, separated by spaces or commas; main-grid duplicates are removed. Pilot/smoke/sanity use seed 42 instead. The main window requires the five prescribed seeds. |
| `--lora-lrs RATE ...` | Calibrated cohort rate, otherwise the Section 3 project default. | Finite positive rates, using decimal or scientific notation. Supply a common list (`--lora-lrs 1e-4 2e-4`) or scoped lists (`--lora-lrs qwen2.5-1.5b=1e-4,1.85e-4`). Scoped names must be selected registry keys, even if `--models` uses exact checkpoint IDs. Sanity uses its original methodology comparisons instead. |
| `--full-lrs RATE ...` | Calibrated cohort rate, otherwise the Section 3 prior. | Same syntax and positivity rules as `--lora-lrs`, for the full track. A model's nominal rate must be present to provide the required window baseline. Example: `--full-lrs qwen2.5-1.5b=1e-5,1.85e-5`. Sanity uses nominal and twice-nominal rates instead. |
| `--optimizer POLICY` | `standard` | `standard`: fp32 AdamW moments for LoRA, 8-bit dense-matrix moments with fp32 head/vector overrides for full FT. `adamw`: fp32 moments in both tracks. A changed policy requires a separate output cohort. This option does not quantize model weights. |
| `--micro-batch N` | `4` | Positive integer divisor of 16: `1`, `2`, `4`, `8`, or `16`. Used for all tracks/ranks in the cohort. A changed partition requires a new output directory. |
| `--accum N` | `16 / micro_batch`; normally `4`. | Positive integer accumulation count. If specified, `micro_batch × accum` must equal 16. Examples: `--micro-batch 8 --accum 2`, or `--micro-batch 2 --accum 8`. |
| `--eval-batch N` | `32` | Positive integer evaluation batch size. Does not reduce the validation set; smoke retains its separate diagnostic evaluation cap. Changing it requires a new cohort. |
| `--deterministic` | Off. | Enable strict deterministic PyTorch algorithms. May reduce speed or fail if an operation lacks a deterministic implementation. Use the same setting across a cohort; this does not promise identical results across hardware/software versions. |
| `--save-full-weights` | Off. | Save each full-FT backbone at every task boundary in addition to always-saved q/v projections and heads. LoRA already saves adapters. Adds substantial disk/I/O use; enabling it later does not retroactively export completed runs. |
| `--force` | Off; completed runs are skipped. | Rerun selected completed configurations and discard their completion/resume markers before restarting. Does not bypass protocol/software compatibility checks and does not mean deleting the entire output directory. Example: `./run.sh run --models qwen2.5-1.5b --ranks 16 --seeds 42 --force`. |
| `--keep-going` | Off; stop after a run fails. | Continue to subsequent selected runs after a failure. Failures are recorded in `failures-worker-<index>.jsonl`; the batch still exits nonzero if any run failed. It does not automatically repair or retry failed runs. |
| `--num-shards N` | `1` | Positive integer count of independent GPU workers. Partitioning keeps each model/seed block's ranks and baseline together. This is independent-run scheduling, not distributed training of one model. Start all intended shards to finish the selected grid. |
| `--shard-index I` | `0` | Zero-based worker index; require `0 <= I < num_shards`. Example: `--num-shards 2 --shard-index 1`. Changing shard assignment does not change the scientific controls. |
| `--hourly-rate USD` | Unset; no USD estimate. | Finite positive USD/hour rate. Use the actual total instance quote, including its GPUs. Worker summaries multiply this by that worker's elapsed time; sums across overlapping workers are not an instance bill. Example: `--hourly-rate 0.40` uses an illustrative rate. |
| `--budget USD` | Unset; no budget admission limit. | Finite positive per-invocation allowance. Requires `--hourly-rate` and `budget > reserve`. The training-time allowance is `(budget - reserve) / hourly_rate` hours. It is not a lifetime account cap or an instruction to terminate the rental. |
| `--reserve USD` | `0` | Finite nonnegative amount excluded from the budget's training-time allowance. Without `--budget`, it does not create a limit. Example: `--budget 25 --reserve 5` leaves a $20 admission allowance for this invocation. |
| `--max-hours HOURS` | Unset; no elapsed-time admission limit. | Finite positive hours. Finish the current run before stopping admission; measured next-run estimates may defer work earlier. When combined with a budget, the smaller time allowance applies. Example: `--max-hours 2.5`. |

For LR lists, a scoped list takes precedence for that model; a common list applies to models without a scoped list. For example:

```bash
./run.sh plan --models llama3.2-1b qwen2.5-1.5b --track lora \
  --lora-lrs 1e-4 2e-4 qwen2.5-1.5b=9.25e-5,1.85e-4
```

Here Llama receives `1e-4` and `2e-4`, while Qwen receives `9.25e-5` and `1.85e-4`. Pilot/smoke still honor compatible LR lists; sanity uses its presets. Supplied ranks, seeds, rates, and numeric controls must pass validation even when a mode overrides their selection.

### `run`: additional option

<!-- cli-options: run -->

| Option | Default | Meaning and example |
|---|---|---|
| `--dry-run` | Off. | Print the plan and exit without training. Accepts the other `run` options. Example: `./run.sh run --models qwen2.5-1.5b --track lora --dry-run`. `plan` already does this, so `plan --dry-run` is not an accepted spelling. |

### `prepare`: every option

<!-- cli-options: prepare -->

| Option and value | Default | Meaning and example |
|---|---|---|
| `--models MODEL ...` | All four fixed models. | Prefetch only these registry keys/exact checkpoint IDs and their tokenizers. Example: `--models qwen2.5-1.5b`. Models must belong to the fixed registry. |
| `--output PATH` | `results` | Destination for selected-sample/tokenizer data provenance under `data/main/`. Model downloads use the Hugging Face cache, not this directory. Example: `./run.sh prepare --models qwen2.5-1.5b --output results`. |

Also accepts `-h` and `--help`. It supports CPU preparation and has no `--cpu` option. Hugging Face authentication/cache environment variables are separate from CLI options.

### `intruders`: every option

<!-- cli-options: intruders -->

| Option and value | Default | Meaning and example |
|---|---|---|
| `--models MODEL ...` | All four fixed models. | Analyze saved results only for these registry keys/exact checkpoint IDs. Example: `--models qwen2.5-1.5b qwen2.5-3b`. |
| `--output PATH` | `results` | Exact results root to read, including its runs and projection cache. Unlike `run --mode smoke`, this command does not append a mode directory. Use `--output results-check/smoke` for smoke artifacts. |
| `--threads N` | `4` | Positive integer CPU thread count used by Torch and the BLAS thread limiter. Example: `--threads 8`. This is not a GPU count. |
| `--solver SOLVER` | `eigh` | Choose `eigh` (float64 Gram eigendecomposition for leading tuned vectors) or `svd` (direct fp32 SVD cross-check). Caches are separated, but active run summaries are replaced; complete a consistent pass. Example: `./run.sh intruders --output results --threads 4 --solver svd`. |

Also accepts `-h` and `--help`. Spectral analysis is CPU-based; there is no `--cpu`, `--mode`, `--force`, rank, epsilon, or top-k CLI option for this command. Protocol constants govern the latter measurements.

### `analyze`: every option

<!-- cli-options: analyze -->

| Option and value | Default | Meaning and example |
|---|---|---|
| `--output PATH` | `results` | Exact results root for aggregates and figures. Does not append a mode directory. Example: `./run.sh analyze --output results`. |
| `--mode MODE` | `main` | Only `main` or `smoke` are accepted. Filters results by their stored mode; a pilot is stored as a reusable main configuration. Use `./run.sh analyze --output results-check/smoke --mode smoke` for smoke reports. Main-root analysis also reads available sanity results from its `sanity/` child. |

Also accepts `-h` and `--help`. There is no model-selection flag here: analyze the compatible completed results in the selected root. A standalone `--mode sanity` or `--mode pilot` is not supported.

### `doctor`: every option

<!-- cli-options: doctor -->

| Option and value | Default | Meaning and example |
|---|---|---|
| `--cpu` | Off; a usable GPU is required. | Validate the preparation/analysis imports without requiring or testing CUDA kernels. Example: `./run.sh doctor --cpu`. Does not enable CPU training. |
| `--optimizer OPTIMIZER` | `adamw8bit` | Choose `adamw8bit` or `adamw`. The CUDA path tests bf16 matrix operations; `adamw8bit` additionally exercises a real bitsandbytes update. `adamw` omits that 8-bit-kernel requirement. Example: `./run.sh doctor --optimizer adamw`. This is a diagnostic selector, not the `run` optimizer-policy option. |

Also accepts `-h` and `--help`. GPU checks require exactly one visible GPU and native bf16 support (Ampere or newer); a T4 is not supported by the training policy.

### `list-models`: every option

<!-- cli-options: list-models -->

Only `-h` and `--help` are accepted. `./run.sh list-models` prints all fixed registry entries, licenses, project LRs, LoRA head ratios, default ranks, and default seeds. It accepts no model filter or output-directory option.

## 12. Related launcher options and environment variables

These belong to other launchers or the shell; do not pass them to `run.sh` as additional CLI flags.

| Launcher or variable | Options/default | Purpose |
|---|---|---|
| `./setup.sh` | `--cpu`: off; `--prefetch`: off; `-h` / `--help` for help. | `--cpu` installs CPU wheels for preparation/analysis; `--prefetch` additionally downloads pinned artifacts after setup and needs any required model access. |
| `./run_parallel.sh GPU_IDS [RUN_OPTIONS]` | Required comma-separated GPU indices, such as `0,1`; forwards run options. | Starts one worker per listed GPU, assigning `--num-shards` and `--shard-index` itself. Do not supply another `run` word or your own shard options. |
| `CUDA_VISIBLE_DEVICES` | Launcher default: `0`. | Choose one GPU per worker. Example: `CUDA_VISIBLE_DEVICES=1 ./run.sh doctor`. |
| `OMP_NUM_THREADS` | Launcher default: `4`. | CPU thread baseline. `intruders --threads` separately sets its numerical thread limits. |
| `TOKENIZERS_PARALLELISM` | Set to `false` by `run.sh`. | Disables tokenizer process/thread parallelism for the launched command. |
| `CUBLAS_WORKSPACE_CONFIG` | Set to `:4096:8` by `run.sh`. | cuBLAS workspace policy used with reproducible execution; does not itself enable strict deterministic algorithms. |
| `HF_TOKEN` | Unset unless supplied or authentication is cached. | Authorizes gated/private Hugging Face downloads. Do not commit its value. |
| `HF_HOME` | Hugging Face library default unless exported. | Selects the external model cache location; it is not the experiment's `--output` directory. |
| `WORKER_LOG_DIR` | `logs/parallel`. | Overrides the parallel launcher's worker-log directory. |
| `PYTHON` | Automatically chosen by `setup.sh` unless supplied. | A supported Python interpreter for setup, such as `PYTHON=python3.11 ./setup.sh`. It is a setup variable; `run.sh` launches its activated environment's `python`. |

## 13. Identify learning rates with `--test-lr`

Run this **before the first main/pilot launch** in a fresh output directory:

```bash
./run.sh plan --test-lr --output results-v3
./run.sh --test-lr --output results-v3 --hourly-rate 0.40
./run.sh plan --output results-v3
./run.sh run --mode pilot --models qwen2.5-3b --output results-v3
./run.sh run --output results-v3
```

`0.40` is an example hourly rate; replace it with the actual rental quote. Calibration requires the same memory policy as training, including enough VRAM for the largest full-FT baseline. Largest models and full FT are tested first to expose fit problems early. Calibration ends after selecting defaults; it does not automatically start the 120-sequence main study.

The bounded search follows this declared procedure:

1. For each selected model and track, test `{0.5,1,2}` times its original methodology prior, with classifier-head LR multipliers `{1,5}`. The [registered grids](PROTOCOL_V3.md#evidence-and-candidate-grid) retain their original centers after the project default promotion. Use rank 16 for LoRA, three epochs, and seed 42. Screening trains on 1,992 RTE training examples and evaluates the disjoint, stratified 498-example training holdout.
2. Keep the best head multiplier for each LR, then the two best **distinct rates**. Screening orders accuracy first, then lower NLL, lower LR and lower head multiplier.
3. Train each finalist from the pinned base through RTE → MNLI → SST-2 at seeds 42 and 123. Each task uses its own fixed 80/20 split inside its 2,490 training examples. Evaluate all previously learned tasks after each boundary. The official validation sets never participate in selection.
4. A candidate is eligible only when its mean diagonal accuracy exceeds the holdout majority baseline on **every task**. Select the highest mean final-row accuracy across tasks and seeds; ties use higher mean diagonal, lower mean final NLL, lower LR, then lower head multiplier. BWT, individual seed scores and sample SD remain visible.
5. Diagnose the chosen LoRA setting at ranks 4 and 64 on the RTE holdout using seed 42. Underlearning is reported rather than deleting ranks; numerical instability prevents activation. The chosen LR/head policy remains common across main LoRA ranks.
6. Activate defaults only after the entire requested scope completes and its charts/report are saved. A budget stop, incomplete run, infrastructure failure or missing eligible candidate leaves defaults unmodified. These are the best tested settings under this selection rule, with no guarantee of a global or every-task optimum. Search-boundary warnings identify winners at the edge of the tested LR grid.

The default all-four-model, both-track search has **48 screening runs + 32 three-task confirmation runs + 8 endpoint checks = at most 88 calibration runs**, or 57,000 optimizer updates. That is about 34% of the main grid's update count, **not** a wall-time or dollar percentage: full FT has different costs. Calibration exports predictions, metrics and step logs, and temporary recovery states while incomplete; it omits large permanent adapter/projection/full-weight exports and spectral decomposition.

Accepted option interactions:

| Option | Meaning during `--test-lr` |
|---|---|
| `--models`, `--track` | Select the calibration scope; omitted means all four models and both tracks. Defaults are activated atomically for the requested scope. |
| `--lora-lrs`, `--full-lrs` | Replace that model/track's three-point candidate grid; require at least two distinct positive finite rates. Use the same global or model-scoped syntax as normal runs. |
| `--head-lr-multiplier` | Fix one head ratio instead of searching `{1,5}`. |
| `--seeds` | Confirmation seeds; default **42 and 123** here. Require at least two distinct valid seeds. The first is used for screening and endpoint checks. Normal main defaults remain five seeds. |
| `--ranks` | Rejected: selection uses rank 16 and endpoint diagnostics use ranks 4/64. |
| `--mode` | Must remain `main` (default); calibration is isolated automatically, not a main run. |
| `--micro-batch`, `--accum`, `--eval-batch`, `--optimizer`, `--deterministic` | Apply to calibration and must match the later main cohort. |
| `--budget`, `--reserve`, `--hourly-rate`, `--max-hours` | Same per-invocation admission limits; finish an admitted case before stopping. An unmeasured case may overrun. They do not stop the rental, and a later main invocation has its own allowance. |
| `--force` | Rerun calibration cases instead of reusing completed/recorded numerically unstable cases. Existing main controls still prevent recalibration in that root. |
| `--keep-going` | Record infrastructure failures and attempt remaining cases. Any such failure prevents activation; rerun to retry failed cases. Numerical divergence is recorded separately as an unstable candidate. |
| `--dry-run` | Print the calibration plan and exit. Also available as `plan --test-lr`. |
| `--num-shards`, `--shard-index` | Must be `1` and `0`: calibration uses one worker. Run the main study across GPUs after calibration. |
| `--save-full-weights` | No permanent calibration weight exports; this switch applies to main runs. Temporary recovery states are still retained until each calibration sequence completes. |
| `--output` | Cohort root for defaults; evidence goes in its `lr_test/` child. |

Reviewable artifacts:

```text
results-v3/
  lr_defaults.json                       # selected LR/head ratio for every calibrated model/track
  lr_test/
    latest.json                          # relative pointer to the latest report
    protocol.json, software.json          # calibration controls/environment
    data/calibration/data_selection_*.json # explicit disjoint train/holdout source indices
    batches/*.json                       # invocation timing/cost records
    runs/<run_id>/
      result.json                        # A, NLL, BWT, data provenance, flags and task summaries
      train_log.jsonl                    # step loss, adapter/backbone LR, head LR, gradient norm
      checkpoints/task*/predictions_*.npz
    experiments/<calibration_id>/
      plan.json
      report.json                        # every trial, finalist, seed score, winner and endpoint check
      report.md
      charts/screen_*.png                 # LR/head-rate accuracy and NLL curves
      charts/confirmation_*.png           # finalist scores/seed SD and learning-retention scatter
      charts/selected_matrix_*.png        # full selected-candidate accuracy matrix
      charts/selected_loss_*.png          # selected-candidate training curves
```

Re-running the same `--test-lr` resumes its completed cases. Repeats of an activated selection write a new report identity, preserving the previous active evidence until replacement succeeds. `./run.sh analyze --output results-v3` can regenerate the latest calibration charts on CPU, even before main results exist. Calibration cases are always excluded from main aggregation, completeness counts and spectral passes.

New `--test-lr` selections are **local to the output cohort**, so subsequent `plan`, `run` and parallel workers using that `--output` load them automatically. The 18 October 2026 LoRA policies were separately promoted to the project defaults in Section 3 at the user's request. Preserve `lr_defaults.json` together with `lr_test/`; matching completed evidence and code/training controls are validated. Explicit main LR lists are sensitivity settings; they do not change the declared baseline anchor. Defaults cannot be retuned after main/pilot controls are written. Use a new output root for a new calibration policy or scientific/numerical cohort.

## 14. New measurement outputs

`analyze` retains the original methodology window and additionally writes:

| Output under `aggregates/` | Content |
|---|---|
| `per_task_learning_retention.csv/json` | Every diagonal, final accuracy and per-task retention change. |
| `task_transition_changes.csv/json` | Each earlier task's change at every successive boundary. |
| `all_task_feasible_sets.json` | Per-task learning/retention constraints, 1/2/5pp tolerance sets, health/completeness checks and paired simultaneous intervals. |
| `all_task_feasible_sets_summary.csv/json` | Compact gate membership and interval decisions. |
| `spectral_diagnostics_long.csv/json` | Per-matrix relative update size, spectral norm, stable/entropy effective rank, 90% energy rank and leading-subspace distances/angles/gaps at k=10 and 20. |

The 2pp supplementary set requires all three mean diagonals within 2pp of the frozen full-FT reference and both earlier-task final-minus-diagonal changes at least −2pp. It requires all five ranks/seeds, a complete baseline, and a baseline that beats the majority accuracy on every task. The original SD-dependent window remains separately available. Intervals use paired seed differences and a Bonferroni 25-constraint family within each model/LR; they assume independent approximately normal seed differences and may remain inconclusive.

Run `intruders` first to calculate new spectral measurements from boundary snapshots on CPU. LoRA update spectra use thin QR and an r×r SVD; full-FT update spectra need dense CPU decompositions. Effective rank uses singular-value probabilities, stable rank uses squared energy, and zero updates are explicitly flagged. Leading-subspace distance compares the leading reference/tuned spans, while the original intruder detector continues to compare individual tuned vectors with all base vectors. Cutoff relative gaps below `1e-3` are flagged as near ties. Charts include per-task gates, update/capacity/subspace curves and k=10/20 sensitivity; matrices are averaged within each run before seed variability is displayed.

## 15. Resume the cloud result download

For the completed 512/1024 cohort, restart interrupted retrieval from the
repository root with:

```bash
python3 -m tools.watch_large_ranks --resume-download
```

The default target is `Finetune-result/large-ranks`; `--root PATH` changes it.
This resumes saved partial archives, skips verified ones, manages the CPU
download endpoint and spending guard, and starts no GPU training. If a
downloader is already active, it reports that process and leaves it running.
Keep the result directory and its `.download`/`.aria2` files. Progress appends
to `execution/download-progress.log` under that target every ten minutes.

The [Modal delivery reference](MODAL_EXECUTION.md#finished-cohort-segmented-delivery)
documents all downloader options, saved settings and direct resumption while
the endpoint is active. The [extension reference](LARGE_RANK_EXTENSION.md#cloud-execution-and-retrieval)
documents all watcher options. `run.sh` continues to control suite training
and analysis; these Python commands retrieve existing cloud results.
