# Using `run.sh`

`run.sh` launches the Feasible Rank Window suite. It changes to the repository directory, activates `.venv` when present, and forwards arguments to `rank_suite.cli`. With no arguments it prints help. Training starts when you select `run`.

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
| `--models MODEL ...` | All four fixed models. | Registry keys or exact checkpoint IDs from Section 3, separated by spaces. Duplicates are removed. Other checkpoints are rejected. Example: `--models llama3.2-1b qwen2.5-3b`. |
| `--output PATH` | `results` | Experiment directory. Relative paths resolve from the repository directory because the launcher changes there. Absolute paths are accepted, for example `--output /workspace/fyp-results`. Main/pilot use it directly; smoke/sanity append their own subdirectory. |
| `--track TRACK` | `both` | Choose `lora`, `full`, or `both`. `full` means all non-embedding backbone weights plus the current task head; token embeddings stay frozen. Ranks apply to LoRA only. |
| `--mode MODE` | `main` | Choose `main`, `pilot`, `sanity`, or `smoke`. Pilot/smoke force rank 16 and seed 42. Sanity additionally forces its preset LR comparisons and RTE training holdout. See Section 2 for data, epoch, and output differences. |
| `--ranks RANK ...` | `4 8 16 32 64` | Positive integers, separated by spaces or commas; duplicates are removed. A rank cannot exceed the smallest `v_proj` output width among selected models: 256 when either Qwen is selected, 512 for a Llama/Granite-only selection. Other valid ranks are sensitivity experiments; the main window requires the prescribed grid. Pilot/smoke/sanity use rank 16 instead. |
| `--seeds SEED ...` | `42 123 456 789 1024` | Integers from `0` through `4294967295`, separated by spaces or commas; main-grid duplicates are removed. Pilot/smoke/sanity use seed 42 instead. The main window requires the five prescribed seeds. |
| `--lora-lrs RATE ...` | Nominal per-model LoRA rates in Section 3. | Finite positive rates, using decimal or scientific notation. Supply a common list (`--lora-lrs 1e-4 2e-4`) or scoped lists (`--lora-lrs qwen2.5-1.5b=1e-4,1.85e-4`). Scoped names must be selected registry keys, even if `--models` uses exact checkpoint IDs. Sanity uses its preset comparisons instead. |
| `--full-lrs RATE ...` | Nominal per-model full-FT rates in Section 3. | Same syntax and positivity rules as `--lora-lrs`, for the full track. A model's nominal rate must be present to provide the required window baseline. Example: `--full-lrs qwen2.5-1.5b=1e-5,1.85e-5`. Sanity uses nominal and twice-nominal rates instead. |
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
| `--output PATH` | `results` | Exact results root to read, including its runs and projection cache. Unlike `run --mode smoke`, this command does not append a mode directory. Use `--output results/smoke` for smoke artifacts. |
| `--threads N` | `4` | Positive integer CPU thread count used by Torch and the BLAS thread limiter. Example: `--threads 8`. This is not a GPU count. |
| `--solver SOLVER` | `eigh` | Choose `eigh` (float64 Gram eigendecomposition for leading tuned vectors) or `svd` (direct fp32 SVD cross-check). Caches are separated, but active run summaries are replaced; complete a consistent pass. Example: `./run.sh intruders --output results --threads 4 --solver svd`. |

Also accepts `-h` and `--help`. Spectral analysis is CPU-based; there is no `--cpu`, `--mode`, `--force`, rank, epsilon, or top-k CLI option for this command. Protocol constants govern the latter measurements.

### `analyze`: every option

<!-- cli-options: analyze -->

| Option and value | Default | Meaning and example |
|---|---|---|
| `--output PATH` | `results` | Exact results root for aggregates and figures. Does not append a mode directory. Example: `./run.sh analyze --output results`. |
| `--mode MODE` | `main` | Only `main` or `smoke` are accepted. Filters results by their stored mode; a pilot is stored as a reusable main configuration. Use `./run.sh analyze --output results/smoke --mode smoke` for smoke reports. Main-root analysis also reads available sanity results from its `sanity/` child. |

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

Only `-h` and `--help` are accepted. `./run.sh list-models` prints all fixed registry entries, licenses, nominal LRs, default ranks, and default seeds. It accepts no model filter or output-directory option.

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
