# CLI reference

Use `./run.sh --help` for commands and `./run.sh run --help` for all run options. See [the runbook](docs/CLOUD_RUNBOOK.md) for the complete workflow.

| Command | Purpose |
|---|---|
| `plan` | Print exact grid counts, revisions and measured cost coverage; works without ML dependencies. |
| `prepare` | Prefetch pinned weights, tokenizers and tokenized GLUE on CPU or GPU host. |
| `run` | Train/evaluate/export; default is the complete main design. |
| `intruders` | Resumable CPU spectra from saved snapshots. |
| `analyze` | CSV/JSON aggregates, figures, report and completeness audit. |
| `doctor` | Dependency imports and actual CUDA/bf16/8-bit optimizer test. |
| `doctor --cpu` | Validate the CPU preparation/analysis stack. |
| `list-models` | Fixed registry and nominal LRs. |

## Run selection

```bash
./run.sh run --models qwen2.5-1.5b --track both
./run.sh run --models llama3.2-1b qwen2.5-1.5b --ranks 4 8 16 32 64 --seeds 42 123 456 789 1024
./run.sh run --dry-run
```

The four registry keys are `llama3.2-1b`, `qwen2.5-1.5b`, `granite3.3-2b`, `qwen2.5-3b`. Their exact Hugging Face IDs are also accepted; other models are rejected. Selecting a subset is useful for scheduling; the report retains the missing-design audit.

`--mode main` is the default. `pilot` selects r=16/full FT, seed 42 and keeps full main training/evaluation. `smoke` uses 64 examples, one epoch, evaluation capped at 32/task in `<output>/smoke/`. `sanity` uses the prescribed RTE LR comparison battery in `<output>/sanity/`, with a training-only calibration split. Diagnostic runs never enter main aggregates.

## Numerical controls

`--micro-batch` must divide 16; `--accum` is inferred or must give exactly 16 effective examples. Default is 4×4. `--eval-batch` defaults to 32. Changing either partition requires a new output cohort. No automatic OOM fallback changes the numerical setting for one rank. `--deterministic` requests strict deterministic kernels; use the same setting for every invocation of a cohort.

`--optimizer standard` uses fp32 AdamW moments for LoRA and 8-bit moments for full-FT dense matrices, with fp32-state heads/vectors. This keeps optimizer quantization out of the primary rank sweep. `--optimizer adamw` creates a separate higher-memory uniform fp32-state robustness cohort. `--save-full-weights` additionally retains each full backbone boundary; exact q/v snapshots are always retained regardless of that flag.

LR overrides support either a common list or model-specific lists:

```bash
./run.sh plan --lora-lrs qwen2.5-1.5b=1e-4,1.85e-4
./run.sh plan --full-lrs 1e-5 2e-5
```

These create explicit sensitivity configurations with separate IDs and summaries. Keep the LR fixed across ranks for a given primary comparison. The feasible-window baseline is the declared nominal full-FT LR, never the highest-accuracy LR selected after seeing outcomes. The `sanity` mode uses its preset nominal comparison rates.

## Scheduling and recovery

`--output results` selects the experiment directory. `--force` explicitly replaces completed selected runs. By default a failure stops further paid work; `--keep-going` is available for diagnostic use. Completed results are skipped, and incomplete sequences resume from the last completed task with the same controls. No within-task optimizer checkpoint is promised.

`--num-shards N --shard-index I` distributes complete model/seed blocks, with one visible GPU per process. On a homogeneous multi-GPU host, `./run_parallel.sh 0,1 [run options]` manages the workers. Expose one GPU manually with `CUDA_VISIBLE_DEVICES=1 ./run.sh run ...` when needed. Shard counts and selected models do not redefine the numerical protocol.

`--hourly-rate`, `--budget`, `--reserve` and `--max-hours` are optional scheduling inputs. Their limits are per invocation and finish whole runs; they are not billing caps. Exit codes are 0 for finished selected work, 1 for run failures, 2 for invalid configuration/environment, 3 for deferred selected work, and 130 for interruption.

## Spectral analysis and artifacts

```bash
./run.sh intruders --output results --threads 4
./run.sh intruders --output results --solver svd --threads 4
./run.sh analyze --output results
```

The default eigensolver and direct-SVD cross-check have separate caches. Running a different solver replaces each run's active spectral summary; use a complete solver pass before interpreting correlations. Analysis requires no model weights from Hugging Face once `results/cache/projections/` and the boundary artifacts are present. Copy the complete output tree when moving machines.

```
results/
  protocol.json                 # immutable controls + source hash
  software.json                 # immutable software identity
  hardware/<model>.json         # one GPU type per model
  invocations/*.json             # environment + exact package freeze per launch
  batches/*.json                # worker outcomes and elapsed-cost estimates
  data/main/                    # immutable sample selection + tokenizer provenance
  runs/<run_id>/
    result.json                 # completion marker, A, metrics, provenance
    train_log.jsonl             # optimizer-step loss/LR/gradient norm
    resume.pt                   # only while a sequence is incomplete
    checkpoints/task*/
      projections.pt            # exact factors or fp32 full-FT projections
      heads.pt
      predictions_*.npz
      adapter_*                 # LoRA only
      spectra/<analysis-id>/matrices.json
  cache/projections/            # pretrained q/v weights required for CPU spectra
  cache/spectra-v2/              # reusable full-precision pretrained vectors
  aggregates/                   # tables and completeness audit
  analysis/                     # figures + report.md
  smoke/                        # isolated diagnostic cohort
  sanity/                       # isolated LR sensitivity cohort
```

`notify.py` and `run_notify.sh` are legacy optional user-configured notification utilities. Core setup and execution do not invoke them or contact messaging services.
