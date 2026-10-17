# Feasible Rank Window — experiment suite

A reproducible study of LoRA rank and sequential classification forgetting in four **fixed base models**: Llama-3.2-1B, Qwen2.5-1.5B, Granite-3.3-2B, and Qwen2.5-3B. The original proposal and subsequent methodology were reviewed against primary literature. Model selection is unchanged; the current [v3 amendment](docs/PROTOCOL_V3.md) adds LR/head calibration, per-task learning/retention gates and update/subspace diagnostics to the [v2 design](docs/PROTOCOL_V2.md).

The main study is **120 full sequences**: four models × (five LoRA ranks + one non-embedding full-FT baseline) × five seeds. Every sequence learns RTE → MNLI Matched → SST-2 with 2,490 training examples/task, three epochs/task, effective batch 16 and full official validation. All ranks, seeds and evaluation data are preserved for cost efficiency.

## Start on a rented GPU

Use a Linux CUDA image with a working NVIDIA driver >=570. A 32 GB RTX 5090 is a practical starting point for all four models; a 48 GB GPU provides more margin. The largest full-FT runs have a static memory floor above 24 GB, so a 4090 is not a safe all-model assumption. See [the cloud runbook](docs/CLOUD_RUNBOOK.md) for memory, disk, budget and multi-GPU guidance.

```bash
git clone https://github.com/RFHACKE/finetune-suite.git fyp-suite
cd fyp-suite
./setup.sh

# Llama requires prior license acceptance and Hugging Face authentication:
.venv/bin/hf auth login  # alternatively set HF_TOKEN in your environment

./run.sh run --mode smoke --models qwen2.5-1.5b --output results-check
./run.sh plan --test-lr --output results-v3
./run.sh --test-lr --output results-v3  # calibrates all four models, saves defaults, then exits
./run.sh plan --output results-v3     # shows selected LR/head defaults
./run.sh run --mode pilot --models qwen2.5-3b --output results-v3
./run.sh run --mode pilot --output results-v3
./run.sh plan --output results-v3 --hourly-rate 0.40  # example rate; enter the actual quote
./run.sh run --output results-v3 --hourly-rate 0.40
```

A pilot is a full-length main-study r=16/full-FT sequence at seed 42, so it is reused automatically. Smoke runs are tiny diagnostic runs saved separately. The optional `run --mode sanity` executes fifteen unique RTE LR-sensitivity checks on a training-only holdout.

`--test-lr` screens three rates × two head multipliers on an RTE training holdout, confirms the two strongest distinct rates on the full three-task sequence at two seeds, and checks the chosen LoRA setting at ranks 4/64. It tunes LoRA/full FT separately and never selects from official validation. JSON, charts and step logs remain under `results-v3/lr_test/`; complete choices become automatic defaults in `results-v3/lr_defaults.json`. Run calibration before any main/pilot launch. The default search has up to 88 calibration runs/57,000 updates; production time and cost must be measured. Use a fresh directory for v3; existing v2 cohorts cannot be reused.

The installer creates a local virtualenv, finds or provisions Python 3.10–3.12, installs pinned torch 2.8.0/CUDA 12.8 and dependencies, and tests real bf16 and optimizer kernels. CUDA wheels include the runtime; no host driver reinstall or local CUDA compiler is needed. Use `./setup.sh --cpu` for a preparation/analysis machine, or `./setup.sh --prefetch` to download pinned artifacts as part of setup.

## What saves money

Spectral decomposition and plotting run **after GPU training, on CPU**. Training exports the exact projection weights or LoRA factors at every task boundary. Completed runs are skipped before model loading, and interrupted sequences recover after the last completed task. Cached datasets and pinned model files avoid repeated downloads.

The suite does not promise a $25 total. `plan` reports unknown cost until measured pilots cover each model/track, then estimates the remaining work with a planning allowance. Optional `--budget 25 --reserve 5 --hourly-rate <actual-rate>` and `--max-hours` control admission of whole runs. They can overrun by an unmeasured run and **do not stop the rented instance**. Multiple identical GPUs can process independent paired seed blocks with `./run_parallel.sh 0,1 --output results-v3`.

## Finish on CPU

Preserve the complete results directory before ending the rental, including calibration evidence and the base projection cache. On your CPU machine:

```bash
./setup.sh --cpu
./run.sh intruders --output results-v3 --threads 4
./run.sh analyze --output results-v3
```

For the separate smoke test, use `--output results-check/smoke` for both commands and `analyze --mode smoke`. Spectra resume per matrix; they compare the top ten tuned left singular vectors with **all** pretrained left singular vectors, with epsilon sensitivity and checksums.

Read `results-v3/aggregates/completeness.json` and `results-v3/analysis/report.md`. Outputs include per-seed accuracy matrices, all diagonals/transitions, BWT, peak forgetting, sample SD, paired rank comparisons, original and all-task feasible sets, fixed-tolerance sensitivity, update norms/effective rank, and leading-subspace distances/angles/gap flags. Missing seeds, baselines or ranks prevent window certification. Nonzero full-FT intruders are reported as findings, not forced to zero.

## Numerical and reproducibility choices

Trainable parameters are fp32 with bf16 autocast computation; normalization storage is fp32 in both tracks. LoRA uses fp32 AdamW moments. Full FT uses 8-bit moments for dense matrices and fp32 moments for heads and vectors; a uniform fp32-state robustness cohort is available. Model weights are never quantized. LoRA targets q/v, alpha=2r, dropout=0.

The completed 48-case Modal calibration supplies the project LoRA defaults:

| Model | Adapter LR | Head LR multiplier |
| --- | ---: | ---: |
| Llama-3.2-1B | `4.00e-4` | `5` |
| Qwen2.5-1.5B | `1.85e-4` | `1` |
| Granite-3.3-2B | `3.20e-4` | `5` |
| Qwen2.5-3B | `3.00e-4` | `5` |

These are the best tested training-holdout policies; three models selected the upper tested LR boundary. They apply automatically in fresh output directories. See [calibration evidence, limitations and precedence](docs/LORA_DEFAULTS.md). The measured LoRA partition was micro-batch 16/accumulation 1; pass those values explicitly to reproduce it. Full-FT priors remain `2.00e-5 / 1.85e-5 / 1.60e-5 / 1.50e-5`. A valid cohort-local `--test-lr` result overrides the project defaults, and explicit CLI choices override the resolved rates/head ratios. The registered calibration and sanity candidate grids retain their original methodology priors.

Model/tokenizer/data revisions, protocol, source hashes, package freeze and hardware are recorded. Changed scientific controls or software require a new results directory. Each model's rank/baseline comparisons must use the same GPU type. Older protocol cohorts are not pooled with v3.

## Research notes and validation

- [Expanded literature review](docs/literature-review.md)
- [Research improvements and candidate hyperparameters](docs/RESEARCH_IMPROVEMENTS.md)
- [Literature review and corrections](docs/LITERATURE_REVIEW.md)
- [Executable protocol amendment](docs/PROTOCOL_V2.md)
- [Current v3 protocol and LR selection](docs/PROTOCOL_V3.md)
- [Cloud and budget runbook](docs/CLOUD_RUNBOOK.md)
- [Modal execution and lossless retrieval](docs/MODAL_EXECUTION.md)
- [Calibrated project LoRA defaults and their evidence](docs/LORA_DEFAULTS.md)
- [Unattended run, remote failure reports and morning recovery](docs/OVERNIGHT.md)
- [Validation record and measured CPU spectral timings](docs/VALIDATION.md)
- [CLI usage](docs/usage.md)

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check rank_suite tests
```

Tests exercise real tiny Llama/Qwen/Granite architectures, accumulation including unequal final micro-batches, checkpoint recovery, SVD/eigensolver agreement, partial-study reporting, pairing and cost calculations. Production-size GPU timing and VRAM remain acceptance tests for your rented host. No scientific experiment results are bundled or fabricated.
