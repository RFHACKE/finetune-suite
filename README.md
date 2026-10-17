# Feasible Rank Window — experiment suite

A reproducible study of LoRA rank and sequential classification forgetting in four **fixed base models**: Llama-3.2-1B, Qwen2.5-1.5B, Granite-3.3-2B, and Qwen2.5-3B. The original proposal and subsequent methodology were reviewed against primary literature. Model selection is unchanged; protocol corrections are explicit in [the v2 amendment](docs/PROTOCOL_V2.md).

The main study is **120 full sequences**: four models × (five LoRA ranks + one non-embedding full-FT baseline) × five seeds. Every sequence learns RTE → MNLI Matched → SST-2 with 2,490 training examples/task, three epochs/task, effective batch 16 and full official validation. All ranks, seeds and evaluation data are preserved for cost efficiency.

## Start on a rented GPU

Use a Linux CUDA image with a working NVIDIA driver >=570. A 32 GB RTX 5090 is a practical starting point for all four models; a 48 GB GPU provides more margin. The largest full-FT runs have a static memory floor above 24 GB, so a 4090 is not a safe all-model assumption. See [the cloud runbook](docs/CLOUD_RUNBOOK.md) for memory, disk, budget and multi-GPU guidance.

```bash
git clone https://github.com/RFHACKE/finetune-suite.git fyp-suite
cd fyp-suite
./setup.sh

# Llama requires prior license acceptance and Hugging Face authentication:
.venv/bin/hf auth login  # alternatively set HF_TOKEN in your environment

./run.sh plan
./run.sh run --mode smoke --models qwen2.5-1.5b
./run.sh run --mode pilot --models qwen2.5-3b
./run.sh run --mode pilot
./run.sh plan --hourly-rate 0.40  # example rate; enter the actual quote
./run.sh run --hourly-rate 0.40
```

A pilot is a full-length main-study r=16/full-FT sequence at seed 42, so it is reused automatically. Smoke runs are tiny diagnostic runs saved separately. The optional `run --mode sanity` executes fifteen unique RTE LR-sensitivity checks on a training-only holdout.

The installer creates a local virtualenv, finds or provisions Python 3.10–3.12, installs pinned torch 2.8.0/CUDA 12.8 and dependencies, and tests real bf16 and optimizer kernels. CUDA wheels include the runtime; no host driver reinstall or local CUDA compiler is needed. Use `./setup.sh --cpu` for a preparation/analysis machine, or `./setup.sh --prefetch` to download pinned artifacts as part of setup.

## What saves money

Spectral decomposition and plotting run **after GPU training, on CPU**. Training exports the exact projection weights or LoRA factors at every task boundary. Completed runs are skipped before model loading, and interrupted sequences recover after the last completed task. Cached datasets and pinned model files avoid repeated downloads.

The suite does not promise a $25 total. `plan` reports unknown cost until measured pilots cover each model/track, then estimates the remaining work with a planning allowance. Optional `--budget 25 --reserve 5 --hourly-rate <actual-rate>` and `--max-hours` control admission of whole runs. They can overrun by an unmeasured run and **do not stop the rented instance**. Multiple identical GPUs can process independent paired seed blocks with `./run_parallel.sh 0,1 --output results`.

## Finish on CPU

Preserve the complete results directory before ending the rental, including its base projection cache. On your CPU machine:

```bash
./setup.sh --cpu
./run.sh intruders --output results --threads 4
./run.sh analyze --output results
```

For the separate smoke test, use `--output results/smoke` for both commands and `analyze --mode smoke`. Spectra resume per matrix; they compare the top ten tuned left singular vectors with **all** pretrained left singular vectors, with epsilon sensitivity and checksums.

Read `results/aggregates/completeness.json` and `results/analysis/report.md`. Outputs include per-seed accuracy matrices, BWT, peak forgetting, sample SD, paired rank comparisons, feasible rank sets, fixed-2pp sensitivity, and spectral summaries. Missing seeds, baselines or ranks prevent window certification. Nonzero full-FT intruders are reported as findings, not forced to zero.

## Numerical and reproducibility choices

Trainable parameters are fp32 with bf16 autocast computation; normalization storage is fp32 in both tracks. LoRA uses fp32 AdamW moments. Full FT uses 8-bit moments for dense matrices and fp32 moments for heads and vectors; a uniform fp32-state robustness cohort is available. Model weights are never quantized. LoRA targets q/v, alpha=2r, dropout=0. The nominal LoRA LRs are 2.00e-4 / 1.85e-4 / 1.60e-4 / 1.50e-4, with full-FT rates ten times smaller. These are declared starting priors, not claimed empirical optima.

Model/tokenizer/data revisions, protocol, source hashes, package freeze and hardware are recorded. Changed scientific controls or software require a new results directory. Each model's rank/baseline comparisons must use the same GPU type. Legacy v1 results are not pooled with v2.

## Research notes and validation

- [Expanded literature review (Word)](literature-review.docx)
- [Research improvements and candidate hyperparameters](docs/RESEARCH_IMPROVEMENTS.md)
- [Literature review and corrections](docs/LITERATURE_REVIEW.md)
- [Executable protocol amendment](docs/PROTOCOL_V2.md)
- [Cloud and budget runbook](docs/CLOUD_RUNBOOK.md)
- [Validation record and measured CPU spectral timings](docs/VALIDATION.md)
- [CLI usage](docs/usage.md)

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check rank_suite tests
```

Tests exercise real tiny Llama/Qwen/Granite architectures, accumulation including unequal final micro-batches, checkpoint recovery, SVD/eigensolver agreement, partial-study reporting, pairing and cost calculations. Production-size GPU timing and VRAM remain acceptance tests for your rented host. No scientific experiment results are bundled or fabricated.
