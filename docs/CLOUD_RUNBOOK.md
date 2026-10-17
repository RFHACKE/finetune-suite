# Running on rented GPUs without wasting the credit

## Choose hardware from memory needs and a measured pilot

For the entire default experiment on one machine, start by comparing **one [RTX 5090 32 GB](https://www.nvidia.com/en-gb/geforce/graphics-cards/50-series/rtx-5090/)** with a competitively priced **48 GB GPU** (for example an [L40](https://www.nvidia.com/en-us/data-center/l40/) or RTX A6000). A 4090 24 GB is useful for LoRA and the smaller full-FT models, but do not rent it assuming every fp32-storage full-FT model will fit. Two independent 24 GB GPUs do not provide a 48 GB address space for one run in this suite.

These are approximate **static lower bounds**, using the methodology's rounded non-embedding counts and the default 8-bit-state optimizer. They exclude activations, CUDA context, workspaces and temporary tensors. Actual peak memory is logged by the pilot.

| Model | Full-FT weights + gradients + moments + frozen embeddings, GiB |
|---|---:|
| Llama-3.2-1B | 9.5 |
| Qwen2.5-1.5B | 12.6 |
| Granite-3.3-2B | 22.8 |
| Qwen2.5-3B | 26.4 |

32 GB is a practical target for the largest model with checkpointing, micro-batch 4 and disabled autocast weight caching, **not a measured fit guarantee**. If its pilot OOMs, use micro-batch 2 and accumulation 8 in a **new output directory for the whole cohort**, or choose 48 GB. If the parameter-memory check fails, a smaller micro-batch cannot solve it. `--optimizer adamw` needs more memory than the table.

Prefer at least 32 GB system RAM (64 GB is comfortable), a reasonably fast CPU, and **150 GB local disk** for all four checkpoints, datasets, environment, adapters, exact full-FT projection snapshots, temporary atomic checkpoints and outputs. The default full-FT projection boundaries alone total roughly 32 GB decimal over the study. A temporary full-model resume file can exceed 11 GB; replacing it briefly needs space for old and new copies. Multiple simultaneous full-FT workers increase this requirement. `--save-full-weights` retains much more and is unnecessary for the primary spectral analysis.

Use an Ubuntu/PyTorch image with an NVIDIA driver exposing CUDA to the container. The setup policy requires driver major >=570 and installs torch 2.8.0 CUDA 12.8 wheels on both Ada and Blackwell. [PyTorch publishes this wheel combination](https://pytorch.org/get-started/previous-versions/), and [bitsandbytes' installation guide](https://huggingface.co/docs/bitsandbytes/v0.48.2/en/installation) documents supported GPU/CUDA builds. The script tests actual bf16 matmul and 8-bit optimizer kernels. It does not reinstall the host's driver or require a CUDA compiler.

If the image includes uv, setup uses concurrent package downloads; otherwise it uses pip. Preserve the installer cache on persistent storage when possible. Slow links may need a longer `UV_HTTP_TIMEOUT`; package download time is separate from experimental throughput.

## Before starting the rental

Accept Llama's model terms on [its model page](https://huggingface.co/meta-llama/Llama-3.2-1B) and obtain an HF read token with access. Put the token in the instance's environment as `HF_TOKEN`, or authenticate with `.venv/bin/hf auth login`. The script never embeds tokens into generated Python or records them in manifests.

You can prepare downloads on a local Linux CPU machine:

```bash
git clone https://github.com/RFHACKE/finetune-suite.git fyp-suite
cd fyp-suite
./setup.sh --cpu
HF_HOME=/path/to/portable-hf-cache ./run.sh prepare
```

Copy that cache to persistent storage on the rental and set the same `HF_HOME` path there. Copy model/data caches rather than a virtualenv from a different machine. If prefetching locally is inconvenient, `./setup.sh --prefetch` performs the downloads on the rental. Download and installation time is billable there; cached files avoid doing it twice after interruption.

## One GPU: recommended progression

```bash
git clone https://github.com/RFHACKE/finetune-suite.git fyp-suite
cd fyp-suite
./setup.sh
./run.sh plan

# Short software checks. Separate results/smoke/; no main-study conclusions.
./run.sh run --mode smoke --models qwen2.5-1.5b
./run.sh intruders --output results/smoke
./run.sh analyze --output results/smoke --mode smoke

# Check the largest model's actual memory and speed first. Full-length main runs.
./run.sh run --mode pilot --models qwen2.5-3b

# Complete the eight-run pilot coverage, skipping completed main configurations.
./run.sh run --mode pilot

# Optional methodology LR-sensitivity battery on a training-only holdout.
./run.sh run --mode sanity

# Supply the actual quoted instance rate. 0.40 is an arithmetic example,
# not a claim about current Vast.ai prices.
./run.sh plan --hourly-rate 0.40

# All 120 main sequences, with already completed pilots skipped.
./run.sh run --hourly-rate 0.40
```

Smoke-test every selected architecture/track before committing the full sweep if the environment is unfamiliar: omit `--models` from the smoke command. A smoke run checks software and optimizer memory; its small examples/evaluation do not prove production throughput or peak activation memory. The largest model's full pilot is still necessary.

Do not interpret pilot accuracies as a reason to keep only promising models, ranks or seeds. Pilot reuse saves compute because those exact configurations already belong to the declared main design.

For SSH disconnections, use tmux if available, or:

```bash
nohup ./run.sh run --hourly-rate 0.40 > training.log 2>&1 &
```

Relaunch the same command after interruption. Entire completed runs are skipped, and interrupted sequences recover after the last completed task. A partially completed task restarts from that boundary. Keep the same output, code, software and GPU type for each model.

## Budgeting with the initial $25

Do not trust the DOCX's 45–55-hour estimate: its run accounting is inconsistent, it predates the fp32-storage correction, and it is not a measurement on your offer. Until full-length pilot timings exist for each selected model/track, `plan` reports unknown cost instead of fabricating one. Afterwards it estimates the remaining grid from measured durations and displays a 1.25–2× planning allowance. That range is a heuristic buffer, not a confidence interval.

Use **the instance's total billed hourly rate**, and include setup, downloads, storage/network and idle time in your own allowance. For example, $25 at $0.40/hour buys 62.5 active hours before those additional charges. This arithmetic says nothing about whether the study will take 62.5 hours.

An optional whole-run admission guard is available:

```bash
./run.sh run --hourly-rate 0.40 --budget 25 --reserve 5
# Or use a wall-time allowance:
./run.sh run --max-hours 8
```

These limits apply to the current invocation, not your account balance or all past invocations. They admit complete runs and never shorten a task, drop seeds or subsample final validation. The first unmeasured run is admitted as a pilot, so the guard can exceed its allowance by that run. Setup/data preparation can also consume time before the next admission check. The guard does not stop the rental, terminate the instance, or guarantee a dollar cap. Exit code 3 indicates deferred work; relaunch later to continue.

The strongest cost savings are: keep model/data downloads cached, reuse pilots, finish paired seed blocks, fail early on infrastructure errors, and move CPU spectral work off rented GPUs. If the full design needs more than the credit, extend the budget or finish in later sessions. Reducing ranks/seeds/epochs/data would change the study and must be reported as a different, incomplete or amended design.

## Multiple GPUs

Use independent workers rather than DDP: a complete sequence is already small enough for one suitable GPU, and there are 120 independent runs. DDP would synchronize updates and complicate the effective-batch protocol. Multiple GPUs can reduce wall time but do not inherently reduce GPU-hours or price.

For identical GPU types on the same host:

```bash
./run_parallel.sh 0,1 --mode main --output results
```

The launcher assigns complete model/seed blocks, propagates failures, and writes separate worker logs. Both ranks and baseline of a seed remain together. Sharding is deterministic and pilot/main assignments match. Use the same shard count when relying on cached pilots on different machines. All workers must use a common Linux filesystem supporting file locks, or synchronize artifacts carefully after execution; copying overlapping result trees blindly can discard data.

For different GPU types, give each model all its ranks and baseline on one GPU type, for example:

```bash
CUDA_VISIBLE_DEVICES=0 ./run.sh run --models llama3.2-1b qwen2.5-1.5b --output results
CUDA_VISIBLE_DEVICES=1 ./run.sh run --models granite3.3-2b qwen2.5-3b --output results
```

Launch those in separate shells. The protocol records hardware per model and rejects mixing GPU types inside a model's comparisons. Cross-model results must disclose the hardware difference. With a shared multi-GPU rental, a worker's reported elapsed-cost estimate describes that instance's wall time; do not sum those estimates as if each were a separate bill.

## After GPU training

Copy the **entire** experiment directory, especially `cache/projections/`, `runs/*/checkpoints/`, protocol/software/hardware records and invocation manifests, to your own durable storage. Check that the copy completed before removing the rented copy. The source Hugging Face cache is useful for resuming training but is not needed for deferred spectral analysis once the base projection cache is present.

On a CPU machine:

```bash
./setup.sh --cpu
./run.sh intruders --output results --threads 4
./run.sh analyze --output results
```

Both commands can be rerun. Spectra resume per matrix. Set threads to a modest number suitable for the CPU: oversubscribed BLAS can be slower. Direct-SVD robustness analysis uses `--solver svd`; it stores separate derived caches and replaces the report's active spectral result with that solver's result. Finish one solver cohort before comparing correlations; do not pool partially updated solver outputs.

Read `results/aggregates/completeness.json` first. Behavioral completion and spectral completion are separate. The report and CSVs include per-seed values, sample SD, paired rank comparisons, feasible sets, fixed-2pp sensitivity, and intruder threshold sensitivity. A single-model completed window does not imply all four models are complete.

Vast.ai bills storage on stopped instances; its [billing FAQ](https://github.com/vast-ai/docs/blob/main/guides/reference/faq/billing.mdx) explains the distinction. Manage stopping/deleting the rental yourself after preserving artifacts. The suite deliberately has no provider credentials or automatic instance-destruction command.

## Validation boundary

The repository tests run real, tiny Llama/Qwen/Granite decoder architectures on CPU, plus independent checks for accumulation, spectra, seed pairing, resume and partial reports. The production pinned Qwen tokenizer and all three real GLUE splits were also checked during this redesign. No production-size GPU training, 5090/4090 VRAM benchmark, empirical LR optimum, or actual rental charge was measured in the development environment. Treat the first full GPU pilot as the acceptance test for your instance.
