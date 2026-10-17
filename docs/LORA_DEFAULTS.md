# Project LoRA defaults from calibration

The project now uses these LR/head policies automatically for fresh output directories. They were selected by the completed 48-case LoRA calibration in Modal cohort `frw-20261018` on 18 October 2026.

| Fixed model | Default adapter LR | Head LR multiplier | Nominal head LR | Upper tested LR boundary |
| --- | ---: | ---: | ---: | --- |
| Llama-3.2-1B | `4.00e-4` | `5` | `2.00e-3` | Yes |
| Qwen2.5-1.5B | `1.85e-4` | `1` | `1.85e-4` | No |
| Granite-3.3-2B | `3.20e-4` | `5` | `1.60e-3` | Yes |
| Qwen2.5-3B | `3.00e-4` | `5` | `1.50e-3` | Yes |

The classifier head uses the adapter LR multiplied by the listed ratio, following the same task-specific warmup/cosine schedule. These were selected jointly; using a different head ratio changes the tested policy. Each model retains one adapter LR across its main ranks.

These are the **best eligible tested policies**, not established global optima. Llama, Granite and Qwen3B won at the largest tested LR; a separate wider search could find a better rate. Two confirmation seeds and RTE-only endpoint checks do not prove optimality for every rank, task or numerical setting. Main official validation scores were not used to choose these defaults.

## Evidence and reproducibility

For each model, calibration screened three methodology-prior rates × head multipliers `{1,5}` on an RTE training holdout at rank 16/seed 42. The two strongest distinct LR candidates were confirmed through RTE → MNLI → SST-2 at seeds 42 and 123. The selected policy was checked on RTE at ranks 4 and 256. Each task used a fixed stratified training-only split of 1,992 fit and 498 holdout examples from the same 2,490-example source subset, three epochs, and effective batch 16. The measured partition was **micro-batch 16, accumulation 1**, evaluation batch 32, bf16 computation, fp32 trainable parameters and AdamW moments, q/v LoRA, alpha=2r and dropout=0.

The ordinary CLI partition remains micro-batch 4/accumulation 4 for memory compatibility with the full-FT track. To reproduce the measured LoRA partition, explicitly pass `--micro-batch 16 --accum 1`. Other task sequences, batch partitions or training policies may warrant their own calibration.

[calibrated-lora-defaults.json](calibrated-lora-defaults.json) preserves the selected values, exact model/data revisions, original code/context fingerprint, tested grids, calibration identities and SHA-256 values of the four original reports. Original reports, per-seed scores, trial logs and charts remain under `/frw-20261018/defaults/lr_test/experiments/<calibration_id>/` on Modal volume `fyp-finetune-results`, and are retrieved into `Finetune-result/lr_test/experiments/`. This static provenance summary is distinct from an activated output-cohort `lr_defaults.json`.

## Defaults and overrides

For each model/track, explicit CLI rates/head ratios take precedence, followed by valid cohort-local calibration, followed by the project defaults above. `list-models` displays registry defaults; `plan --output ...` shows the actual resolved policy, including cohort overrides. An explicit LR list changes the tested adapter rates but retains the resolved head ratio unless `--head-lr-multiplier` is also supplied.

```bash
./run.sh list-models
./run.sh plan --track lora --micro-batch 16 --accum 1 --output results-promoted
# When ready to allocate GPU time, the planned fresh cohort can be run with:
./run.sh run --track lora --micro-batch 16 --accum 1 --output results-promoted
```

Full-FT LRs remain the uncalibrated methodology priors `2.00e-5 / 1.85e-5 / 1.60e-5 / 1.50e-5`, with head multiplier 1 unless overridden or separately calibrated. They are not recomputed from the promoted LoRA rates. The four model checkpoints, default ranks and seeds remain fixed.

The preregistered `--test-lr` grid still uses `{0.5,1,2}` times the **original methodology priors** listed in [PROTOCOL_V3.md](PROTOCOL_V3.md), so promoting a default does not silently move the candidate grid. Its normal head search remains `{1,5}`. Use explicit `--lora-lrs` for a separate wider search in a fresh output cohort. The fifteen-run `sanity` battery also retains its original methodology rate comparisons.

## Running cohort and later analysis

Updating project defaults changes the scientific source fingerprint. The already-running Modal workers and preservation observer retain their original immutable image and activated LR/head policy. Their results and calibration fingerprints must not be rewritten to match the new source.

The original complete checkout is pinned at Git commit `3408ffb11bcae1a3bb8919a3edcbbfd661dafc58`, scientific source SHA-256 `ff3d682687b8ca5db927a383539cf816691c06a647c96c7b2750048f14d3fb16`. A frozen local checkout has been prepared in `Finetune-result/execution/frozen-checkout/`. Use that source for this cohort's CPU analysis, finalization and any recovery training; its own cohort-local defaults already contain the selected rates above. [OVERNIGHT.md](OVERNIGHT.md) contains the exact commands and a recreation command for a fresh clone.

Use a fresh output directory for experiments launched from the promoted-defaults source. Source/context validation continues to reject incompatible old calibration files and results; no integrity checks are bypassed.
