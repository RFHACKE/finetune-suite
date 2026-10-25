# Protocol frw-v4 — rank × learning-rate decoupled LoRA forgetting study

Status: **pre-registered 2026-10-25, before any v4 result was observed.** Every rule below
(grids, selection objective, seeds, window definitions, statistics) is fixed in code
(`frw/config.py`, `frw/jobs.py`, `frw/train.py`, `frw/analysis.py`) and in this document.
Any later deviation is reported as a deviation.

Scope decision by the user (2026-10-25): **LoRA only**. Full fine-tuning is deferred to a later
cohort; its code path exists but is not scheduled.

## 1. Why v3 cannot answer the question

The v3 cohorts (ranks 4–256, then 512/1024; 180 sequences) used `alpha = 2r`, i.e. a constant
LoRA multiplier of 2, with one learning rate per model selected at r = 16. Under the standard
initialisation (A random, B = 0) the optimal learning rate for a constant multiplier **decreases
with rank** (Chen, Villar & Hayou 2026, μA: η ∝ r^-1/2 under their SignSGD/Adam proxy; Biderman et
al. 2024 had to lower the LR for r = 256 with α = 2r because of loss spikes). A fixed LR therefore
became several times too large as rank grew. The data show exactly this failure:

| model | ranks that never learned (A11 RTE ≈ 50%, A22 MNLI ≈ 34%) | ranks with bimodal seed collapse |
|---|---|---|
| Llama-3.2-1B | 128, 256, 512, 1024 | 32, 64 |
| Qwen2.5-1.5B | 1024 (most seeds), 512 (2 seeds) | 128 |
| Granite-3.3-2B | 128, 256, 512, 1024 | 8, 32, 64 |
| Qwen2.5-3B | 256, 512, 1024 | 64, 128 |

Their "near-zero forgetting" is the absence of anything to forget. In v3, rank was confounded
with the effective step size, so v3 cannot locate a feasible rank window. Its low-rank results are
still a valid single-LR cohort and are reported separately as the motivating pilot.

## 2. Design

**Question.** For sub-3B decoder-only models fine-tuned sequentially with one LoRA adapter, which
ranks give high accuracy on every task over the sequence (learning) and little loss on earlier
tasks (forgetting) **when every rank receives a fairly tuned learning rate**?

**Models** (unchanged; pinned commits as v3): Llama-3.2-1B, Qwen2.5-1.5B, Granite-3.3-2B,
Qwen2.5-3B. Qwen 1.5B/3B is the within-family size pair.

**Tasks/data** (unchanged selection): GLUE RTE, MNLI-matched, SST-2; 2,490 training examples per
task drawn once with seed 42 (RTE: whole train split); rows frozen to JSON on each volume and
checked identical to the Hub pipeline. Encoding `Premise: … \nHypothesis: …` / `Text: …`,
max 256 tokens, right padding to a multiple of 8.

**Orders.** Primary O1 = RTE → MNLI → SST-2 (proposal). Robustness O2 = MNLI → SST-2 → RTE and
O3 = SST-2 → RTE → MNLI complete a cyclic Latin square (every task in every position once).

**Classifier.** One linear head per task on the last non-pad token of the final hidden state;
heads seeded by (seed, position) only, so identical across ranks and LRs; old heads frozen
(task-incremental). Head LR fixed at 1e-3 for every rank and LR (in v3 the head LR was a multiple
of the adapter LR, which would make it co-vary with the rank-specific LR).

**LoRA.** q_proj and v_proj of every layer; **α = 32 fixed, scale α/r** (PEFT default semantics;
μA "Init[A], α ∝ 1/r" regime, in which the optimal LR is approximately rank-invariant — tested
here, not assumed); dropout 0; A Kaiming-uniform, B = 0; fp32 adapter and head storage,
fp32 AdamW moments, bf16 autocast compute. One adapter continues across tasks (no merging).
Ranks r ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}. Ranks above a matrix's width
(v_proj width 256 for Qwen, 512 for Llama/Granite) are over-complete factorizations; the
maximum update rank per matrix is recorded.

**Optimisation (all ranks).** AdamW β = (0.9, 0.999), ε = 1e-8, weight decay 0, gradient clip 1.0
over adapter + current head, batch 16, 3 epochs per task, 6% linear warmup then cosine to zero,
optimizer and schedule reset at each task boundary.

## 3. Learning-rate selection (per model and rank; never uses official validation)

Stage A (tune): for every (model, rank), LR ∈ {1e-4, 2e-4, 4e-4, 8e-4} × tuning seeds {11, 22};
each job fits on a stratified 80% (1,992) of every task's 2,490 examples and scores the remaining
20% (498, the same split as the v3 calibration) after every task, order O1.

Objective: mean over tuning seeds of the final-row holdout accuracy (ACC = average accuracy on all
three tasks after the last task). Ties → lower LR. If the winner lies on an edge of the tested grid,
LR/2 (or 2·LR) is added with both seeds and the rule is re-applied (bounds 1.25e-5 … 6.4e-3).
Diverged or failed runs score 0. The chosen LR is frozen in `selections/<model>__r<rank>.json`
before any reported job for that pair starts.

Stage B (main, reported): at the selected LR, seeds {42, 123, 456, 789, 1024}, full 2,490
examples per task, full official validation (RTE 277, MNLI-m 9,815, SST-2 872) after every task.

Stage C (order robustness, reported separately): r ∈ {1, 4, 16, 64, 256, 1024}, orders O2 and O3,
seeds {42, 123, 456}, LR selected in Stage A for O1.

## 4. Measurements

Per run: full lower-triangular accuracy and NLL matrices; predictions at every boundary; step
losses/gradient norms/LRs; for main runs a linear-probe accuracy for every seen task at every
boundary (L2 logistic regression on frozen final-token features, fit on that task's training
examples) to separate representation forgetting from head misalignment; spectral diagnostics of
every adapted matrix at every boundary (||ΔW||_F and relative norm, stable/entropy/90%-energy rank
from the exact factor spectrum, intruder dimensions at ε ∈ {0.5…0.9} with k = 10 against all
pretrained left singular vectors, leading-subspace distance at k = 10); final adapters (bf16) for
main runs.

Definitions (T = 3, A[i][j] = accuracy on task j after training task i):
ACC = mean_j A[T][j]; plasticity = mean_j A[j][j]; BWT = mean_{j<T} (A[T][j] − A[j][j]);
forgetting F = mean_{j<T} (max_{l<T} A[l][j] − A[T][j]); per-task components are always shown.

## 5. Feasible rank window (pre-registered)

All criteria use seed means at the selected LR (Stage B) and are computed per model.

* **Learning:** every diagonal A[j][j] ≥ (best rank's mean diagonal for task j) − δ.
* **Retention:** BWT ≥ −δ_F, where δ_F is the stability tolerance.
* Primary δ = 2 pp, δ_F = 2 pp; sensitivity δ, δ_F ∈ {1, 3, 5} pp.
* The **ACC window** (directly answers "highest accuracy over the whole sequence"), two tiers:
  *point* — mean ACC within δ of the best rank's mean ACC; *confident* — the lower bound of the
  95% paired bootstrap CI (over seeds) of ACC(r) − ACC(best) is ≥ −δ (non-inferiority by margin δ).
  The best rank itself is always in both tiers.
* Windows are reported in raw r, r/d_model and r/(v_proj width); disconnected or empty sets are
  valid outcomes. No interpolation to untested ranks.

## 6. Statistics

Seeds are paired across ranks (same seed ⇒ same heads and data order). Effects are reported as
mean ± SD and paired differences with 95% percentile bootstrap CIs (10,000 resamples over seeds);
rank trends with a mixed model (`metric ~ log2(rank) + log2(rank)^2 + (1|seed)`) per model and
Spearman ρ over rank means. Multiplicity across ranks within a model is handled by Holm
correction on the paired tests against the best rank. Task-order robustness: windows recomputed
per order and averaged over orders. Mechanistic analyses (spectral, probe) are associations.

## 7. Compute and budget

Six Modal Starter workspaces, each capped at $30 (hard stop at $29 real metered spend, admission
cap $26 (lowered from $27 at launch +30 min) on a real-time ledger). Priority: Stage A (all ranks interleaved so partial completion
still spans the range), then Stage B seeds 1–3, then seeds 4–5, then Stage C. Every job records
GPU type, wall time and cost.

## 8. Known limitations (declared up front)

Three GLUE classification tasks; q/v targets only (all-linear LoRA, full fine-tuning and other
task families are future cohorts); one data subset; LR selected with two seeds on a holdout of
the training data; head LR fixed rather than tuned per model; bf16 kernels are not bitwise
deterministic, so reproducibility is statistical (seeds, data, code and versions are pinned).

## 9. Execution notes and deviations (recorded during launch, 2026-10-24 UTC)

1. **Smoke test excluded.** Three validation runs on account 1 (Llama r=16 tune and main, Granite
   r=1024 main, LR 2e-4 chosen by hand) checked the GPU pipeline. They are archived in
   `results-v4/smoke/`, removed from the account's queue, and never enter the analysis.
2. **Spectral precision.** Account 2's first five workers (started 21:05 UTC) computed the factor
   QR/SVD for the update diagnostics in float64. Every later worker uses float32 on the GPU,
   because float64 is ~64× slower on L40S. The relative difference in singular values is ~1e-6.
   Training, evaluation and probes are unaffected.
3. **Budget guards.** The real-billing hard stop runs inside each account's watchdog, using the
   container's workspace identity, plus a local supervisor. Account 1's ledger includes the GPU
   benchmark and the smoke test (~$1.8).
4. **Accounts and hardware.** Every run uses one GPU type (NVIDIA L40S 48 GB) and the same image
   (torch 2.8.0 + cu128, transformers 4.57.6, peft 0.17.1, datasets 4.4.1).
