# Protocol frw-v3: calibrated learning rates and richer measurements

This user-authorized amendment implements all five improvements requested after the literature review: per-task learning/retention gates, LR testing, head-LR calibration, merged-update norms/effective rank, and leading-subspace diagnostics. The four exact checkpoint identities/revisions, task order, 2,490 examples/task, five main ranks, five main seeds, three epochs, effective batch 16, q/v targets, alpha=2r, dropout=0, storage/compute/optimizer precision policies and official main validation sets remain as specified in [v2](PROTOCOL_V2.md).

The implementation uses suite 3.0.0 and protocol `frw-v3`. Start a new output cohort. v2 results are not loaded into v3 analysis; source hashes and immutable controls prevent silent reuse. No production-checkpoint calibration results, optimal rates, accuracy improvements or rental-time measurements were generated during implementation.

## Evidence and candidate grid

[Lee et al., Learning Rate Matters](https://arxiv.org/html/2602.04998v2) show that model, task and LoRA parameterization affect useful LR ranges; [Biderman et al.](https://arxiv.org/html/2405.09673v2) motivate measuring learning alongside retention and checking the optimization policy. Neither study establishes exact optimal rates for this four-model classification protocol. [Mosbach et al.](https://arxiv.org/abs/2006.04884v3) motivate replicated calibration and checks for failed learning.

Use the methodology's printed rates as transparent priors and test the three-point grids below. Their small differences across families are heuristic starting points, not a verified size-transfer law. The head multipliers `{1,5}`, screening/finalist budget, majority gates and tie breakers are declared project design choices rather than published exact-suite optima.

| Model | LoRA candidates | Full-FT candidates |
|---|---|---|
| Llama-3.2-1B | 1e-4, 2e-4, 4e-4 | 1e-5, 2e-5, 4e-5 |
| Qwen2.5-1.5B | 9.25e-5, 1.85e-4, 3.70e-4 | 9.25e-6, 1.85e-5, 3.70e-5 |
| Granite-3.3-2B | 8e-5, 1.6e-4, 3.2e-4 | 8e-6, 1.6e-5, 3.2e-5 |
| Qwen2.5-3B | 7.5e-5, 1.5e-4, 3e-4 | 7.5e-6, 1.5e-5, 3e-5 |

## Calibration and activation

`./run.sh --test-lr --output results-v3` is an alias for `run --test-lr`. Calibration completes and exits before main training begins. `plan --test-lr` and `run --test-lr --dry-run` print the plan with no model loading. Defaults cover all four models and both tracks unless the scope is explicitly narrowed.

Each task's fixed training subset is split once, stratified 80/20 at data seed 42: 1,992 train and 498 holdout examples. The explicit source indices, fingerprints and split hashes are saved. Official validation does not participate. Later main training uses all 2,490 training examples and the untouched official validation sets.

1. Screen three rates × two head multipliers on RTE, seed 42, three epochs. LoRA uses rank 16. Each candidate starts fresh from the pinned base with matched head initialization/shuffle policies.
2. Choose the best head multiplier for each rate by RTE accuracy, then NLL, lower LR and lower multiplier. Confirm the top two distinct rates.
3. Train each finalist through RTE → MNLI → SST-2 at seeds 42 and 123, using training holdouts at every boundary. Optimizer/scheduler reset at each task exactly as in main training.
4. Require each candidate's mean diagonal to exceed its own per-task holdout majority baseline. Select the eligible candidate with the highest mean final-row accuracy over tasks and seeds. Ties use higher mean diagonal, lower mean final-row NLL, lower LR, then lower head multiplier. Raw scores, BWT, learning checks, seed SD and boundary warnings stay visible. Average final accuracy is an explicit selection objective, not a proof that every task/rank has an individually optimal LR.
5. Diagnose the selected LoRA rate/head policy at ranks 4 and 64 on RTE, seed 42. Report underlearning without removing ranks; numerical instability blocks activation. These checks are not independent multi-task validation at every main rank.
6. Only a complete requested scope with valid evidence and saved plots can become active. Infrastructure failures, incomplete confirmation, budget stops and no healthy candidate leave defaults unmodified. Expected numerical divergence is recorded as an unstable candidate; it is never represented as a completed scientific score.

The default search has 48 single-task screens, 32 three-task confirmations and eight single-task endpoint checks: up to 88 calibration cases and 57,000 updates. This is 33.8% of the main study's 168,480 updates, not a dollar/time fraction. Full-FT cases are more expensive than LoRA and constitute a larger share of calibration. The initial $25 remains an allowance rather than a guaranteed total. Admission limits stop before another case when exhausted; an admitted unmeasured case may overrun. Calibration omits permanent projection/adapter/full-weight exports, keeping predictions/metrics/logs and only temporary recovery states for incomplete sequences.

Results are resumable per case/task boundary. JSON reports retain every trial, shortlist, finalist, seed score, learning gate, endpoint, failure and selected LR/head multiplier. Charts show screening accuracy/NLL versus LR for both head ratios, confirmation scores with seed SD, learning/retention scatter, winning accuracy matrices and training-loss trajectories. Batches retain invocation time/cost records. `analyze` can regenerate the latest charts on CPU.

Active `<output>/lr_defaults.json` is written atomically and is local to that output cohort. Main `plan`, `run` and parallel workers load it automatically, validating code/controls and matching completed evidence. The selected LoRA LR is common across main ranks; full FT has its own calibrated LR and head ratio. The baseline, completeness checks and cross-model figures use the frozen declared rates rather than the old priors or outcome-selected main baselines. Explicit main LR lists are labeled sensitivity settings and do not move the baseline anchor. Calibration cannot change a root after main/pilot controls are registered. Shared/exclusive locks serialize activation against main launches.

## Classifier-head learning rates

The current task head uses `head_multiplier × adapter_or_backbone_lr`, with the same warmup/cosine schedule fraction. Old task heads remain frozen. Logs record both actual scheduled rates. Full-FT dense matrices keep 8-bit moments under `standard`; classifier heads and vectors retain fp32 states even when optimizer parameter groups are split. `--head-lr-multiplier` allows an explicit fixed ratio before main collection; changes require a separate immutable cohort.

## Supplementary all-task gates

Keep the original methodology window and fixed-2pp sensitivity as separate outputs. For each tested rank at a fixed LR, supplement them with these seed-mean constraints:

- For each of three tasks, LoRA diagonal accuracy minus the matched, declared full-FT diagonal must be at least `−δ`.
- For each of two earlier tasks, LoRA final accuracy minus its own diagonal must be at least `−δ`.

Use δ=2pp primarily and 1/5pp as declared sensitivity. Require the full five-rank/five-seed design, complete declared full-FT baseline, and a full-FT diagonal exceeding the official validation majority baseline on every task. Do not certify a set from underlearned/missing references. Empty/disconnected sets remain valid outcomes.

Display all diagonals, final task changes and the three transition changes; aggregate averages alone can cancel. Per-task learning comparisons pair the same seed in LoRA and full FT. Retention comparisons use within-run boundaries. Bonferroni 95% t intervals cover 25 comparisons within each model/LR family: five ranks × five constraints. Interval-supported/failing/inconclusive decisions remain distinct from point-estimate operational sets. They assume independent approximately normal seed differences and do not establish uncertainty over new dataset draws or all task orders.

## Merged-update capacity and subspaces

Run these on CPU through `intruders`; do not keep a rented GPU for spectral analysis. Analyze every q/v matrix at every task boundary in both tracks. Preserve the original top-10/all-pretrained-left-vector intruder statistic and epsilons. New versioned caches prevent reuse/mixing of old partial summaries.

For ΔW report Frobenius norm, relative Frobenius displacement, spectral norm, stable rank `Σσ² / σmax²`, entropy effective rank `exp(−Σp log p)` where `p=σ/Σσ`, numerical rank and 90% energy rank. Zero updates receive explicit flags and zero dimensionality measures. Numerical rank uses `max(shape) × eps(float64) × σmax`. Report effective rank divided by nominal LoRA rank, or full matrix dimension for full FT. Two thin QR factorizations reduce LoRA's merged-update spectrum to an r×r SVD, making the metrics invariant to reciprocal rescaling of A/B. Full-FT updates require dense CPU decompositions.

At k=10 and 20 compute principal angles and `Dk = 1 − ||U0,kᵀ Ut,k||F²/k` for the leading reference/tuned subspaces. Retain at least k+1 tuned singular values for the cutoff gap. Report absolute/relative gaps; relative gaps below 1e-3 are diagnostic near ties, not a universal significance threshold. These measures are invariant to basis rotations inside the same span but remain sensitive when the cutoff itself is nearly degenerate. If a matrix is smaller, record the actual k; absent next singular values give null gaps. The full reference SVD is cached once and the tuned top-21 decomposition is reused for all these metrics.

Save per-matrix diagnostics, q/v identity and normalized layer depth to JSON/CSV. Average matrices within each run before seed variability in charts, so layers do not become artificial independent replications. These are descriptive mechanisms/associations, not causal demonstrations. The other literature-review proposals remain optional rather than silently entering the main grid.
