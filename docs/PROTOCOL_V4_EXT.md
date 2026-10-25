# frw-v4 extension cohorts (pre-registered 2026-10-25, before launch)

The main v4 study ([FINDINGS_V4.md](FINDINGS_V4.md)) left two questions open that bear directly on
where the feasible rank window ends and how far it generalises. Everything else (models, data,
tasks, order O1, optimiser, α = 32, head LR, evaluation) is unchanged from
[PROTOCOL_V4.md](PROTOCOL_V4.md). Budget: the unused October credit, with the same caps
(admission $27, hard stop $29 per account, deadline 2026-10-31 22:00 UTC).

## E1 — Is high-rank instability caused by rank or by the learning rate?

**Why it matters.** The upper edge of the window (Granite r ≥ 64, Qwen2.5-3B r ≥ 256) rests on 1–2
collapsed seeds out of 5, at LRs chosen with only two tuning seeds. If the collapses disappear at
a slightly lower LR, the honest conclusion is "high ranks need a lower LR", not "high ranks are
infeasible".

**Design.** Granite-3.3-2B and Qwen2.5-3B (the two models with collapses), q/v LoRA, ranks
{16 (control), 64, 128, 256, 512, 1024}. Each rank runs at its frozen v4-selected LR and at half of
it, with five **new** seeds {2024, 3033, 4044, 5055, 6066}: 120 runs, official validation, no
probes. At the selected LR, these are pooled with the five existing main seeds, giving n = 10.

**Failure (pre-registered).** A run fails if its final ACC is more than 10 pp below the median ACC
of that model's r = 16 runs at the selected LR, or if any task diagonal is within 5 pp of that
task's majority-class accuracy (the task was never learned).

**Analysis.**
1. Failure rate per (model, rank, LR) with Wilson 95% CIs.
2. Rank effect: an exact Cochran–Armitage trend test of failure across ranks at the selected LR.
3. LR effect: a two-sided Fisher exact test comparing selected vs half LR, pooled over ranks ≥ 64
   within each model.
4. For non-failed runs, ACC/BWT at half vs selected LR as paired-free mean differences with
   bootstrap CIs.

**Decision rule for the window.** A rank whose failure rate is ≤ 1/10 at either LR is reinstated
as feasible *at that LR*, provided its non-failed ACC is within 2 pp of the r = 16 control at the
same LR.

## E2 — Does the window depend on LoRA target modules (q/v vs all linear layers)?

**Why it matters.** Current practice (QLoRA, Biderman et al. 2024, "LoRA Without Regret" 2025)
applies LoRA to every linear layer, and attention-only LoRA is known to underperform. A window
measured only for q/v might not transfer.

**Design.** Llama-3.2-1B and Qwen2.5-1.5B (the two cheapest models), LoRA on q, k, v, o, gate, up
and down projections, ranks {4, 16, 64, 256}. The LR search and selection rule are identical to
v4: 4 LRs × tuning seeds {11, 22} on the training holdout, extended at grid edges. Then five main
seeds {42, 123, 456, 789, 1024} on official validation with linear probes. The main seeds are the
same as the q/v cohort, so the comparison is paired by seed (same heads, same data order).
Spectral diagnostics stay on q/v matrices only, so they remain comparable.

**Analysis.**
1. ACC / plasticity / BWT / probe-BWT vs rank for all-linear.
2. The same windows as in v4.
3. Paired differences all-linear − q/v at each matching rank and seed, with bootstrap CIs and Holm
   correction across ranks.

**Hypotheses** (stated before data): all-linear learns slightly more (higher plasticity) at equal
rank. Whether it forgets more is open: Biderman et al. found that capacity and forgetting rise
together. The window for all-linear is expected at lower or equal ranks.

## Assignment

| Account | Work |
|---|---|
| 1 | E1 Granite r = 512, 1024 |
| 2 | E1 Granite r = 16, 64, 128 |
| 3 | E1 Qwen2.5-3B r = 16, 64, 128, 256 |
| 4 | E2 Qwen2.5-1.5B |
| 6 | E1 Qwen2.5-3B r = 512, 1024; E2 Llama-3.2-1B |
| 5 | E1 Granite r = 256 (small remaining credit); also receives overflow |
