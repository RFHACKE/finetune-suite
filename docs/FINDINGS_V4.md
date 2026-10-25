# frw-v4 findings: the feasible LoRA rank window for 1–3B decoders

Data: 780 completed runs (0 failed, 0 diverged), 2026-10-24/25, six Modal accounts, NVIDIA L40S. All
runs used one GPU type and one image. Compute spend was $109.63 of the $180 pool, and $0 was billed
beyond the free credit. Protocol: [PROTOCOL_V4.md](PROTOCOL_V4.md). Every number below comes from
`results-v4/analysis/*.json`; the figures are in `results-v4/analysis/figures/` (each has a `.json`
with the plotted data), and everything is in one page in `results-v4/analysis/report.html`.

- 416 learning-rate search runs: training holdout, seeds 11/22.
- 220 reported runs: 4 models × 11 ranks × 5 seeds, official validation, RTE → MNLI → SST-2.
- 144 task-order robustness runs: 4 models × 6 ranks × 2 orders × 3 seeds.

## 1. Answer

> **Update after the extension cohorts (Section 6):** the instability at high ranks was a
> learning-rate effect, not a rank effect. At half the selected LR, ranks up to 1024 had no
> failures for Qwen2.5-3B and few for Granite. The practical window is therefore r = 4–32 at the
> tuned LR, and it widens to r ≤ 1024 if the LR is lowered for high ranks, with no accuracy gain.
> LoRA on all linear layers raises accuracy (much more for Llama than Qwen) and keeps the rank
> curve flat.

**A rank between 4 and 32 (best 8–16) is feasible for all four models.** In that range the mean
accuracy over all three tasks after the full sequence (ACC) is within 2 pp of each model's best
rank, forgetting is at or near its lowest, and no seed collapsed. Ranks above that range give no
gain. From r = 64 (Granite) or r = 256 (Qwen2.5-3B) upward they become *unreliable*: one or two of
five seeds collapse or forget catastrophically. Ranks 1–2 are slightly worse for Qwen2.5-1.5B and
Granite.

| Model (d_model) | Best-ACC rank | ACC window, confident (95% CI within 2 pp of best) | ACC window, point (mean within 2 pp) | Learning + retention window (δ = 2 pp) | Unreliable ranks (≥1 collapsed seed) |
|---|---|---|---|---|---|
| Llama-3.2-1B (2048) | 32 (82.2%) | 4, 16–32, 512 | 1, 4–32, 128–1024 | empty (BWT ≤ −4.5 at every rank) | none severe |
| Qwen2.5-1.5B (1536) | 512 (86.0%) | 2–64, 256–512 | 1–1024 | 4, 512 | 128 (one weak seed) |
| Granite-3.3-2B (2048) | 8 (86.6%) | 8–16, 1024 | 2–32, 1024 | empty (BWT ≤ −4.5 at every rank) | 1, 64, 128, 512 |
| Qwen2.5-3B (2048) | 8 (89.5%) | 1–128 | 1–128 | 1–8, 32, 128 | 256, 512, 1024 |

In units of the hidden width, the common window r = 4–32 is r/d ≈ 0.002–0.016 (0.003–0.021 for
Qwen2.5-1.5B). This is two orders of magnitude below H-LoRA's suggested ≈ 2/3·d. In our
sequential-classification setting, very high ranks only add risk.

## 2. Supporting evidence

**The rank curve is a plateau, not a slope.** With a properly tuned LR, the ACC spread across
"healthy" ranks is about 1–3 pp per model (`main_acc_final_vs_rank.png`). No rank differs from
the best rank after Holm correction (all p_holm ≥ 0.625, `paired_tests.json`). The Spearman trends
of ACC and BWT against log rank are not significant for any model (`trends.json`, all p ≥ 0.056).
Over 1 ≤ r ≤ 128, five seeds detect no rank effect on forgetting larger than seed noise.

**The earlier "rank causes / prevents forgetting" curves were a learning-rate artifact.** In v3,
the step size grew with rank. ACC fell to chance at r ≥ 128 and BWT went to ≈ 0 because nothing
was learned. With the LR re-tuned per rank, the same models keep ACC flat up to r = 1024
(`v3_vs_v4.png`). In the LR search, raising the LR from 1e-4 to 4e-4 tripled forgetting at every
rank while learning stayed the same (`lr_surface_*.png`). **The learning rate controls forgetting
far more strongly than rank.**

**High ranks fail by instability, not by systematically worse means.** Examples of single
collapsed seeds:
- Granite r = 128, seed 42: ACC 60.2, BWT −44.3 (the other four seeds −3 to −9).
- Qwen2.5-3B r = 1024, seed 42: ACC 60.0 (the other four seeds 88.4–89.0).
- Qwen2.5-3B r = 256 (selected LR 4e-4), seed 123: ACC 60.8.

Within a model, these failure rates rise with rank (`seed_variability.png`). For deployment, a
rank that is right on average but fails 1 run in 5 is not feasible. This is why the confident
windows stop at 32 or 128.

**Model size and family.** Forgetting at the best ranks:

| Model | BWT at best ranks |
|---|---|
| Qwen2.5-3B | ≈ −1.5 to −2 pp |
| Qwen2.5-1.5B | ≈ −2 pp |
| Granite-3.3-2B | ≈ −4.5 to −5 pp |
| Llama-3.2-1B | ≈ −5 to −6 pp |

Within the Qwen pair (1.5B → 3B), the larger model learns more (plasticity 86.7 → 90.5%), forgets
slightly less, and has a wider confident window (r = 2–64 → 1–128). Family differences (Llama,
Granite vs Qwen) are larger than the size effect. Llama and Granite never reach |BWT| ≤ 2 pp at
any rank, which is why their learning-plus-retention window is empty at δ = 2 (it opens at
δ = 5; see `feasible_window_map.png`).

**Task order.** In the two other cyclic orders (`task_order_robustness.json`):
- Ranks 4–64 remain in the high-ACC plateau for every model.
- Forgetting is generally smaller than in the proposal's order. RTE → MNLI → SST-2 is the hardest
  order for retention.
- The Qwen2.5-3B r = 256 instability reappears in order O2 (ACC 78.8 ± 18.1).

**Most measured forgetting is head misalignment, not lost knowledge.** Linear probes re-fitted on
the final representation recover most of the "forgotten" accuracy. For example, Qwen2.5-1.5B
r = 4–512 has head BWT about −2 pp but probe BWT about 0 pp; Granite r = 16 has head BWT −4.5 pp
but probe BWT +0.1 pp (`probe_vs_head_bwt.png`). The representation largely keeps the earlier
tasks; the frozen old heads drift out of alignment with it. This matches Zheng et al. (2025,
"spurious forgetting").

**Spectral diagnostics are weak predictors.** Intruder-dimension fraction and update norm correlate
with forgetting only weakly and inconsistently (Spearman ρ −0.16 to 0.55; significant only for
Qwen2.5-1.5B, `spectral_association.json`). RQ3 receives a negative/inconclusive answer at this
scale: intruder counts do not explain rank-dependent forgetting here.

**LR vs rank.** The selected LRs stay within 2.5e-5 to 4e-4 and show no monotone trend with rank
(`selected_lr_vs_rank.png`). This is consistent with the approximate rank-invariance predicted
for fixed α. Choices between neighbouring LRs were often close calls with two tuning seeds, and
this adds scatter to individual cells.

## 3. Practical recommendation (for these tasks/models)

Use **q/v LoRA with r = 8 or 16, α fixed (e.g. 32), LR ≈ 1e-4–2e-4** for sub-3B decoders learning a
sequence of classification tasks. Do not raise the rank to reduce forgetting: it does not help,
and above r ≈ 64–128 it makes training unreliable. To reduce forgetting, lower the LR, choose a
model family that forgets less, or refresh the old task heads (the probes show the information is
still there).

## 4. Limitations

- Three GLUE tasks, q/v targets only; no full fine-tuning reference (deferred by the user).
- Five seeds give limited power: "no significant difference" means within-noise, not proven
  equality.
- RTE validation has only 277 examples.
- LRs were selected with two tuning seeds, so neighbouring-LR choices are noisy (e.g. Qwen2.5-3B
  r = 256 picked 4e-4).
- The order-robustness runs reuse the O1-selected LR.
- Results are task-incremental (separate heads). Single-head or generative settings may behave
  differently.

## 5. Possible follow-ups (all accounts have their November credit untouched)

- Full fine-tuning reference cohort.
- All-linear LoRA at r ∈ {4, 16, 64} to test whether the window moves.
- More seeds at r ∈ {64, 128, 256, 512} to estimate collapse rates precisely.
- A base-model linear-probe baseline.

## 6. Extension cohorts (pre-registered in [PROTOCOL_V4_EXT.md](PROTOCOL_V4_EXT.md))

Data: `results-v4/analysis/ext_stability.json`, `ext_all_linear.json`, and the figures
`ext_stability_failure_rate.png` and `ext_all_linear_vs_qv.png`.

### E1 — high-rank instability is driven by the learning rate, not by rank

There were 120 new runs with five fresh seeds (2024–6066) at each rank's selected LR and at half of
it, for Granite-3.3-2B and Qwen2.5-3B. At the selected LR they are pooled with the five original
seeds (n = 10). A run counts as a failure if its ACC is more than 10 pp below the r = 16 median or
if any task was never learned.

| | Granite failures, selected LR | Granite failures, half LR | Qwen2.5-3B failures, selected LR | Qwen2.5-3B failures, half LR |
|---|---|---|---|---|
| r = 16 | 0/10 | 0/5 | 0/10 | 0/5 |
| r = 64 | 1/10 | 0/5 | 0/10 | 0/5 |
| r = 128 | 1/10 | 0/5 | 1/10 | 0/5 |
| r = 256 | 0/10 | 0/5 | 2/10 (LR 4e-4) | 0/5 |
| r = 512 | 0/10 | 1/5 | 1/10 | 0/5 |
| r = 1024 | 0/10 | 1/5 | 1/10 | 0/5 |

- **Rank trend in failures:** none (Cochran–Armitage at the selected LR: Granite p = 0.54,
  Qwen2.5-3B p = 0.21).
- **LR effect, ranks ≥ 64:**
  - Qwen2.5-3B: 5 of 50 runs failed at the selected LR and 0 of 25 at half the LR (Fisher
    p = 0.16). Every Qwen failure occurred at the selected LR.
  - Granite: 2/50 vs 2/25 (p = 0.60). With 10 seeds, its failure rate is low at every rank.
- **Accuracy of runs that did not fail, half LR:** flat in rank for Qwen2.5-3B (89.2–89.5% from
  r = 16 to r = 1024). Granite is noisier (81.5–87.6%).

**Interpretation.** The v4 "collapses" at high rank came from a learning rate that was on the
edge of stability, chosen with only two tuning seeds. They do not show that high ranks are
inherently unstable. Under the pre-registered decision rule, every tested rank up to 1024 is
reinstated as feasible at one of the two LRs for Qwen2.5-3B. For Granite, ranks 16, 64, 128,
256 and 1024 are reinstated; r = 512 is not reinstated at either LR. The original recommendation
still stands for cost and
simplicity: r = 8–16, because higher ranks buy no accuracy. But the upper edge is set by the
learning rate, not by rank: if a high rank is needed, lower the LR by about 2×.

### E2 — all-linear LoRA (q, k, v, o, gate, up, down)

E2 covers Llama-3.2-1B and Qwen2.5-1.5B at r ∈ {4, 16, 64, 256}, with the same per-rank LR search
and the same five reported seeds as q/v, so the comparison is paired by seed.

| model | r | LR (all-linear) | n | ACC all-linear | ACC q/v | Δ ACC (95% CI) | plasticity Δ | BWT all-linear | BWT Δ |
|---|---|---|---|---|---|---|---|---|---|
| llama3.2-1b | 4 | 1e-04 | 5 | 84.4 ± 0.8 | 80.9 | +3.5 [+2.5, +4.7] | +1.2 | -3.3 | +3.4 |
| llama3.2-1b | 16 | 1e-04 | 5 | 84.3 ± 0.7 | 81.4 | +2.9 [+2.1, +3.7] | +1.1 | -3.6 | +2.7 |
| llama3.2-1b | 64 | 1e-04 | 5 | 84.9 ± 1.1 | 80.2 | +4.7 [+3.5, +5.8] | +2.8 | -3.2 | +2.9 |
| llama3.2-1b | 256 | 1e-04 | 5 | 85.3 ± 0.6 | 81.0 | +4.3 [+3.3, +5.6] | +2.5 | -1.8 | +2.7 |
| qwen2.5-1.5b | 4 | 2e-04 | 5 | 86.1 ± 0.7 | 85.7 | +0.4 [-0.0, +0.8] | +1.9 | -4.0 | -2.2 |
| qwen2.5-1.5b | 16 | 1e-04 | 5 | 86.7 ± 0.7 | 85.4 | +1.3 [+0.6, +2.0] | +1.9 | -2.8 | -0.9 |
| qwen2.5-1.5b | 64 | 2e-04 | 5 | 86.4 ± 0.9 | 85.1 | +1.4 [+0.8, +1.9] | +2.8 | -4.1 | -2.2 |
| qwen2.5-1.5b | 256 | 5e-05 | 5 | 86.0 ± 0.9 | 85.4 | +0.6 [-0.3, +1.5] | +1.1 | -2.8 | -0.6 |

- **Within all-linear the rank curve is flat.** For both models the ACC window, both point and
  confident, covers the whole tested range 4–256. The rank conclusion therefore does not depend on
  the target modules.
- **Llama-3.2-1B:** all-linear LoRA improves ACC by about 3–5 pp at every rank. It both learns
  more (+1 to +3 pp) and forgets less (BWT +2 to +3 pp).
- **Qwen2.5-1.5B:** all-linear learns more (+1 to +3 pp) but also forgets more (BWT −0.6 to
  −2.2 pp), so the net ACC gain is small (+0.4 to +1.4 pp). This matches the learning–forgetting
  trade-off reported by Biderman et al. (2024).
- **Statistical note:** with n = 5 paired seeds the smallest possible exact sign-flip p-value is
  0.0625, so no Holm-adjusted p can fall below 0.25 across 4 ranks. The 95% bootstrap CIs, which
  exclude zero for most Llama differences, are the more informative summary.

**Updated recommendation.** For sub-3B decoders on sequential classification, use **LoRA on all
linear layers at r = 8–16 with a tuned LR (≈1e-4 for Llama, 1e-4–2e-4 for Qwen)**. Raising the
rank does not reduce forgetting. If a high rank is used, halve the LR.
