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
