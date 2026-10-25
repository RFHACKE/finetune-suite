# Literature review behind protocol frw-v4 (2026-10-25)

This review extends the earlier one in [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) and
[literature-review.md](literature-review.md). It focuses on what changed the design. Each
entry states what the source shows, and what it means for the question "which LoRA ranks keep
learning high and forgetting low over sequential fine-tuning of sub-3B decoders?".

## 1. Rank and learning rate are coupled; a fixed LR confounds every rank sweep

- **Chen, Villar & Hayou (2026), "Learning Rate Scaling across LoRA Ranks and Transfer to Full
  Finetuning" (μA), arXiv:2602.06204.** Under the standard initialisation (A random, B = 0),
  the optimal LR depends on the LoRA multiplier. With α/r scaling (fixed α) it is approximately
  **rank-invariant**, although the stable range is narrow. With a constant multiplier it
  **decreases with rank** (≈ r^-1/2 in their analysis). The project's α = 2r is a constant
  multiplier of 2. *Consequence:* v3's single per-model LR (tuned at r = 16) was increasingly too
  large at high ranks. It explains the chance-level collapse at r ≥ 128 in the v3 data. v4 uses
  fixed α = 32 and still searches the LR per rank.
- **Thinking Machines Lab (2025), "LoRA Without Regret"** (thinkingmachines.ai/blog/lora). With
  α = 32 and scale α/r, the optimal LR varies by less than 2× between ranks 4 and 512. The paper
  explains why the 1/r prefactor makes early dynamics rank-independent. It also finds that
  attention-only LoRA clearly underperforms MLP/all-layer LoRA. That bears on external validity:
  the project keeps q/v targets for continuity, and all-linear is listed as a future cohort.
- **Biderman et al. (2024), "LoRA Learns Less and Forgets Less", TMLR** (arXiv:2405.09673).
  Forgetting is controlled by rank, but not monotonically. With α = 2r they had to *lower* the
  LR for r = 256 (code IFT 2e-4 → 1e-4; math 1e-4 → 5e-5) because of loss spikes. LoRA was more
  LR-sensitive than full fine-tuning. *Consequence:* the methodology's claim (Sec. 2.3) that
  α = 2r makes the step size rank-independent holds per parameter, not per function. The update
  BA is a sum of r terms, so its effect grows with r.
- **Kalajdzievski (2023), rsLoRA, arXiv:2312.03732.** With α/r scaling and a fixed LR, learning
  slows at very high rank (γ_r = α/√r stabilises it in the r → ∞ limit). This is one reason v4
  re-tunes the LR at every rank instead of trusting either parameterisation blindly.
- **Zhang et al. (2026), "The primacy of magnitude in low-rank adaptation", NeurIPS 2026** (as
  cited by arXiv:2605.12752). Apparent gains of LoRA variants often come from update
  *magnitude*, not direction. *Consequence:* v4 records ΔW norms at every boundary, so rank
  effects can be separated from magnitude effects.
- **Lee et al. (2026), "Learning Rate Matters: Vanilla LoRA May Suffice", arXiv:2602.04998.**
  Comparisons between LoRA configurations can reverse once the LR is tuned. This supports the
  per-rank search.

## 2. Rank and forgetting: what is (and is not) known

- **Kalajdzievski (2024), "Scaling Laws for Forgetting When Fine-Tuning LLMs", arXiv:2401.05605.**
  For Llama-2-7B-chat, forgetting follows a shifted power law in the number of fine-tuned
  parameters and update steps. LoRA still forgets, and forgetting rises roughly linearly with
  fine-tuning gains. *Consequence:* learning and forgetting must be read together (ACC plus BWT
  plus per-task diagonals). Lower forgetting alone is not a virtue.
- **Lu et al. (2024–2026), "Take Only What You Need" / CoDyRA, arXiv:2412.01004.** In
  continual learning, higher LoRA rank gives more plasticity and more forgetting. Moderately
  small ranks maximise the benefit.
- **Steele (2026), "Subspace Geometry Governs Catastrophic Forgetting in LoRA",
  arXiv:2603.02224.** Forgetting tracks the minimum principal angle between task-gradient
  subspaces, and is roughly rank-invariant when tasks are dissimilar (RoBERTa on GLUE, ViT on
  CIFAR). *Consequence:* the task-order cohorts (O2/O3) and the rank-trend tests are needed
  before claiming a rank effect. A flat curve is a possible, valid result.
- **Shuttleworth et al. (2025), "LoRA vs Full Fine-tuning: An Illusion of Equivalence", NeurIPS**
  (arXiv:2410.21228). LoRA creates "intruder" singular directions, which accumulate in
  continual learning and grow with higher LR. Very high-rank LoRA also forgets more of the
  pretraining distribution. v4 keeps the intruder measurement (ε 0.5–0.9, k = 10) as a
  mechanistic association, never as proof.
- **Luo et al. (2023/2025), "An Empirical Study of Catastrophic Forgetting in LLMs During
  Continual Fine-tuning", arXiv:2308.08747.** Forgetting is observed from 1B to 7B and is more
  severe at larger scale. The later version adds that larger models also start higher.
- **Zheng et al. (2025), "Spurious Forgetting in Continual Learning of Language Models", ICLR.**
  Much apparent forgetting is lost task *alignment*, not lost knowledge.
  **Davari et al. (2022), CVPR**: re-fitted linear probes reveal small *representation*
  forgetting even when task accuracy collapses. *Consequence:* v4 adds a linear probe for every
  seen task at every boundary. The first smoke run already showed this: Granite r=1024 had RTE
  head accuracy 75.1% but probe accuracy 85.2%.
- **Lun & Ming (2026), H-LoRA, CMC 88(1):87** (Qwen-0.5B, MiniMind). They report retention
  with high ranks (≈ 2/3 of d_model) under a simplified scaling. Their setting is domain SFT, not
  sequential BWT. v4's grid reaches r = 1024 (= d/2 for d = 2048; 2/3·d for Qwen-1.5B), so the
  high-rank regime is actually tested, at a properly tuned LR, instead of the v3 collapse.
- **Park (2026), "Rethinking the Rank Threshold for LoRA Fine-Tuning", arXiv:2605.03724.** For
  binary GLUE classification, r = 1 is competitive. For MNLI (3 classes) the optimal rank exceeds
  1. *Consequence:* v4's grid starts at r = 1.

## 3. Evaluation and hyperparameter-selection practice

- **Cha & Cho (2025), "Hyperparameters in Continual Learning: A Reality Check", TMLR**
  (arXiv:2403.09066). Tuning hyperparameters on the same stream that is evaluated inflates
  results. *Consequence:* v4 selects LRs only on a training holdout with separate seeds (11, 22)
  and reports different seeds (42, 123, 456, 789, 1024) on the untouched official validation
  sets.
- **Mirzadeh et al. (2020), "Understanding the Role of Training Regimes in Continual Learning",
  NeurIPS.** LR, batch size and dropout change forgetting through minima geometry. LR therefore
  has to be either controlled or tuned for each rank.
- **Lopez-Paz & Ranzato (2017)** (BWT) and **Chaudhry et al. (2018)** (forgetting F) for metric
  definitions. **De Lange et al. (2022)** for the task-incremental setting.
- Narula et al. (2026, ICML workshop) is cited in the proposal for seed sensitivity. It could not
  be independently verified (see the earlier review). The v3 data show the same phenomenon
  directly: Granite RTE retention was 51.6% vs 84.3% between two calibration seeds. v4 therefore
  uses five reported seeds and paired, seed-level inference.

## 4. Problems found in the proposal / methodology documents

1. **α = 2r with one LR per model** (methodology Sec. 2.2–2.3) confounds rank with effective
   step size. This is the decisive flaw, and the v3 data confirm it (Section 1 above;
   `docs/PROTOCOL_V4.md` Sec. 1).
2. **A full-FT "plasticity ceiling" in the window definition** makes the window depend on a
   second, separately tuned method. At the user's direction, full FT is deferred. v4 defines
   learning relative to the best LoRA rank, and adds an ACC non-inferiority window.
3. **"max(2, 2σ)" tolerances** favour noisy ranks, because more variance gives a wider margin.
   v4 uses fixed δ with δ-sensitivity and paired bootstrap confidence instead.
4. **A single task order** cannot separate rank effects from order effects. v4 adds a cyclic
   Latin square (O2, O3).
5. **RTE validation has 277 examples** (≈2.7 pp standard error per evaluation). v4 pairs seeds
   across ranks, reports per-task components, and uses the probe analysis. Conclusions rest on
   the paired, multi-task evidence, not on RTE alone.
6. **Claims of being "the first study"** cannot be established by a literature search. The
   defensible claim is a controlled, LR-decoupled rank sweep for 1–3B decoders on sequential
   classification.
7. **Factual corrections** carried over from the earlier review: Qwen2.5-3B uses the Qwen
   Research License, not Apache 2.0; Granite-3.3-2B was released in April 2025.
