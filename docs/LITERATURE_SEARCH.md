# Literature search and source audit

Review date: 17 October 2026. The expanded review is [literature-review.md](literature-review.md); its Word deliverable is [literature-review.docx](../literature-review.docx). Source records and access qualifications are in [literature_sources.tsv](literature_sources.tsv). The TSV is tab-separated UTF-8 and can be opened in a spreadsheet.

This was a structured critical scoping review, not a registered systematic review or an exhaustive census. No PRISMA screening totals, database hit counts, independent reviewer agreement, or formal risk-of-bias scores were manufactured. The 62 reference records include papers, official model/implementation documentation, historical bibliography, and two restricted manuscript leads. They are not 62 independently replicated findings or 62 papers all read cover to cover.

The proposal and methodology were extracted from their original DOCX XML and compared with the current code, `PROTOCOL_V2.md`, and `VALIDATION.md`. The four methodology checkpoints were treated as fixed. The suite currently has no full-study experimental findings to synthesize.

## Search approach

Web search was used for discovery, followed by primary-paper pages, relevant full-text sections, proceedings records, publisher pages, and official model cards. Search indexes and access state can change; this record describes the review session rather than promising an exactly repeatable search-engine ranking. Search result summaries were not substituted for unrestricted primary evidence.

| Search family | Representative queries used | What was checked |
|---|---|---|
| Direct rank/retention studies | `LoRA Learns Less and Forgets Less Biderman 2024 rank learning rate TMLR`; `LoRA vs Full Fine-tuning An Illusion of Equivalence intruder dimensions Shuttleworth NeurIPS 2025`; `H-LoRA Rethinking Rank Selection Controllable Knowledge Retention Edge AI Lun Ming 2026` | Scope, rank/scaling policies, forgetting definition, main results and limitations. |
| Recent theory and optimization | `LoRA rank catastrophic forgetting continual learning learning rate 2026`; `A Dynamical Theory of LoRA in Continual Learning`; `Learning Rate Matters LoRA 2602.04998`; `Continual LoRA ranks and placements` | Recent preprints, revised manuscripts, optimization confounds and theoretical assumptions. |
| Continual PEFT neighbors | `O-LoRA orthogonal subspace learning language models continual 2023`; `CLoRA 2025`; `SLoRA 2026 catastrophic forgetting`; `SpaRTA Rank Continual Learning`; `InfLoRA interference-free low-rank adaptation continual learning CVPR 2024`; `SD-LoRA scalable decoupled low rank adaptation 2025`; `Janus-LoRA 2026` | Whether a method changes replay, modules, objectives, isolation, routing, or rank allocation. |
| Parameterization and initialization | `rank-stabilized LoRA Kalajdzievski`; `LoRA+ Efficient Low Rank Adaptation Large Models Hayou 2024`; `Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning` | Scaling versus induced update size; fixed-rank versus adaptive/alternative methods. |
| Exact model/novelty screen | `Llama-3.2-1B rank forgetting`; `Qwen2.5-1.5B rank continual`; `granite-3.3-2b forgetting`; `feasible rank window LoRA`; `Stacked LoRA Patil 2025` | Overlap with fixed checkpoints and the contribution claim. |
| Proposal citation audit | `Capacity-gated forgetting Narula LoRA`; `Exploring Selective Layer Freezing Strategies in Transformer Fine-Tuning`; `Mitigating Catastrophic Forgetting Forgetting-Aware Pruning`; `Catastrophic Interference McCloskey Cohen 1989` | Existence, publication venue, method scope, and source access. |
| Evaluation and representation | `Three scenarios for continual learning`; `Gradient Episodic Memory 2017`; `Can BERT Refrain from Forgetting on Sequential Tasks`; `TRACE comprehensive benchmark continual learning`; `HANS Right for the Wrong Reasons`; `Annotation Artifacts in Natural Language Inference Data` | Task identity, BWT, recoverability, NLI robustness and broader retention constructs. |
| Uncertainty and calibration | `On the Stability of Fine-tuning BERT`; `Fine-Tuning Pretrained Language Models Weight Initializations Dodge 2020`; `The Hitchhiker's Guide Statistical Significance Dror 2018`; `On Calibration of Modern Neural Networks` | Initialization/order noise, paired comparisons, small samples and confidence metrics. |
| Spectral/numerical controls | `effective rank Roy Vetterli 2007`; direct primary pages for CKA, 8-bit optimizers, bfloat16 training, GPT-3, muP and SciPy subspace angles | Definitions, measurement limitations and optimizer/precision policy. |

## Important access and version qualifications

- Biderman: relevant sections and appendices of arXiv `2405.09673v2`; TMLR publication record checked.
- Shuttleworth: arXiv HTML `2410.21228v3`, including intruder definition, interventions and sequential classification; NeurIPS 2025 proceedings independently verified. The proposal's ICLR label is incorrect.
- H-LoRA: publisher full text and limitations; the high-rank heuristic is scoped to the studied compact regime.
- CoDyRA: `2412.01004v9` (5 September 2026), now titled *Take Only What You Need*. The arXiv page reports Findings of EMNLP 2026 acceptance; this is identified as author-reported rather than an independently opened final proceedings paper.
- Lee et al.: relevant sections/tables of `2602.04998v2`. Broad LR sensitivity is relevant; no exact-suite optimum is asserted.
- Marchetta et al.: `2609.39367v1` (30 September 2026), a recent theory preprint with a simplified setting.
- Choi: `2609.03900v1` (3 September 2026), a recent preprint involving two project checkpoints. Its factual/replay setting limits transfer to this FYP.
- Narula: the forum and direct PDF delivery were browser-challenged. Indexed primary PDF text was found at `https://openreview.net/pdf/62c9ae220d3326fb6b0381257854394d388220fa.pdf`. Its seed warning is treated as qualified context; anecdotal numerical values and a formal withdrawal/retraction were not certified.
- TF-SSD: indexed primary text was available for an anonymous submission (`yrZwqv5ACT`). Authors, publication status and numerical findings were not certified. It is only a lead showing that Granite continual adaptation is not an untouched topic.
- Hwang: direct publisher delivery intermittently failed; indexed publisher methods/hyperparameters and the authors' university abstract were checked. Do not transfer its runtime improvements to this suite.
- RTE historical chapter: the DOI endpoint was unavailable. It is retained as dataset-history bibliography; no detailed original-paper result is attributed.
- McCloskey–Cohen: historical metadata was checked against the author-institution record and indexed original chapter. No claim of reading an inaccessible publisher copy is made.

## Inclusion and exclusions

The review includes direct evidence and neighboring work needed to interpret rank, retention, optimization, spectra and uncertainty. Generative benchmarks, vision continual learning and alternative PEFT are labeled as adjacent evidence. Intentional unlearning/backdoor removal, diffusion customization, adapter-serving systems, application-only posts and unverified claims of universal retention are not treated as direct evidence for this classification protocol.

Model choices and experiment code were not changed by writing the review. All candidate hyperparameters, screening triggers, feasibility tolerances and optional cohorts are recommendations requiring a declared protocol amendment before adoption. Source URLs, definitions and status should be rechecked if the thesis is submitted substantially later.

## Rebuilding the documents

The Markdown review is the editable source; the TSV supplies the numbered bibliography. The optional documentation dependency is separate from the experiment environment:

```bash
python3 -m pip install -r requirements-docs.txt
python3 tools/build_literature_review.py
```

The builder writes the root `literature-review.docx` and extracts all 28 proposals into `docs/RESEARCH_IMPROVEMENTS.md`. It checks citation coverage, source identifiers, recommendation counts and table structure. It does not execute experiments or change their controls.

Subsequent implementation: the user authorized all five summarized improvements. The [frw-v3 amendment](PROTOCOL_V3.md) implements per-task gates, LR/head calibration, update norms/effective rank and leading-subspace diagnostics. Other candidate interventions remain proposals; the review tables label their original v2 baseline status.
