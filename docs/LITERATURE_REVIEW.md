# Literature and design review

Current implementation: see the [frw-v3 amendment](PROTOCOL_V3.md) for LR/head selection and the five adopted measurement improvements. The v2 notes below are preserved as design history.

The expanded review and recommendation tables are available in [the editable review](literature-review.md) and [the improvement tables](RESEARCH_IMPROVEMENTS.md). The Word review can be generated locally using the [build instructions](LITERATURE_SEARCH.md#rebuilding-the-documents) and is excluded from Git. The notes below preserve the earlier suite-design review; source access and newer related work are qualified in [the expanded search record](LITERATURE_SEARCH.md).

Reviewed 17 October 2026 against both supplied DOCX files and primary sources. The original proposal motivates the study; methodology v1.1 supersedes its model selection. The four selected base checkpoints are preserved. This review separates published observations from choices made for this experiment. No experimental result is claimed here.

## What the project can establish

The question is how vanilla LoRA rank changes the learning–retention tradeoff when a single adapter learns RTE → MNLI Matched → SST-2. There are four models, five ranks, and five repeated initialization/shuffle seeds. Each task has its own head and task identity is available at evaluation. This is task-incremental classification, not class-incremental learning or general language-generation retention. The Qwen 1.5B/3B contrast is the primary size comparison; family differences remain in comparisons involving Llama and Granite. Even the Qwen pair differs in width, depth and GQA geometry: it is not a controlled scaling-law experiment.

The defensible contribution is a reproducible, narrowly scoped rank ablation with behavioral and spectral measurements. A literature search cannot establish that no overlapping work exists. Avoid “first ever,” “rank causes forgetting at every scale,” or deployment recommendations outside the measured tasks and ranks.

## Evidence map

| Primary source | Evidence relevant here | Consequence for this suite |
|---|---|---|
| [Hu et al., LoRA, ICLR 2022](https://arxiv.org/abs/2106.09685) | Frozen pretrained weights plus trainable low-rank factors provide a practical adaptation method; attention q/v targeting is an established configuration. | Retain vanilla LoRA on every q/v projection. Do not substitute QLoRA, rsLoRA, DoRA, rank-allocation methods or different target modules into the main study. |
| [Biderman et al., TMLR 2024, §§4.1–4.7 and appendices](https://arxiv.org/html/2405.09673v2) | Learning and retention depend on rank, LR, target modules and task/domain. Larger ranks sometimes close learning gaps. Their recommendations include scaling alpha with rank and using different LR ranges for LoRA and full FT. | Retain alpha = 2r, report learning and forgetting separately, and treat the two tracks' LR choices as explicit policies. Their 7B generation experiments do not prove a monotonic rank–BWT relationship for these classifiers. |
| [Shuttleworth et al., NeurIPS 2025, Algorithm 1](https://arxiv.org/html/2410.21228v3) | Intruder measurement compares top tuned singular vectors with all pretrained singular vectors. Published full-FT models usually show few intruders; their forgetting setting differs from this project. | Use all left reference vectors and absolute cosine, preserve every task boundary, report all four epsilon thresholds, and measure full-FT counts without forcing them to zero. Correlation with sequential BWT is a hypothesis to test. |
| [Lun & Ming, H-LoRA, CMC 2026](https://www.techscience.com/cmc/v88n1/67336/html) | Experiments on MiniMind and Qwen-0.5B motivate examining capacity limitations and larger rank fractions. Their domains, rank range, adaptation details and retention measurements differ. | The current r=4–64 study tests a low-rank regime. It cannot confirm or refute an optimum around two-thirds of hidden width. Treat this as motivation rather than a directly contradictory BWT result. |
| [Rathore et al., Findings 2025](https://aclanthology.org/2025.findings-ijcnlp.58/) | Rank tradeoffs in Q&A and domain generalization are task dependent. | Avoid a universal best rank; publish per-model curves and the tested conditions. |
| [Patil et al., Stacked LoRA, 2025](https://aclanthology.org/2025.ijcnlp-srw.4/) | Sequential isolated adapters address lifelong adaptation through a different method. | Relevant neighboring work, not an equivalent vanilla single-adapter rank sweep. |
| [Lee et al., Learning Rate Matters, 2026, v2](https://arxiv.org/abs/2602.04998v2) | Hyperparameter search can substantially alter comparisons among LoRA methods. | A literature LR is a starting value, not an experimentally established optimum. Keep within-model rank LRs fixed; publish sensitivity diagnostics and any deviations. |
| [Lopez-Paz & Ranzato, GEM, NeurIPS 2017](https://papers.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf) | Backward transfer measures final changes relative to the accuracy after learning an earlier task. | Preserve the full lower-triangular accuracy matrix and compute signed BWT. A task's diagonal accuracy need not be its historical maximum. |
| [Dettmers et al., 8-bit Optimizers, ICLR 2022](https://arxiv.org/abs/2110.02861) | Block-wise quantization reduces optimizer-state storage and performs well in the reported experiments. | Retain fp32 AdamW states for the primary LoRA sweep; use 8-bit moments for the large full-FT baseline. This quantizes moments, not model weights. The cross-track optimizer difference is explicit, with uniform fp32-state AdamW available for robustness. |
| [Zamirai et al., Revisiting BFloat16 Training](https://arxiv.org/abs/2010.06192) | Nearest rounding can cancel small weight updates when weights themselves are maintained in low precision. | Keep trainable parameters in fp32 while using bf16 autocast. Do not make an apparently cheaper full-FT baseline by silently discarding small updates. |

The proposal's Narula et al. seed-warning citation points to [OpenReview](https://openreview.net/forum?id=AmQVxvyRJC). During this review the paper endpoint required browser verification. I could not independently verify the exact revision or the quoted numerical example, so those numbers and the statement that a claim was retracted are not treated as verified evidence. Five-seed replication remains a sound design choice independently of that anecdote.

## Hyperparameters: evidence versus inference

There is no primary source establishing optimal LRs for this exact combination of checkpoints, heads, sample counts, task order and optimizer. The methodology's nominal LRs below remain the declared calibration/sanity priors. A subsequent 48-case training-holdout experiment supplied the [promoted project LoRA defaults](LORA_DEFAULTS.md); the historical priors are separate from those empirical choices:

| Model | LoRA LR | Non-embedding full-FT LR |
|---|---:|---:|
| Llama-3.2-1B | 2.00e-4 | 2.00e-5 |
| Qwen2.5-1.5B | 1.85e-4 | 1.85e-5 |
| Granite-3.3-2B | 1.60e-4 | 1.60e-5 |
| Qwen2.5-3B | 1.50e-4 | 1.50e-5 |

The proposed N^(-1/3) interpolation is an author-designed heuristic inspired by a few entries in [Brown et al.'s GPT-3 pretraining schedule, Table 2.1](https://arxiv.org/html/2005.14165v4). Pretraining is not classification fine-tuning. That table does not validate transferring this exponent to LoRA, distilled models or an 8-bit optimizer. The printed nominal rates above define the original methodology priors; their rounding does not exactly reproduce the formula.

[QLoRA](https://arxiv.org/abs/2305.14314) supplies useful precedent for adaptation LR scales, but uses quantized backbones and different data. Its reported rates do not establish an optimum for this unquantized experiment. Keep quantized model weights out of the main design.

The remaining values are conservative controlled choices: r={4,8,16,32,64}, alpha=2r, dropout=0, effective batch=16, three epochs/task, Adam betas=(0.9,0.999), epsilon=1e-8, weight decay=0, norm clipping=1, 3% warmup followed by cosine decay, max length=256. They give a tractable preregistered experiment; no paper proves this entire combination best. Freeze them before the rank sweep and inspect training losses, diagonal accuracies, class distributions and the held-out sensitivity battery.

The LR sensitivity battery uses only an 80/20 split of the selected RTE **training** examples. Official validation examples are reserved for reported evaluation. A small accuracy difference on one seed does not prove equivalence or plateau membership. The suite does not automatically lower a rate for a particular rank, tune on final validation accuracy, select the best epoch, or stop training early because a result looks favorable.

## Corrections to the documents' interpretation

1. There are **25 LoRA sequences per model**, not 20, with five ranks and five seeds. The main study is 100+20=120 sequences and 168,480 optimizer updates. The stated two-LR sanity design has 16 nominal comparisons but only **15 unique runs**, because Llama's two LoRA rates coincide. Smoke checks are additional, and pilots reuse eight main configurations.
2. Alpha/r=2 fixes a forward scaling coefficient. It does **not** make the induced update to W, its norm, Adam's adaptive step, or the effective function change rank independent. Report update capacity and trainable parameter counts, and frame conclusions under this specific LoRA parameterization.
3. Epsilon=0.9 counts **more** vectors as intruders than epsilon=0.6. It is a sensitive threshold, not proof of a conservative detector. A rotated basis can count as intruders while remaining inside the pretrained span. Near-degenerate singular values make individual-vector comparisons basis sensitive; do not interpret every counted vector as a new orthogonal subspace.
4. Nonzero intruders in full FT are not automatically a measurement defect. Synthetic unchanged-weight, sign, permutation and rotated-vector tests validate implementation. Empirical full-FT behavior is an observation.
5. Negative BWT is forgetting relative to each task's post-training diagonal. The report separately includes forgetting relative to each earlier task's maximum observed accuracy. Mean final-row accuracy is the standard final average; mean over observed matrix cells is supplementary.
6. A low-forgetting model may simply have learned little. Inspect **all task diagonals**, not only final SST-2 accuracy. Full FT is an empirical reference, not a guaranteed ceiling or successful learner.
7. max(2pp, 2*SD) can favor noisy configurations. The suite states exactly whose SD is used, requires the prescribed five seeds and baseline, retains disconnected feasible intervals, and adds a fixed-2pp sensitivity result. These margins are operational tolerances, not statistical non-inferiority tests.
8. Absolute intruder totals depend on depth and output geometry. Within-model associations are primary; pooled exploration uses the fraction of inspected vectors. Neither pooling nor Spearman correlation establishes causality. Seed-paired rank differences are reported with exploratory uncertainty intervals; five seeds do not remove all uncertainty.
9. [Qwen's release documentation](https://qwenlm.github.io/blog/qwen2.5-llm/) identifies the 3B model's **Qwen Research License**, rather than Apache 2.0. The model remains unchanged. The study includes approximately 3.09B parameters at its upper endpoint, so “1B–3B-class” is more accurate than a strict “under 3B” claim.
10. [Granite's base model card](https://huggingface.co/ibm-granite/granite-3.3-2b-base) lists 16 April 2025 as its release date; the methodology's December 2024 entry is incorrect.

## What remains empirical

The direction of the rank curve, existence of any feasible interval, whether earlier tasks were learned sufficiently, the link between spectra and BWT, the practical GPU throughput, and the total rental cost must come from measured runs. The suite makes no favorable-result assumption. A null association, disconnected feasible set, empty window, weak full-FT baseline, or cost exceeding the initial $25 are all legitimate findings to report.
