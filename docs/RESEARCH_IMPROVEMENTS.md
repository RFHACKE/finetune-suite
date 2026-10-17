# Research improvements and candidate hyperparameters

Extracted from the [full literature review](literature-review.md), reviewed 17 October 2026. A Word copy can be generated locally using the [build instructions](LITERATURE_SEARCH.md#rebuilding-the-documents) and is excluded from Git. The four selected base models remain fixed. Selected measurement and LR/head-calibration proposals are implemented in [frw-v3](PROTOCOL_V3.md); the remaining entries are optional proposals. New/Extend statuses describe the reviewed v2 baseline.

Priority P1 means the best return for this FYP; P2 means a targeted follow-up after the core design is sound; P3 means optional future work. “New” identifies work absent from the reviewed v2 baseline. “Extend” means that baseline already supplies part of the measurement. The subsequent v3 implementation is identified in the executive assessment and protocol amendment. These are design recommendations and candidate values, not claims of demonstrated improvement.

Cost categories refer to additional work: CPU means analysis of already saved artifacts; evaluation means additional forward passes; calibration means isolated training-only checks; cohort means additional experimental training. Actual dollars require measured timings and the current rental quote. Preserve existing primary outcomes even when adding richer secondary measurements.


### Table 5. Measurements and analysis improvements

| ID / priority | Proposed measurement and concrete specification | Advantages | Disadvantages and additional cost | Status / evidence |
|---|---|---|---|---|
| M01 / P1 | Report every diagonal, the two final BWT components, and all three transition changes; retain the full matrix. | Separates learning failure, NLI transfer and later interference; exposes cancellation. | More outputs need interpretation; no additional GPU work. | Extend: diagonals/BWT already saved. CL evaluation [3–5]. |
| M02 / P1 | Supplementary all-task set: every mean LoRA diagonal within 2pp of the matched full-FT diagonal, and each earlier task's mean final-minus-diagonal change at least −2pp. Show 1/2/5pp sensitivity and worst-task loss. | Avoids approving a rank solely because SST-2 is good or BWT cancels. | Stricter sets may be empty; full FT itself needs a healthy-learning check. Tolerances are operational values, not significance tests. CPU analysis. | New all-task gates; current fixed-2pp sensitivity partly exists. Design proposal. |
| M03 / P1 | Balanced accuracy, macro-F1, confusion matrices and per-class recall at every existing boundary. | Detects majority-class behavior and selective class damage. | Metrics are correlated and can be noisy in small classes. CPU from saved predictions. | New reporting; labels/classes already available. Dataset context [35–40]. |
| M04 / P1 | Correct→incorrect and incorrect→correct fractions per old task and transition; verify example identity. | Reveals hidden churn despite unchanged net accuracy. | Requires reliable alignment and does not explain the cause. CPU from saved predictions. | New; no logits needed. Project diagnostic. |
| M05 / P1 | Pair seed effects; add matched-example uncertainty, fixed-margin interval decisions and predeclared multiplicity handling. | Separates practical effect size, seed variation and finite validation noise. | Five seeds limit power and bootstrap tails; formal non-inferiority may remain inconclusive. CPU. | Extend: sample SD/paired intervals exist. [41–43]. |
| M06 / P1 | Frozen-backbone head-only reference plus refitted-head recovery at selected boundaries; fit on training data only. Optional matched-example CKA on captured features. | Tests undertrained heads and recoverable representation drift; CKA describes feature similarity. | Extra feature extraction/storage; a refit changes the evaluation contract. CKA cannot establish retained knowledge or replace primary BWT. | New. Probe motivation [33], CKA [46]; decoder transfer remains a hypothesis. |
| M07 / P1 | Per-matrix relative Frobenius displacement, spectral norm, stable rank, entropy effective rank, and 90% energy rank of merged ΔW. | Measures update strength and actual use of nominal capacity; QR makes LoRA analysis cheap. | Norms do not identify lost knowledge; full-FT calculations cost CPU. | New; q/v snapshots suffice. [11,29,45]. |
| M08 / P1 | Leading-subspace distance/principal angles at k=10; k=20 sensitivity; record the cutoff gap using at least k+1 values. | Distinguishes basis rotation from subspace movement and flags unstable singular vectors. | Near-degenerate cutoff subspaces remain ambiguous; additional CPU eigensolves. | New; preserve current k=10 intruder outcome. [47], mathematical extension. |
| M09 / P2 | Energy-weighted intruder fraction, top-k energy coverage, and continuous maximum-cosine quantiles at all declared epsilons. | Distinguishes weak counted directions from dominant changes and reduces dependence on one threshold. | A top-k fraction is not total matrix energy; increases analysis choices. CPU only. | Extend: threshold counts already exist. Proposal based on spectral diagnostics [9]. |
| M10 / P2 | Joint layer-depth profiles of intruders, relative update norms and effective rank, q/v separately; normalize depth descriptively. | Shows where interference concentrates and respects GQA geometry. | Layers within one run are dependent; no layer-level sample-size inflation. CPU. | Extend: layer/type counts partly exist. [52–56]. |
| M11 / P2 | Add fixed MNLI-mismatched and HANS evaluation, with declared label mapping and identical examples for all configurations. | Tests whether retained NLI behavior transfers beyond familiar validation patterns. | More inference and analysis; label mapping and task difficulty affect interpretation. | New evaluation; no backbone changes. [36,39,40]. |
| M12 / P2 | Save fp32 class logits; report NLL/Brier and an explicitly binned ECE. Optionally add a fixed external-text ΔNLL proxy at base and boundaries. | Detects confidence damage; external text separates language-model retention from task BWT. | Logits are not currently stored. External text requires the original LM head and extra inference; WikiText is a proxy, not the actual pretraining distribution. | New capture/evaluation. [44,59]. |
| M13 / P3 | Matched fresh-base single-task references for representative ranks/models, using the same data and optimizer budget. | Separates intrinsic task learning from sequence-induced plasticity limitations. | Adds training; a full model×rank×seed×task grid is expensive. Random future heads do not supply a meaningful FWT baseline. | New cohort. CL evaluation motivation [4,5]. |
| M14 / P3 | Attenuate selected final-checkpoint intruder components, with matched-norm control directions and an unmodified reference. | Tests a mechanism more directly than a correlation. | Requires full reconstructable models and extra inference; control/direction selection matters. Do not merge these modified checkpoints into main results. | New exploratory diagnostic; published intervention precedent [9]. |

### Table 6. Hyperparameter and experimental-design candidates

| ID / priority | Concrete candidate change | Advantages | Disadvantages and additional cost | Evidence and recommendation |
|---|---|---|---|---|
| H01 / P1 | Training-only LR sensitivity: multipliers {0.5,1,2} of each nominal LoRA/full-FT rate; inspect LoRA r=4 and r=64. | Tests a high-impact assumption and whether rank endpoints train healthily. | Calibration costs GPU time; RTE-only selection does not prove later-task adequacy. Freeze one per-model/track rate across ranks. | LR sensitivity supported by [8,32]; exact grid is proposed, not proven optimal. |
| H02 / P1 | Separate head LR multiplier {1,5}; try 10 only in a declared follow-up if the head remains a bottleneck. | Helps a new random classifier learn when backbone LR is small, especially in full FT. | Could change apparent method/rank gaps; must apply a consistent policy across tracks/ranks. Extra calibration. | Author-designed candidate; diagnose with M06 first. No paper validates these exact ratios. |
| H03 / P2 | Separate rsLoRA cohort: fixed alpha=8, scaling alpha/sqrt(r), anchored to the existing coefficient at r=16. | Checks sensitivity to rank scaling without replacing the chosen models. | Changes the adaptation method; needs independent tuning and additional sequences. | Scaling motivation [14]. Matching alpha=8 is a design choice, not an empirical optimum. |
| H04 / P2 | LoRA dropout {0,0.05} in a matched sensitivity cohort; keep backbone/head policy explicit. | Tests whether small-data regularization changes underlearning or retention. | Injects stochasticity and may slow learning; more seeds/cohort work. | Exact 0.05 is a conservative proposed trial; do not assume dropout reproduces rank constraints. |
| H05 / P3 | AdamW weight decay {0,0.01}, with declared parameter groups. | Tests overfitting sensitivity. | Decay on A/B is not a penalty on merged ΔW; interacts with LR and factor scaling. Extra cohort. | Optimizer framework [58]; neither value is an exact-suite optimum. |
| H06 / P2 | Uniform fp32-state AdamW robustness for the full-FT comparator, preserving other controls. | Checks whether baseline differences reflect moment precision. | Higher VRAM and rental cost; do not silently mix optimizer cohorts. | Motivation [48,49]; current LoRA states are already fp32. |
| H07 / P3 | Add r=128 as an explicitly separate extension if the r=64 boundary remains scientifically unresolved. | Tests whether the search stops before adequate learning/capacity. | Twenty extra sequences for four models × five seeds; cannot retroactively enlarge a preregistered main window. | Boundary extension is a proposed follow-up, not a transfer of H-LoRA's optimum. |
| H08 / P3 | q/v versus all-attention or attention+MLP adaptation in a separate, parameter-budget-aware comparison. | Tests whether a q/v bottleneck explains limited learning. | Adds many trainables and changes the treatment; equal rank is not equal budget. Extra training and spectra. | Module-sensitivity motivation [8,29]. Keep q/v primary. |
| H09 / P2 | One reverse-order cohort on Qwen-1.5B, ranks {4,16,64}+full FT, five seeds: 20 sequences. | Checks whether the NLI-first ordering drives the conclusion. | Adds roughly one-sixth of the main sequence count; one reverse order does not cover all six orders. | Task-order threat [4]. Fixed checkpoint selection is preserved. |
| H10 / P2 | Max length 256 versus 512 only if a predeclared training-only truncation audit warrants it. | Tests whether truncation rather than rank limits NLI learning. | More activation memory and runtime; different tokenizers complicate comparisons. A 1% trigger is an author-chosen screening rule. | Proposed diagnostic; count lost tokens/examples before choosing the main setting. |
| H11 / P3 | Warmup fraction {0.03,0.10}, holding epochs and LR grid policy fixed. | Tests early optimization/head-instability sensitivity. | Changes effective update exposure and may undertrain with short schedules. Calibration/cohort cost. | Optimization motivation [42]; no evidence that 10% is universally better here. |
| H12 / P1 | Throughput pilot: micro-batch 4×accumulation4 versus 8×2, effective batch fixed at 16; fixed eval batch selected before main. | Can reduce rental time without deleting data, seeds or ranks. | Dynamic padding/kernel arithmetic can change numerics; validate memory and use separate cohort identity. | Engineering optimization, not a promised statistical equivalence. |
| H13 / P2 | Three versus five epochs/task on a small declared sensitivity cohort after checking learning curves. | Tests whether low-rank underlearning is caused by insufficient optimization time. | About 67% more task-training updates for the five-epoch arm; longer training may increase forgetting. | Candidate duration check, not outcome-dependent early stopping. |
| H14 / P3 | Optional replay reference: buffer 0 versus 32 training examples per previous task, stratified and fixed in advance. | Gives a practical retention reference while keeping the four backbones. | Changes the rehearsal-free question; declare whether replay replaces or adds update slots and account for storage. | Replay context [5,21,34]; 32 is a proposed budget-conscious value. |

The 28 proposals above are not a request to run a 28-way experiment. First add inexpensive analysis and isolate major calibration failures. A broad Cartesian product over rank, LR, dropout, decay, modules and epochs would overwhelm the budget and weaken the clarity of the main question.

### Table 7. Exact proposed LR calibration grids

| Fixed model | Current LoRA LR | Proposed LoRA {0.5×,1×,2×} | Current full-FT LR | Proposed full-FT {0.5×,1×,2×} |
|---|---:|---|---:|---|
| Llama-3.2-1B | 2.00e-4 | 1.00e-4, 2.00e-4, 4.00e-4 | 2.00e-5 | 1.00e-5, 2.00e-5, 4.00e-5 |
| Qwen2.5-1.5B | 1.85e-4 | 9.25e-5, 1.85e-4, 3.70e-4 | 1.85e-5 | 9.25e-6, 1.85e-5, 3.70e-5 |
| Granite-3.3-2B | 1.60e-4 | 8.00e-5, 1.60e-4, 3.20e-4 | 1.60e-5 | 8.00e-6, 1.60e-5, 3.20e-5 |
| Qwen2.5-3B | 1.50e-4 | 7.50e-5, 1.50e-4, 3.00e-4 | 1.50e-5 | 7.50e-6, 1.50e-5, 3.00e-5 |


These are candidate search points, not a revised declaration of the best LR. If an adopted calibration selects a different baseline rate, update the preregistered protocol, configuration and baseline-analysis rule before main training; the current executable window rule anchors to the current nominal rate.

## Sources cited in these tables

[3] van de Ven, G. M., and Tolias, A. S. (2019). [Three scenarios for continual learning](https://arxiv.org/abs/1904.07734). arXiv:1904.07734.

[4] De Lange, M., et al. (2022). [A Continual Learning Survey: Defying Forgetting in Classification Tasks](https://arxiv.org/abs/1909.08383). IEEE TPAMI 44(7), 3366–3385; online 2021.

[5] Lopez-Paz, D., and Ranzato, M. (2017). [Gradient Episodic Memory for Continual Learning](https://papers.neurips.cc/paper_files/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html). NeurIPS 30.

[8] Biderman, D., et al. (2024). [LoRA Learns Less and Forgets Less](https://arxiv.org/html/2405.09673v2). Transactions on Machine Learning Research.

[9] Shuttleworth, R., Andreas, J., Torralba, A., and Sharma, P. (2025). [LoRA vs Full Fine-tuning: An Illusion of Equivalence](https://arxiv.org/html/2410.21228v3). NeurIPS 38; initial preprint 2024.

[11] Rathore, D., Kumar, V., Bansal, C., and Moitra, A. (2025). [How Much is Too Much? Exploring LoRA Rank Trade-offs for Retaining Knowledge and Domain Robustness](https://aclanthology.org/2025.findings-ijcnlp.58/). Findings of IJCNLP-AACL, 1003–1013.

[14] Kalajdzievski, D. (2023). [A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA](https://arxiv.org/abs/2312.03732v1). arXiv:2312.03732v1.

[21] Wang, X., et al. (2023). [Orthogonal Subspace Learning for Language Model Continual Learning](https://aclanthology.org/2023.findings-emnlp.715/). Findings of EMNLP, 10658–10671.

[29] Lu, H., Zhao, C., Xue, M., Yao, L., Moore, K., and Gong, D. (2026). [Take Only What You Need: Rank Minimization as an Implicit Forgetting Regularizer in Continual Learning](https://arxiv.org/html/2412.01004v9). arXiv:2412.01004v9; authors report Findings of EMNLP 2026 acceptance.

[32] Lee, Y.-A., Ko, C.-Y., Chen, P.-Y., and Yeh, M.-Y. (2026). [Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning](https://arxiv.org/html/2602.04998v2). arXiv:2602.04998v2; preprint.

[33] Tao, M., Feng, Y., and Zhao, D. (2023). [Can BERT Refrain from Forgetting on Sequential Tasks? A Probing Study](https://arxiv.org/abs/2303.01081). ICLR 2023.

[34] Wang, X., et al. (2023). [TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models](https://arxiv.org/abs/2310.06762). arXiv:2310.06762.

[35] Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., and Bowman, S. R. (2019). [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://arxiv.org/abs/1804.07461). ICLR 2019; initial preprint 2018.

[36] Williams, A., Nangia, N., and Bowman, S. R. (2018). [A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://aclanthology.org/N18-1101/). NAACL-HLT, 1112–1122.

[37] Socher, R., et al. (2013). [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](https://aclanthology.org/D13-1170/). EMNLP, 1631–1642.

[38] Dagan, I., Glickman, O., and Magnini, B. (2006). [The PASCAL Recognising Textual Entailment Challenge](https://doi.org/10.1007/11736790_9). Machine Learning Challenges, LNCS 3944, 177–190.

[39] Gururangan, S., et al. (2018). [Annotation Artifacts in Natural Language Inference Data](https://arxiv.org/abs/1803.02324). NAACL-HLT 2018.

[40] McCoy, R. T., Pavlick, E., and Linzen, T. (2019). [Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference](https://arxiv.org/abs/1902.01007). ACL 2019.

[41] Dodge, J., et al. (2020). [Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping](https://arxiv.org/abs/2002.06305v1). arXiv:2002.06305v1.

[42] Mosbach, M., Andriushchenko, M., and Klakow, D. (2021). [On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines](https://arxiv.org/abs/2006.04884v3). ICLR 2021.

[43] Dror, R., Baumer, G., Shlomov, S., and Reichart, R. (2018). [The Hitchhiker’s Guide to Testing Statistical Significance in Natural Language Processing](https://aclanthology.org/P18-1128/). ACL, 1383–1392.

[44] Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html). ICML, PMLR 70, 1321–1330.

[45] Roy, O., and Vetterli, M. (2007). [The Effective Rank: A Measure of Effective Dimensionality](https://doi.org/10.5281/zenodo.40328). EUSIPCO, 606–610.

[46] Kornblith, S., Norouzi, M., Lee, H., and Hinton, G. (2019). [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html). ICML, PMLR 97, 3519–3529.

[47] SciPy contributors (version 1.15.3). [scipy.linalg.subspace_angles](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.linalg.subspace_angles.html). Official implementation documentation, not an experimental paper.

[48] Dettmers, T., Lewis, M., Shleifer, S., and Zettlemoyer, L. (2022). [8-bit Optimizers via Block-wise Quantization](https://arxiv.org/abs/2110.02861). ICLR 2022; initial preprint 2021.

[49] Zamirai, P., Zhang, J., Aberger, C. R., and De Sa, C. (2020; revision 2021). [Revisiting BFloat16 Training](https://arxiv.org/abs/2010.06192v2). arXiv:2010.06192v2.

[52] Meta (2024). [Llama-3.2-1B model card](https://huggingface.co/meta-llama/Llama-3.2-1B). Official model documentation.

[53] Qwen Team (2024). [Qwen2.5-1.5B model card](https://huggingface.co/Qwen/Qwen2.5-1.5B). Official model documentation.

[54] Qwen Team (2024). [Qwen2.5-3B model card](https://huggingface.co/Qwen/Qwen2.5-3B). Official model documentation.

[55] IBM Granite Team (2025). [Granite-3.3-2b-base model card](https://huggingface.co/ibm-granite/granite-3.3-2b-base). Official model documentation.

[56] Ainslie, J., et al. (2023). [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245). EMNLP 2023.

[58] Loshchilov, I., and Hutter, F. (2019). [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101). ICLR 2019.

[59] Merity, S., Xiong, C., Bradbury, J., and Socher, R. (2017). [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). ICLR 2017; initial preprint 2016.

