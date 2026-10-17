# Literature Review: The Feasible Rank Window

Rank, learning and forgetting in compact decoder-only language models

Review date: 17 October 2026. Project reviewed: the original proposal, methodology v1.1, and executable protocol `frw-v2`. This is a research synthesis and recommendation document, not a report of completed experiments.

Subsequent update, 18 October 2026: completed 48-case LoRA calibration supplied the [current project LR/head defaults](LORA_DEFAULTS.md). The nominal rates and candidate tables in this historical review describe the original methodology priors, which remain the registered calibration/sanity search centers.

## Executive assessment

The project remains worthwhile as a controlled empirical study of how vanilla LoRA rank affects sequential classification in four fixed base checkpoints. Its strongest contribution would be a reproducible account of when an adapter learns enough, retains earlier tasks, and uses its available update capacity. An empty feasible set, a non-monotonic rank curve, or no spectral association would all be useful results if the design and uncertainty are reported correctly.

The evidence does not support a universal claim that increasing rank necessarily increases, or necessarily reduces, forgetting. Papers examine different knowledge, task sequences, parameterizations, optimizers, and evaluation endpoints. Low forgetting can also reflect weak learning. The review therefore separates pretrained-capability retention, earlier-task retention, and recoverability after changing a classifier or disabling an adapter.

Keep the four chosen model identities fixed. Improve measurement and calibration before expanding the model or method grid. The highest-value additions are per-task learning/retention gates, paired uncertainty, update norms and effective rank, subspace diagnostics, and a small head-recovery study. Most can use existing artifacts or a modest amount of extra evaluation.

This review describes the audited frw-v2 baseline. A subsequent user-authorized implementation adopts per-task gates, LR/head calibration, update-capacity measurements and subspace diagnostics in the [frw-v3 amendment](https://github.com/RFHACKE/finetune-suite/blob/main/docs/PROTOCOL_V3.md). Remaining recommendations are proposals. Existing controls and results remain distinguishable from the amendment; no production experiment results are claimed.

## 1. Scope, search approach and strength of evidence

This structured critical scoping review draws on 62 references: research papers, an implementation reference, official model cards, and explicitly identified limited-access contextual records. It is not a registered systematic review or a claim to have found every publication. The literature search covered web-indexed primary sources available through 17 October 2026, with particular attention to work published after the original proposal's core references.

Primary sources were sought through arXiv, ACL Anthology, NeurIPS/ICLR proceedings, PMLR, publishers, and authors' official model/project pages. Starting points were every identifiable citation in the supplied documents, followed by related-method searches, reference tracing, and exact searches for the four checkpoints. Search themes included rank and forgetting; continual LoRA; intruder dimensions; rank scaling and initialization; classification-head recovery; learning-rate selection; seed uncertainty; NLI robustness; and numerical memory policies.

Selection prioritized direct rank/retention evidence, methods that explain relevant confounds, and sources needed to define measurements. Diffusion personalization, intentional unlearning, general adapter-serving systems, and application-only posts were not treated as equivalent experiments. Aggregators helped discovery but were not used as the authority for experimental claims.

Evidence levels matter. The core empirical papers were inspected in their relevant full-text sections. Neighboring methods and foundational/context sources were checked at abstract, method, documentation, or bibliographic level, as recorded individually in `literature_sources.tsv`. A preprint is labeled as such. Acceptance mentioned only by authors is identified as author-reported. Indexed manuscript text with inaccessible delivery is not treated as independently certified numerical evidence.

Narula et al. [13] and the anonymous replay submission [62] have limited-access qualifications. The historical RTE chapter [38] is bibliographic context, not a source of detailed empirical claims here. A search/access record accompanies this review in `LITERATURE_SEARCH.md`.

## 2. The project being evaluated

The current suite studies one continually updated adapter or backbone over RTE → MNLI matched → SST-2. Task identity is supplied at evaluation, each task has its own classifier, and an old classifier is frozen after its task. This is task-incremental classification. It is not class-incremental recognition, open-ended generation, or direct measurement of a model's entire pretrained knowledge.

The fixed main design has 4 models × (5 ranks + 1 non-embedding full-FT baseline) × 5 seeds = 120 sequences. Each sequence uses 2,490 training examples per task, three epochs per task, effective batch 16, maximum length 256, q/v adaptation in every layer, alpha = 2r, and ranks 4/8/16/32/64. The prescribed seeds are 42/123/456/789/1024. The current protocol preserves full official validation and records task-boundary artifacts.

The model cards establish the checkpoint families and architecture context [52–55]. The table below uses the suite's pinned registry. Parameter totals are approximate; adapter counts are calculated for q/v factors alone and exclude heads and biases.

| Fixed checkpoint | Approx. parameters | Layers / hidden width | Value-projection width | q/v adapter parameters at r=16 |
|---|---:|---|---:|---:|
| meta-llama/Llama-3.2-1B | 1.24B | 16 / 2,048 | 512 | 1,703,936 |
| Qwen/Qwen2.5-1.5B | 1.54B | 28 / 1,536 | 256 | 2,179,072 |
| ibm-granite/granite-3.3-2b-base | 2.53B | 40 / 2,048 | 512 | 4,259,840 |
| Qwen/Qwen2.5-3B | 3.09B | 36 / 2,048 | 256 | 3,686,400 |

For these square query projections, the adapter count is Lr(3d + wᵥ). Consequently, equal rank is not equal trainable-parameter budget across models. Granite has more q/v adapter parameters than Qwen-3B at the same rank. Grouped-query attention also makes value matrices narrower than query matrices [56]. At r=64, a Qwen value update has rank fraction 64/256 = 25%, whereas its query update occupies a much smaller fraction of its available rank. Report r/min(matrix dimensions), q/v separately, rather than treating r/hidden-width as the only normalized capacity measure.

The Qwen pair provides the strongest family-matched size contrast. It still changes depth, width, and projection geometry and supplies only two size points. Llama's pruning/distillation history and Granite's architecture/training lineage further limit pooled scaling claims. These four models support comparisons of observed behavior, not identification of a universal model-size law. “1B–3B-class” is more accurate than “strictly below 3B.”

## 3. Foundations: what forgetting means

McCloskey and Cohen [1] provide the historical sequential-interference context. EWC [2] protects important parameters through a regularized training objective, while the continual-learning taxonomy and survey [3,4] show why task identity, output heads, data access, and task order must be specified before comparing methods. These are framing sources rather than rank-selection prescriptions.

GEM [5] motivates the accuracy matrix and backward transfer. Let Aᵢⱼ be accuracy on task j after training task i, with accuracy stored as a fraction. For this three-task study:

> BWT = ½[(A₃₁ − A₁₁) + (A₃₂ − A₂₂)].

Multiply by 100 to report percentage points. Negative BWT means earlier-task performance fell relative to its immediate post-learning accuracy. Positive BWT means backward improvement. Signed −BWT can therefore be negative; it is not an absolute forgetting magnitude.

A single BWT average can hide cancellation: an illustrative +5-point change on RTE and −5-point change on MNLI produce zero BWT. An adapter that never learned either task can also have near-zero BWT. Always inspect both components, all three diagonal accuracies, final average accuracy, and worst-task retention.

The suite already reports nonnegative peak forgetting using the maximum observed accuracy including the final boundary. That answers a different question from BWT when MNLI first improves RTE and SST-2 later removes that improvement. Keep both definitions, with their exact conventions, rather than renaming one as the other.

Freezing old heads prevents direct overwriting of their parameters, but their input representations can move. Poor performance with a frozen old head is a valid operational failure. It does not, by itself, establish that the information is irrecoverably absent from the representation.

## 4. Direct evidence on rank, learning and retention

### 4.1 Large-model adaptation is a useful precedent, not a transferable rank law

Biderman et al. [8] compare LoRA and full fine-tuning in Llama-2-7B code/math settings. They find a learning–retention tradeoff, substantial dependence on training regime, and sensitivity to rank, LR, scaling, and target modules. Their forgetting evaluations emphasize source/general capabilities rather than this project's earlier-task BWT. Their setup supports careful calibration and separate reporting of learning and retention; it does not establish the sign of the present rank–BWT curve.

The important comparison is conditional: given the declared LR, target modules, training budget, and scaling, what does changing rank do? Calling rank the sole manipulated setting is reasonable within a fixed policy, but the resulting treatment bundles capacity with changes in the induced update. Equal training steps also do not guarantee equal optimization progress.

### 4.2 Spectral structure gives a mechanism to test

Shuttleworth et al. [9] analyze intruder singular directions, distinguish LoRA from full fine-tuning, examine sequential RoBERTa classification, and intervene on selected singular components. Their continual-learning and pretraining-retention observations need to be kept distinct. This is the closest mechanistic precedent, including evidence that lower-rank adapters can retain earlier tasks less robustly in their sequential setting.

For the project, the contribution would be testing whether spectral measurements remain informative in these particular compact decoders, with these tasks and frozen heads. A correlation is not a mechanism established anew. A null correlation would narrow the usefulness of the diagnostic under this protocol rather than invalidate every published setting.

### 4.3 Compact-model high-rank results concern a different regime

H-LoRA [10] examines compact MiniMind/Qwen-0.5B specialization and recoverability with much larger rank fractions and a direct additive scale. Its approximately two-thirds-hidden-width guideline is explicitly scoped as an empirical heuristic. Its domains, parameterization, and retention definitions differ from this sequence. It should not be presented as a clean contradiction of the 7B studies or as a reason to substitute r≈512 into the main sweep.

The present r=4–64 study can detect low-rank underlearning and a feasible set inside that tested grid. It cannot test a high-rank optimum outside the grid. An optimum at an endpoint is evidence that the search boundary may matter, not proof of a global optimum.

### 4.4 Closely related studies constrain novelty

Rathore et al. [11] study rank-dependent behavior in larger Q&A models and include update-norm analysis. Patil et al. [12] already evaluate Llama-3.2-1B in sequential NLP tasks, using isolated stacked adapters and task-specific activation. Their method changes the memory and parameter-isolation assumptions. Some reported computational comparisons are estimates, so they should not be treated as measured rental-time guarantees for this suite.

Newer evidence makes a blanket “1B–3B rank/retention research does not exist” claim untenable. Choi [60] uses Llama-3.2-1B and Qwen2.5-1.5B in continual factual updating with replay-capacity and evaluation-time comparisons. This is a recent preprint and a different task setting, but it directly overlaps two model choices and the capacity–retention question. Its relevance is reporting the whole evaluation region rather than selecting one favorable endpoint.

CoDyRA's updated manuscript [29] examines rank and placement, includes a norm-matched probe, and extends to LLM continual learning. Cite its current v9 title and date, not only the earlier vision-language title. Marchetta et al. [30] provide a September 2026 dynamical theory in a solvable teacher–student model, with qualitative sequential-MNIST evidence. Its rank-dependent interference account supplies hypotheses, not a theorem about modern decoder classification.

The indexed Narula manuscript [13] warns about seed-dependent forgetting and describes some rank claims as single-seed hypotheses. Direct delivery remained restricted in this review. Do not repeat its numerical anecdote as independently verified, or call the paper formally retracted without evidence. Five-seed replication has independent support from the fine-tuning variability literature [41,42].

<!-- landscape -->

### Table 1. Main empirical and theoretical precedents

| Source | Relevant setting | What it contributes | Boundary for this project |
|---|---|---|---|
| Biderman et al. [8] | 7B code/math adaptation | Learning and source-retention tradeoffs depend on the training policy. | Source capability loss differs from sequential task-head BWT. |
| Shuttleworth et al. [9] | Spectra, RoBERTa sequences, 7B adaptation | Intruders, sequential retention, and spectral intervention. | Compact decoder behavior and this head policy remain empirical. |
| Lun and Ming [10] | Sub-1B domain specialization | High-rank capacity and recoverability. | Different rank fractions, scaling and outcomes. |
| Rathore et al. [11] | Larger Q&A models | Task-dependent rank behavior and update magnitudes. | Does not supply this exact classification protocol. |
| Patil et al. [12] | Llama-3.2-1B sequential NLP | Same small-model family already studied in lifelong adaptation. | Stacked, isolated adapters differ from one continually updated adapter. |
| Lu et al., CoDyRA [29] | Rank/placement probes and continual adaptation | Capacity control, norm matching, adaptive rank selection. | Adaptive methods do not isolate the present fixed-rank policy. |
| Marchetta et al. [30] | Two-task solvable model; sequential MNIST | Dynamical hypotheses for interference and capacity saturation. | Recent theory preprint; assumptions do not match a full decoder. |
| Lee et al. [32] | LR sweeps over LoRA methods and model scales | A warning against under-tuned baselines and fixed-LR method rankings. | No exact optimum established for these heads/tasks. |
| Choi [60] | Two fixed project models; temporal factual updates | Capacity/endpoint-conditioned comparisons and cost frontiers. | Replay and factual updating differ from rehearsal-free classification. |

<!-- portrait -->

## 5. Neighboring methods: what they change

LoRA [6] constrains each adapted weight increment to a factorized update, while intrinsic-dimensionality work [7] helps motivate restricted adaptation without specifying an optimal LoRA rank. A useful review must distinguish a study of rank under vanilla LoRA from a competition among methods that alter the training objective, parameterization, initialization, or access to old data.

<!-- landscape -->

### Table 2. Continual-learning neighbors and their tradeoffs

| Literature | Mechanism | Advantage relevant to the project | Disadvantage or comparison caveat |
|---|---|---|---|
| O-LoRA [21] | Separate orthogonal task subspaces. | Directly targets interference without replay. | Adds isolation/constraints; not the same shared-adapter rank treatment. |
| CLoRA [22] | Subspace regularization of adaptation. | Connects output change and retention. | Regularization strength becomes another explanation for a result. |
| Subspace-denoised SLoRA [23] | Filter update components by base-subspace similarity. | Motivates examining update directions and energy. | Filtering changes the trained solution; keep it outside the vanilla main grid. |
| Forgetting-aware pruning [24] | Prune using update/base-weight relationships. | Motivates normalized displacement measurements. | Post-training modification is a different intervention; elementwise ratios are unstable near zero weights. |
| InfLoRA [25] | Interference-aware low-rank subspaces. | Gives an alternative geometric account. | Primarily a vision continual-learning setting. |
| SD-LoRA [26] | Decoupled magnitude/direction learning. | Addresses scalable stability–plasticity balance. | Class-incremental evaluation and a changed method limit direct comparison. |
| ASO-LoRA [27] | Soft orthogonality and task-related sharing. | Highlights beneficial transfer as well as protection. | Uses additional task-structure assumptions/modules. |
| SpaRTA [28] | Shared/specific branches and routing. | Distinguishes shared knowledge from task-specific capacity. | Extra branches/router change the capacity budget and inference policy. |
| CoDyRA [29] | Adaptively select/minimize active ranks. | Makes capacity allocation an explicit retention control. | Dynamic ranks answer a different question from a fixed-rank ablation. |
| Janus-LoRA [31] | Correct interacting factor updates and improve feature separation. | Warns that factor-level conditions need not protect the composite update. | Adds objectives and optimization machinery; evidence is not this exact NLP design. |
| Stacked LoRA [12] | Frozen task-specific adapter composition. | Preserves independent task modules. | Storage grows with tasks and task-specific activation changes the evaluation contract. |
| Memory-as-Markov-matrix [61] | New-token memory/embedding mechanism. | Broadens the interpretation of recoverability. | Different adaptation target; not a replacement for the fixed non-embedding protocol. |

<!-- portrait -->

“SLoRA” is an ambiguous abbreviation: Stacked LoRA [12] and subspace-denoised SLoRA [23] are different methods. O-LoRA [21] is also distinct from similarly named orthogonal-initialization variants. Use full titles in the thesis to prevent accidental citation substitution.

The anonymous TF-SSD manuscript [62] mentions Granite3.3 continual fine-tuning, but its publication status and numerical findings were not certified here. It is a novelty-screening lead, not evidence for adopting teacher ensembles. TRACE [34] provides broader heterogeneous generative continual-learning context; adding its full benchmark would greatly expand this FYP rather than merely improve its present measurements.

### Table 3. Parameterization and initialization alternatives

| Method | Potential advantage | Cost or limitation | Recommended role |
|---|---|---|---|
| Rank-stabilized LoRA [14] | Makes the rank-scaling policy explicit. | Changes the treatment and needs matched calibration. | Small separate robustness cohort if affordable. |
| LoRA+ [15] | Different A/B rates can improve optimization. | Introduces a rate ratio; must tune fairly. | Future work after basic LR/head checks. |
| Initialization analysis [16] | Shows zero-A and zero-B starts are not interchangeable. | Initialization changes can alter optimization and spectra. | Record and retain the current random-A/zero-B policy. |
| AdaLoRA [17] | Allocates limited capacity across matrices. | Dynamic allocation prevents a simple fixed-rank interpretation. | Neighboring work; not a main-grid replacement. |
| DoRA [18] | Separates weight magnitude and direction. | Changes trainable structure and update geometry. | Optional later method comparison. |
| PiSSA [19] | Uses spectral initialization. | Directly changes the spectral starting point under investigation. | Do not combine with the vanilla intruder hypothesis unnoticed. |
| QLoRA [20] | Reduces backbone memory with weight quantization. | Introduces a quantization interaction. | Keep outside the unquantized main study. |

## 6. Hyperparameters: reliable starting points versus established optima

No reviewed primary source establishes optimal hyperparameters for these exact four pinned checkpoints, task-specific heads, sampled training sets, sequence, optimizer policies, and training budget. “Reliable” should mean justified, reproducible, and checked for gross failure, not guaranteed optimal.

The methodology's nominal LoRA rates are 2.00e-4 / 1.85e-4 / 1.60e-4 / 1.50e-4, with full-FT rates ten times smaller. Its N^(−1/3) rule is an author-designed interpolation inspired by pretraining schedules. GPT-3's table [50] does not derive that exponent for fine-tuning. QLoRA's successful settings [20] use different quantized backbones and objectives. Neither validates a four-model fine-tuning transfer law. The printed rates can remain declared priors while their adequacy is checked.

Lee et al. [32] show why method rankings are sensitive to LR tuning. The practical inference is to spend calibration effort on LR and head learning before introducing many adapter variants. An LR comparison on one RTE seed does not demonstrate plateau membership or adequacy for every later task. Literature-derived settings also need separate calibration for the full-FT baseline; a weak baseline is not a dependable plasticity ceiling.

With W = W₀ + sBA and s = alpha/r, setting alpha = 2r makes s constant. It does not equalize the induced movement in W. An illustrative first-step SGD calculation, with B initially zero and independent A entries of variance v, gives an expected update proportional to eta × s² × r × v × G. This is a calculation under stated simplifications, not an Adam theorem. Adaptive moments and feature evolution further complicate the relationship. The methodology's own honesty note recognizes this; its claim that the issue is “neutralized” should be softened.

The primary estimand can remain the effect of rank under vanilla scaling. Measure update magnitude rather than pretending it is controlled away. If scaling robustness is added, anchor a separate rsLoRA cohort at r=16: fixed alpha=8 with alpha/sqrt(r) gives the same coefficient 2 at r=16. The matching anchor is this review's design choice, not a published optimum.

AdamW [58] and 8-bit-state optimization [48] are relevant to the declared baseline policy. The suite uses fp32 LoRA moments and quantized matrix moments in full FT. That saves memory but prevents a pure adaptation-method comparison independent of optimizer precision. Low-precision storage can cancel small updates [49]; retaining fp32 trainable parameters while using bf16 computation is scientifically preferable to silently weakening the baseline to fit a smaller GPU.

MuP-style hyperparameter transfer [51] depends on its parameterization assumptions. It cannot be applied retrospectively to these independently pretrained model families merely by substituting a different size exponent. Keep the family-matched contrast descriptive and calibrate the actual released checkpoints.

## 7. Behavioral, spectral and statistical measurement

### 7.1 Measure learning and retention jointly

Retain BWT, all accuracy-matrix cells, peak forgetting, final-row average, and all task diagonals. Add the three transition changes A₂₁−A₁₁, A₃₁−A₂₁, and A₃₂−A₂₂. They separate NLI-to-NLI transfer from subsequent sentiment interference without another model forward pass.

Compute balanced accuracy, macro-F1, confusion matrices, and per-class retention from saved classes and labels. Net accuracy can hide individual examples being forgotten while others are recovered. For each old task, report both correct→incorrect and incorrect→correct rates, using identical example IDs/order across boundaries. Never infer identities from row order unless the provenance confirms it.

GLUE, MultiNLI and SST supply established tasks [35–38], but a 2,490-example sequence is a project-specific small-data protocol. NLI annotation artifacts and HANS [39,40] limit claims about semantic reasoning. An optional fixed MNLI-mismatched/HANS evaluation would test robustness, with an explicitly declared mapping from MNLI contradiction/neutral to HANS non-entailment. This adds evaluation cost and does not replace the main validation sets.

Tao et al. [33] motivate refitting a simple classifier on frozen representations to assess recoverability. Keep ordinary frozen-head BWT primary. A probe that recovers accuracy shows recoverable information under that probe/training budget; it does not erase the operational failure. Fit only on training data, preserve an independent selection split, and label the probe result as a separate diagnostic.

If those features are captured, centered kernel alignment (CKA) [46] can describe representation similarity across boundaries on the same fixed examples. Use identical layers, pooling and example order. CKA is a complementary similarity measurement, not a substitute for accuracy or evidence that a particular fact was preserved. Neither feature recovery nor CKA can be recovered from q/v snapshots alone for a completed full-FT run.

### 7.2 Make the spectral hypothesis falsifiable

The current intruder statistic compares each of the top ten tuned left singular vectors with all pretrained left vectors and applies an absolute-cosine threshold [9]. Preserve that definition and its four thresholds as the preregistered outcome. Nonzero counts in full FT are possible observations; do not tune the detector until the baseline reaches zero.

A vector can have low similarity to every particular basis vector while remaining inside the same span. In a complete output basis, an assertion that an intruder lies outside the entire span would be mathematically wrong. Degenerate or nearly degenerate singular values can also rotate individual vectors without a comparable change of the leading subspace. Therefore complement counts with principal-angle/subspace distance [47], continuous similarities, and the gap between the last inspected singular value and the next one.

For fixed k, a basis-invariant leading-subspace diagnostic is:

> Dₖ = 1 − ||U₀,ₖᵀ Uₜ,ₖ||²_F / k.

Here both matrices contain k leading orthonormal left directions. Do not use all reference directions in this subspace distance: a complete basis would make the comparison trivial. Compare k=10 with a separately labeled k=20 sensitivity analysis; record near-ties at the cutoff.

Measure the merged update rather than the separate factor norms, since BA is invariant to reciprocal rescaling of B and A. Report ||ΔW||_F / ||W₀||_F, ||ΔW||₂, stable rank, energy concentration, and entropy effective rank [45]. For nonzero updates, define pᵢ = σᵢ/Σσ and erank = exp(−Σpᵢ ln pᵢ). Treat an exactly zero update separately. A low effective-rank/nominal-rank ratio indicates unused algebraic capacity, not automatically a failed learner.

For LoRA, two thin QR decompositions reduce the update singular-value problem to an r×r matrix. This is an algebraic implementation opportunity: it avoids decomposing a full dense update merely to measure effective rank. Full-FT q/v snapshots support the analogous diagnostics with dense calculations. Report q and v separately and normalize depth when making descriptive cross-model plots.

An energy-weighted intruder fraction should state its denominator. Weighting intruder indicators by squared singular values and dividing by the energy of the inspected top-k directions measures their share within that inspected set. It is not the fraction of total matrix energy. Report top-k energy coverage as well, and retain the unweighted count for comparability.

### 7.3 Quantify uncertainty at the right level

Dodge et al. [41] and Mosbach et al. [42] establish that initialization, data order, optimization failure and generalization variation matter in small-data fine-tuning. These findings motivate multiple seeds and healthy-learning checks; their BERT experiments do not dictate an exact variance for this suite.

Five fixed seeds are a budget-conscious replication choice. They estimate initialization/shuffle variability on fixed examples and one task order. They do not measure variability over new dataset draws, all task orders, or architectures. Retain sample SD, paired seed differences, individual points and uncertainty intervals. Pair only genuinely matching seed/configuration runs; common seed numbers do not make different-rank adapter arrays identical.

Following the paired-evaluation principles discussed by Dror et al. [43], distinguish seed uncertainty from finite-validation uncertainty. Resample matched seeds as blocks and matched examples within tasks, preserving the same sampled examples across compared ranks and boundaries. A hierarchical bootstrap can be useful descriptively, but five top-level seeds provide weak tail estimation. State assumptions for a paired t interval; use multiplicity correction for formal families of rank comparisons.

The current max(2pp, 2×SD) feasibility margin can reward noisy configurations. A fixed, declared tolerance and separate uncertainty are easier to interpret. A 2pp tolerance is operational, not a validated non-inferiority test. On 277-example RTE validation, one prediction is approximately 0.361pp; small differences require careful paired analysis. Five same-direction paired effects give a minimum two-sided exact sign-test p-value of 2/32 = 0.0625. This does not preclude other tests under stronger assumptions, but it illustrates the limited resolution.

For mechanistic associations, each run is the behavioral unit. Layers and matrices from one run are not independent BWT observations. Analyze within model and LR; inspect associations across ranks separately from residual seed variation. Use normalized spectral summaries for exploratory pooling, cluster uncertainty by seed/run as appropriate, and avoid causal language for a pooled Spearman coefficient.

Calibration measures such as NLL and Brier score can reveal confidence damage hidden by accuracy; ECE is sensitive to bins and sample size [44]. The current `.npz` artifacts contain class predictions and labels only. NLL/Brier/ECE therefore require saving logits/probabilities during evaluation or rerunning inference. They cannot be reconstructed from argmax classes.

## 8. Research gap, claim corrections and testable questions

The defensible gap is a particular controlled comparison, not the absence of all small-model continual-learning research. A suitable contribution statement is: “We characterize vanilla q/v LoRA rank under a fixed sequential classification policy in four specified 1B–3B-class base checkpoints, using paired seeds, behavioral learning/retention measurements, and boundary-wise spectral diagnostics.” This is a proposed wording, not a claim that the search established priority over every unpublished study.

Frame the research questions without presupposing the answers. RQ1 should ask how rank changes each task's learning and retention under the chosen policy. RQ2 should ask which tested ranks satisfy declared learning and retention tolerances, allowing disconnected or empty sets. RQ3 should ask whether spectral change explains variation beyond update magnitude and nominal rank. RQ4 should ask how the observed sets differ across the four checkpoints, treating the Qwen pair as the primary family-matched contrast.

### Table 4. Corrections needed when writing the thesis

| Existing claim or ambiguity | Evidence-based revision |
|---|---|
| “First rank–forgetting study below 3B”; the whole band is unexplored. | Related small-model sequential and rank/capacity work already exists [12,60]. State the exact protocol contribution and search limits. |
| Higher rank must cause more forgetting, with compact studies proving the opposite. | These settings measure different outcomes and use different parameterizations. Test the sign, shape and learning adequacy rather than imposing a directional result. |
| Shuttleworth et al. is an ICLR 2025 paper or has no sequential rank evidence. | The verified publication is NeurIPS 2025. Its RoBERTa continual-classification experiment is directly relevant [9]. |
| Alpha = 2r makes effective update size identical at every rank. | It fixes the forward coefficient. The induced update remains rank- and optimizer-dependent; the methodology's honesty note already partly acknowledges this. |
| Intruders are necessarily directions outside the pretrained span. | The detector measures failure to match individual reference vectors. Complement it with leading-subspace and spectral-gap analysis. |
| Full FT has zero intruders by definition and is always the learning ceiling. | Both are empirical outcomes. The present baseline also freezes embeddings and uses a different moment-precision policy. |
| A large seed SD justifies a larger equivalence margin. | Keep practical tolerances fixed and display uncertainty separately. The current SD-dependent window is descriptive, not a formal equivalence result. |
| No loss in BWT means successful retention. | Check all diagonals, class baselines, per-task retention and final performance. Underlearning and cancellation can yield near-zero BWT. |
| Qwen-3B is strictly below 3B and shares an Apache license with every Qwen. | Its card lists 3.09B and the Qwen Research License [54]. Keep the checkpoint and use accurate scope wording. |
| Granite 3.3-2B was released in December 2024. | The official base card gives 16 April 2025 [55]. |
| The sanity battery is 12 runs, or main contains 72 runs. | The executable full design is 120 sequences; its present sanity battery has 15 unique single-task runs. |
| The Narula anecdote establishes a retracted universal rank relationship. | Direct numerical/withdrawal verification remains limited. Treat it as a qualified lead and use independent variability evidence [41,42]. |

Hwang et al. [57] provide relevant sub-3B NLI feasibility evidence under their own short training and freezing policies. They do not demonstrate this entire sequential grid's runtime, optimal LR or baseline adequacy. Similarly, Pethkar et al. [61] study a different memory mechanism; their claims are not a guarantee that an updated shared classifier retains earlier tasks. Scope each related result to its measured behavior.

## 9. Improvement tables: exact proposals, benefits and disadvantages

Priority P1 means the best return for this FYP; P2 means a targeted follow-up after the core design is sound; P3 means optional future work. “New” identifies work absent from the reviewed v2 baseline. “Extend” means that baseline already supplies part of the measurement. The subsequent v3 implementation is identified in the executive assessment and protocol amendment. These are design recommendations and candidate values, not claims of demonstrated improvement.

Cost categories refer to additional work: CPU means analysis of already saved artifacts; evaluation means additional forward passes; calibration means isolated training-only checks; cohort means additional experimental training. Actual dollars require measured timings and the current rental quote. Preserve existing primary outcomes even when adding richer secondary measurements.

<!-- landscape -->

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

<!-- portrait -->

These are candidate search points, not a revised declaration of the best LR. If an adopted calibration selects a different baseline rate, update the preregistered protocol, configuration and baseline-analysis rule before main training; the current executable window rule anchors to the current nominal rate.

## 10. Recommended plan within a cost-conscious budget

### 10.1 Preserve the main comparison

Keep the selected checkpoints, full five-rank grid, five seeds, complete evaluation and task-boundary exports. Keep the 120-sequence accounting explicit. Do not save money by deleting inconvenient seeds, shrinking evaluation after inspecting results, mixing numerical policies across ranks, or relabeling a reduced-data smoke test as a main experiment.

The suite already saves time through reusable full-length pilots, cached data, task-boundary recovery, independent seed-block workers and deferred CPU spectra. Full validation and export I/O count toward rental time. The existing validation record contains CPU/software checks and synthetic spectral timings, not production-size GPU throughput or experimental accuracy.

### 10.2 Buy information in stages

Stage A is low-cost preparation: verify artifact requirements, perform the training-only token/truncation audit, add desired logits or feature capture before data collection, and implement M01–M05/M07–M08 reporting. These measurements do not require expanding the training grid. Record exact definitions and denominators before inspecting final results.

Stage B is controlled calibration. An expanded H01 battery has four models × [(two LoRA rank endpoints × three rates) + (one full-FT track × three rates)] = 36 single-task runs at one declared seed. With 1,992 training examples, effective batch 16 and three epochs, each has 375 optimizer updates: 13,500 total. That is about 8.0% of the main grid's 168,480 update count, not an 8.0% wall-time or dollar guarantee. Full FT and different sequence lengths have different per-update costs.

Use the isolated training holdout, choose a declared model/track LR policy rather than tuning each main rank independently, and confirm any strong finding with another calibration seed if affordable. A smaller r=16-only screen is cheaper but cannot check rank endpoints. State that limitation instead of calling it a proven plateau. Do not reuse a full-data main pilot as a training-holdout calibration run: its training exposure is different.

Stage C is the full main experiment, after freezing all adopted controls. Stage D is CPU analysis and a limited M06 head-recovery diagnostic. Add one robustness cohort only if the main measurements identify a specific uncertainty: scaling, task order, baseline moment precision, or high-rank boundary. Report such follow-ups as exploratory if chosen after viewing main results.

### 10.3 Measure cost rather than promise a rental total

Use cost = actual hourly rate × instance wall hours + storage/transfer/other billed charges. Measure the largest model/track first for fit and throughput. On a multi-GPU instance, sum non-overlapping instance billing time, not overlapping per-worker elapsed times multiplied by the full instance price. More GPUs shorten wall time only when concurrency and price make the total cost favorable.

Keep spectra on CPU after exporting verified artifacts. Existing q/v snapshots support spectral diagnostics, but they are not complete full-FT backbones. Representation recovery, external-text LM evaluation and causal weight interventions require full models at the chosen boundary. Compute those while the model is available, or explicitly retain full-FT weights with `--save-full-weights` and budget the disk/transfer/inference cost. LoRA boundaries can be reconstructed from the pinned base and saved adapter.

The initial $25 is a planning allowance, not an experimental validity condition or a demonstrated cost estimate. Freeze the scientific scope first and decide which optional extensions to fund using measured run costs. A budget stop should leave a visibly incomplete study, not a partial report presented as a complete feasible-window result.

## 11. What to implement first and what to report

My recommended first amendment combines M01–M05 and M07–M08, a small H01/H02 calibration, and H12 throughput checks. M06 is the most informative additional mechanism diagnostic if the heads or old-task accuracies behave unexpectedly. It should be planned before main collection because full-FT reconstruction and feature capture affect artifact requirements.

Use a supplementary all-task feasibility definition alongside the current preregistered one, not a hidden replacement chosen because it produces a nicer window. Keep window membership over the discrete tested ranks; do not interpolate across gaps or present all intermediate integer ranks as measured. Report raw rank, per-matrix rank fraction, and trainable count together.

A useful, explicitly exploratory mechanistic extension is to attenuate selected intruder components in final q/v matrices, alongside matched-norm control directions. Compare prior-task retention and current-task accuracy against the unmodified checkpoint. This requires full reconstructable models and extra inference, not simply the saved count table. The intervention is a follow-up to published spectral work [9]; it must be listed as a separate diagnostic, not called a new vanilla training result. Its advantage is stronger mechanistic evidence than correlation; its disadvantages are added reconstruction/evaluation cost and sensitivity to direction/control selection.

For a final thesis, report the complete accuracy matrix and seed points; per-task learning/retention and fixed tolerances; nominal and effective update capacity; CPU spectral definitions and numerical checks; within-model associations with uncertainty; the Qwen family contrast and cross-family limitations; and measured resource use. Mark published findings, mathematical deductions, author-selected candidate settings, and actual project observations separately. At present, none of the proposed rank optima, accuracy gains, GPU-hour totals or causal effects has been established by this FYP.

## References

Numbered references are keyed to the accompanying [source ledger](literature_sources.tsv). Access qualifications and manuscript versions are preserved there; contextual or restricted sources are not represented as verified full-paper experiments.
