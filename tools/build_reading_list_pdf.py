"""Build docs/reading-list.pdf: research papers directly relevant to the LoRA rank x forgetting project."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parents[1] / "docs" / "reading-list.pdf"
INK, MUTED, LINK = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), "#1c5cab"

ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=18, leading=22, alignment=TA_LEFT, textColor=INK, spaceAfter=4)
SUB = ParagraphStyle("sub", parent=ss["Normal"], fontSize=9.5, leading=13, textColor=MUTED, spaceAfter=10)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13, leading=16, textColor=INK, spaceBefore=12, spaceAfter=4)
INTRO = ParagraphStyle("intro", parent=ss["Normal"], fontSize=9, leading=12.5, textColor=MUTED, spaceAfter=6)
TITLE = ParagraphStyle("t", parent=ss["Normal"], fontSize=9.8, leading=13, textColor=INK, fontName="Helvetica-Bold")
META = ParagraphStyle("m", parent=ss["Normal"], fontSize=8.6, leading=11.5, textColor=MUTED)
WHY = ParagraphStyle("w", parent=ss["Normal"], fontSize=9, leading=12.5, textColor=INK, spaceAfter=7)

# (title, authors, venue, status, url, why it matters for this project)
SECTIONS = [
    ("1. LoRA rank, learning and forgetting (core to the research question)",
     "Direct evidence on how LoRA rank relates to learning and forgetting.", [
        ("MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning", "Jiang et al.",
         "arXiv preprint, 2024", "Preprint", "https://arxiv.org/abs/2405.12130",
         "Argues low-rank updates limit learning of new knowledge and gives a high-rank alternative at equal "
         "parameter count; the opposite position to the flat rank curve in v4, useful for the discussion."),
        ("LoRA Training in the NTK Regime has No Spurious Local Minima", "Jang, Lee, Ryu", "ICML, 2024 (oral)",
         "Peer-reviewed", "https://arxiv.org/abs/2402.11867",
         "Proves LoRA with rank around the square root of the number of training examples avoids bad local minima "
         "on NLP classification; a theoretical lower bound on the rank needed, relevant to why small ranks suffice."),
        ("Revisiting Catastrophic Forgetting in Large Language Model Tuning", "Li, Ding, Fang, Tao",
         "Findings of EMNLP, 2024", "Peer-reviewed", "https://aclanthology.org/2024.findings-emnlp.249/",
         "Links forgetting in LLM fine-tuning to the sharpness of the loss landscape; a mechanism for why the "
         "learning rate, not rank, drove forgetting in v4."),
        ("An Empirical Investigation of the Role of Pre-training in Lifelong Learning",
         "Mehta, Patil, Chandar, Strubell", "JMLR 24(214), 2023", "Peer-reviewed",
         "https://jmlr.org/papers/v24/22-0496.html",
         "Pre-trained models forget less in task sequences (15 NLP tasks) because they sit in wider minima; "
         "background for the model-size and family differences in v4."),
        ("Scaling Laws for Forgetting When Fine-Tuning Large Language Models", "Kalajdzievski",
         "arXiv preprint, 2024", "Preprint", "https://arxiv.org/abs/2401.05605",
         "Forgetting follows a power law in the number of fine-tuned parameters and steps, and LoRA still forgets: "
         "a quantitative frame for rank as 'parameters fine-tuned'."),
        ("Adaptive Rank, Reduced Forgetting: Continual Learning with Dynamic Rank-Selective LoRA (CoDyRA)",
         "Lu, Zhao, Xue, Yao, Moore, Gong", "arXiv preprint, 2024–2026", "Preprint",
         "https://arxiv.org/abs/2412.01004",
         "Higher rank gives more plasticity and more forgetting in continual learning; motivates rank as a "
         "stability–plasticity lever."),
        ("Subspace Geometry Governs Catastrophic Forgetting in Low-Rank Adaptation", "Steele",
         "arXiv preprint, 2026", "Preprint (single author)", "https://arxiv.org/abs/2603.02224",
         "Argues forgetting is set by the angle between task-gradient subspaces and is nearly rank-invariant: one "
         "explanation for the flat BWT-vs-rank curve in v4."),
        ("Understanding Catastrophic Forgetting in LoRA via Mean-Field Attention Dynamics", "(2024 authors)",
         "arXiv preprint, 2024", "Preprint", "https://arxiv.org/abs/2402.15415",
         "A theoretical account of how LoRA updates to attention cause forgetting; relevant because v4 adapts q/v "
         "attention matrices."),
        ("On Catastrophic Forgetting in Low-Rank Decomposition-Based Parameter-Efficient Fine-Tuning",
         "(2026 authors)", "arXiv preprint, 2026", "Preprint", "https://arxiv.org/abs/2603.09684",
         "Compares LoRA-family methods under sequential training and finds low ranks forget most; a counterpoint "
         "to discuss (vision setting)."),
        ("Rethinking the Rank Threshold for LoRA Fine-Tuning", "Park", "arXiv preprint, 2026",
         "Preprint (single author)", "https://arxiv.org/abs/2605.03724",
         "Theory and experiments suggesting r = 1 suffices for binary GLUE tasks, while MNLI needs more; matches the "
         "v4 observation that r = 1 already learns these tasks."),
    ]),
    ("2. Learning rate and the LoRA parameterisation (why v3 was confounded)",
     "Papers that explain why rank cannot be compared without controlling the learning rate.", [
        ("Learning Rate Scaling across LoRA Ranks and Transfer to Full Finetuning (μA)", "Chen, Villar, Hayou",
         "ICML (PMLR v306), 2026", "Peer-reviewed", "https://arxiv.org/abs/2602.06204",
         "Shows the optimal LR is roughly rank-invariant with α/r scaling but shrinks with rank under a constant "
         "multiplier (the v3 α = 2r setting). The key citation for the v3 → v4 change."),
        ("A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA (rsLoRA)", "Kalajdzievski",
         "arXiv preprint, 2023", "Preprint", "https://arxiv.org/abs/2312.03732",
         "Proposes α/√r so high ranks do not slow learning; an alternative parameterisation to discuss."),
        ("LoRA+: Efficient Low Rank Adaptation of Large Models", "Hayou, Ghosh, Yu", "ICML, 2024",
         "Peer-reviewed", "https://arxiv.org/abs/2402.12354",
         "Different LRs for the A and B factors; relevant to how LoRA's optimisation depends on its parameterisation."),
        ("LoRA Without Regret", "Schulman and Thinking Machines Lab", "Technical blog, 2025",
         "Not peer-reviewed", "https://thinkingmachines.ai/blog/lora/",
         "With α fixed, the optimal LR varies under 2× from rank 4 to 512; all-linear LoRA beats attention-only. "
         "Supports both the v4 parameterisation and the all-linear decision."),
        ("Simple and Scalable Strategies to Continually Pre-train Large Language Models",
         "Ibrahim, Thérien, Gupta, et al.", "TMLR, 2024", "Peer-reviewed", "https://arxiv.org/abs/2403.08763",
         "Learning-rate re-warming at each new stage causes forgetting, and replay plus schedule changes reduce it; "
         "directly relevant to the per-task LR restart proposed for the next phase."),
        ("Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning", "Lee, Ko, Chen, Yeh",
         "arXiv preprint, 2026", "Preprint", "https://arxiv.org/abs/2602.04998",
         "Comparisons between LoRA configurations can flip once the LR is tuned; justifies the per-rank LR search."),
    ]),
    ("3. Model scale and forgetting (the sub-3B question)",
     "Evidence on how model size changes forgetting, relevant to RQ2/RQ4.", [
        ("An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning",
         "Luo, Yang, Meng, Li, Zhou, Zhang", "arXiv preprint, 2023 (revised 2025)", "Preprint",
         "https://arxiv.org/abs/2308.08747",
         "Forgetting measured across 1B–7B decoders; reports more severe forgetting at larger scale, a useful "
         "contrast to the Qwen 1.5B → 3B result in v4."),
        ("Scaling Laws for Forgetting during Finetuning with Pretraining Data Injection", "Bethune et al.",
         "arXiv preprint, 2025", "Preprint", "https://arxiv.org/abs/2502.06042",
         "Forgetting scaling laws across several model sizes, and how mixing in pretraining data reduces it; "
         "relevant to size effects and to cheap mitigations."),
        ("Effect of Scale on Catastrophic Forgetting in Neural Networks", "Ramasesh, Lewkowycz, Dyer",
         "ICLR, 2022", "Peer-reviewed", "https://openreview.net/pdf/54d452543db390a26979d51399ae9c6ed01f4de1.pdf",
         "Larger pretrained models forget less, linked to more orthogonal representations; theory for why "
         "Qwen2.5-3B forgot least."),
        ("The Importance of Being Lazy: Scaling Limits of Continual Learning", "(ICML 2025 authors)", "ICML, 2025",
         "Peer-reviewed", "https://arxiv.org/abs/2506.16884",
         "Whether scale reduces forgetting depends on the parameterisation (it does under NTP, not under μP); "
         "relevant to interpreting size effects."),
    ]),
    ("4. What 'forgetting' really is (supports the probe finding)",
     "Papers showing that measured forgetting is often misalignment, not lost knowledge.", [
        ("Spurious Forgetting in Continual Learning of Language Models", "Zheng, Cai, Qiu, Ma", "ICLR, 2025",
         "Peer-reviewed", "https://arxiv.org/abs/2501.13453",
         "Much apparent forgetting is lost task alignment rather than lost knowledge, which matches the v4 linear-probe "
         "result (head forgetting ≫ representation forgetting)."),
        ("Probing Representation Forgetting in Supervised and Unsupervised Continual Learning",
         "Davari, Asadi, Mudur, Aljundi, Belilovsky", "CVPR, 2022", "Peer-reviewed",
         "https://openaccess.thecvf.com/content/CVPR2022/html/Davari_Probing_Representation_Forgetting_in_Supervised_and_Unsupervised_Continual_Learning_CVPR_2022_paper.html",
         "The linear-probe method used in v4 to separate representation forgetting from head misalignment."),
        ("Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution (LP-FT)",
         "Kumar, Raghunathan, Jones, Ma, Liang", "ICLR, 2022", "Peer-reviewed", "https://arxiv.org/abs/2202.10054",
         "Training a randomly initialised head distorts features; LP-FT is the proposed fix for the head policy in "
         "the next phase."),
        ("Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning",
         "Aghajanyan, Zettlemoyer, Gupta", "ACL, 2021", "Peer-reviewed", "https://aclanthology.org/2021.acl-long.568/",
         "GLUE-type tasks need very few parameters, and fewer for larger models; explains why rank barely affects "
         "accuracy on RTE/MNLI/SST-2."),
    ]),
    ("5. LoRA-based continual learning of LLMs (methods and baselines)",
     "Methods your results can be compared with or extended towards.", [
        ("SAPT: A Shared Attention Framework for Parameter-Efficient Continual Learning of LLMs",
         "Zhao et al.", "ACL, 2024", "Peer-reviewed", "https://aclanthology.org/2024.acl-long.625/",
         "Parameter-efficient continual learning across 770M–13B models; a strong baseline for longer task sequences."),
        ("InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning", "Liang, Li", "CVPR, 2024",
         "Peer-reviewed", "https://arxiv.org/abs/2404.00228",
         "Constrains each task's LoRA update to a subspace that does not interfere with earlier tasks; a principled "
         "single-adapter alternative to compare with plain sequential LoRA."),
        ("Merge before Forget: A Single LoRA Continual Learning via Continual Merging (SLAO)", "Qiao, Mahdavi",
         "ICLR, 2026", "Peer-reviewed", "https://arxiv.org/abs/2512.23017",
         "Keeps one LoRA across tasks (like v4) but merges updates with orthogonal initialisation; tested on Llama "
         "models with constant memory."),
        ("Mitigating Catastrophic Forgetting in Large Language Models with Self-Synthesized Rehearsal",
         "Huang et al.", "ACL, 2024", "Peer-reviewed", "https://aclanthology.org/2024.acl-long.77/",
         "Replay without the original data, using model-generated examples; a cheap mitigation baseline for the "
         "next phase."),
        ("Continual Learning with Low Rank Adaptation (CoLoR)", "Wistuba, Sivaprasad, Balles, Zappella",
         "NeurIPS Workshop on Distribution Shifts, 2023", "Peer-reviewed (workshop)",
         "https://arxiv.org/abs/2311.17601",
         "Shows LoRA-based continual learning beats prompt-tuning methods at similar parameter cost."),
        ("Analyzing and Reducing Catastrophic Forgetting in Parameter Efficient Tuning (I-LoRA)",
         "Ren, Li, Wang, Zhao, Qin", "arXiv preprint, 2024", "Preprint", "https://arxiv.org/abs/2402.18865",
         "Mode connectivity in LLM continual learning with LoRA; interpolation-based method to balance stability and "
         "plasticity."),
        ("Fine-tuned Language Models are Continual Learners", "Scialom, Chakrabarty, Muresan", "EMNLP, 2022",
         "Peer-reviewed", "https://aclanthology.org/2022.emnlp-main.410/",
         "Sequential learning of 8 generation tasks with retention; relevant to the move to generative tasks."),
        ("TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models",
         "Wang et al.", "arXiv preprint, 2023", "Preprint", "https://arxiv.org/abs/2310.06762",
         "A ready-made mixed-task sequence for LLM continual learning, including general-ability retention checks."),
        ("Hyperparameters in Continual Learning: A Reality Check", "Cha, Cho", "TMLR, 2025", "Peer-reviewed",
         "https://arxiv.org/abs/2403.09066",
         "Tuning on the evaluated stream inflates results; supports the separate tuning seeds and training holdout "
         "used in v4."),
        ("Continual Learning of Large Language Models: A Comprehensive Survey", "Shi et al.",
         "ACM Computing Surveys, 2025", "Peer-reviewed (survey)", "https://arxiv.org/abs/2404.16789",
         "Map of the field to position the project and find further related work."),
    ]),
]


def build():
    s = [Paragraph("Reading list: LoRA rank and forgetting in small language models", H1),
         Paragraph("Feasible Rank Window project · prepared 25 October 2026. Papers selected for direct relevance to "
                   "rank × forgetting in sequential fine-tuning of sub-3B decoders. Foundational papers that only "
                   "define methods or metrics (LoRA itself, BWT, EWC, catastrophic forgetting) are left out. Each "
                   "entry states whether it is peer-reviewed or a preprint; preprints and the blog post should be "
                   "cited with that caveat. Links were checked on 25 October 2026.", SUB)]
    n = 0
    for head, intro, papers in SECTIONS:
        s.append(Paragraph(head, H2))
        s.append(Paragraph(intro, INTRO))
        for title, authors, venue, status, url, why in papers:
            n += 1
            s.append(KeepTogether([
                Paragraph(f'{n}. <link href="{url}" color="{LINK}">{title}</link>', TITLE),
                Paragraph(f"{authors} · <b>{venue}</b> · {status}", META),
                Paragraph(f'<link href="{url}" color="{LINK}">{url}</link>', META),
                Paragraph(f"<i>Why it matters:</i> {why}", WHY),
            ]))
    s.append(Spacer(1, 4))
    s.append(Paragraph(f"{n} papers. Rebuild with: .venv/bin/python tools/build_reading_list_pdf.py", META))
    SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                      bottomMargin=1.6 * cm, title="Reading list: LoRA rank and forgetting in small language models",
                      author="Feasible Rank Window project").build(s)
    print(OUT, n)


if __name__ == "__main__":
    build()
