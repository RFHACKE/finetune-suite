"""Build docs/next-phase-plan.pdf: task choice, rank grid, 10-seed cost estimates and recalibration plan."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parents[1] / "docs" / "next-phase-plan.pdf"
INK, MUTED, RULE, HEAD = (colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#d9d8d4"),
                          colors.HexColor("#f0efec"))

ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=18, leading=22, alignment=TA_LEFT, textColor=INK, spaceAfter=4)
SUB = ParagraphStyle("sub", parent=ss["Normal"], fontSize=9.5, textColor=MUTED, spaceAfter=12)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13, leading=16, textColor=INK, spaceBefore=12, spaceAfter=6)
H3 = ParagraphStyle("h3", parent=ss["Heading3"], fontSize=10.5, leading=13, textColor=INK, spaceBefore=6, spaceAfter=3)
BODY = ParagraphStyle("body", parent=ss["Normal"], fontSize=9.5, leading=13.5, textColor=INK, spaceAfter=6)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.5, leading=11, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")
NOTE = ParagraphStyle("note", parent=BODY, fontSize=8.5, leading=11.5, textColor=MUTED)


def table(rows, widths, bold_last=False):
    data = [[Paragraph(str(c), CELLB if i == 0 or (bold_last and i == len(rows) - 1) else CELL) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD), ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def bullets(items):
    return [Paragraph("• " + t, BODY) for t in items]


s = [Paragraph("Next phase plan: tasks, ranks, 10-seed costs and recalibration", H1),
     Paragraph("Feasible Rank Window project · revised 25 October 2026 (v3: AG News and MNLI removed; rank grid "
               "4–256) · based on the frw-v4 results (1,088 runs, docs/FINDINGS_V4.md)", SUB)]

s.append(Paragraph("Why change the tasks", H2))
s.append(Paragraph(
    "On RTE, MNLI and SST-2, LoRA rank 1 already learned every task to within about 1 point of the best rank for "
    "all four models, so accuracy was flat across ranks. Classification tasks of this kind have a very low "
    "intrinsic dimension (Aghajanyan et al., ACL 2021). A rank window can only show real edges on tasks that need "
    "more adaptation capacity.", BODY))

# ---------------------------------------------------------------------------
s.append(Paragraph("1. Task sequence (revised)", H2))
s.append(table([
    ["#", "Task", "Dataset", "Capacity needed", "Train / eval", "Metric"],
    ["1", "Extractive question answering", "SQuAD v1.1", "medium", "4,000 / 500", "exact match / F1"],
    ["2", "Dialogue summarisation", "SAMSum", "high", "4,000 / 500", "ROUGE-L"],
    ["3", "Maths word problems (step by step)", "GSM8K", "high", "4,000 / 500", "exact match (final answer)"],
], [0.7, 4.6, 3.0, 2.6, 2.4, 3.7]))
s.append(Spacer(1, 6))
s += bullets([
    "<b>One format for every task:</b> the model generates the answer as text through its own output layer, so "
    "there are no per-task classifier heads (in v4 these caused most of the measured forgetting).",
    "<b>Equal data:</b> 4,000 training examples per task; a fixed 500-example evaluation subset per task.",
    "<b>General-knowledge check:</b> a fixed 500-question MMLU subset (answer-likelihood scoring) after each task.",
    "<b>Two task orders:</b> SQuAD → SAMSum → GSM8K, and the reverse.",
])

s.append(Paragraph("Decision on AG News and MNLI: removed", H3))
s.append(Paragraph("<b>Why removing them is valid (your position).</b>", BODY))
s += bullets([
    "The v4 data already answer the question for classification: r = 1 learns RTE, MNLI and SST-2 to within about "
    "1 point of the best rank. AG News (4-way topic classification) is easier still. Rank cannot show a window on "
    "tasks that saturate at the smallest rank.",
    "Every run spends about 40% of its compute on these two tasks. Removing them cuts the cost per run and puts the "
    "whole budget on tasks where rank can matter.",
    "Since SQuAD, SAMSum and GSM8K all need real capacity, any rank effect on forgetting is not diluted by tasks "
    "that need almost nothing.",
])
s.append(Paragraph("<b>What is lost (my side), and how to handle it.</b>", BODY))
s += bullets([
    "<b>No within-study control.</b> If the window moves to higher ranks, part of the move could come from the new "
    "setup (text generation instead of heads, all-linear LoRA) rather than from task difficulty. A low-capacity "
    "task in the same sequence would separate the two. Mitigation: argue from v4 (same models, classification, "
    "flat curve) and state the change of setup as a limitation; or add MNLI back as a fourth task only at 3 ranks "
    "(4, 64, 1024) for about +15% cost.",
    "<b>A shorter sequence.</b> Three tasks gives two 'after' measurements for the first task, the same as v4. "
    "To study how forgetting builds up over many fine-tunings, a fourth high-capacity task would be better. "
    "A cheap option is WebNLG (data-to-text generation, short inputs), for about +30% cost.",
    "My recommendation: drop AG News without reservation. MNLI is optional; the plan below assumes it is dropped, "
    "as you decided.",
])

# ---------------------------------------------------------------------------
s.append(Paragraph("2. Rank grid: 4, 8, 16, 32, 64, 128, 256", H2))
s.append(Paragraph("<b>Why this grid is a good choice.</b>", BODY))
s += bullets([
    "Factor-2 spacing locates the edges of the window within a factor of 2, across seven ranks.",
    "It stays in the genuine low-rank regime. With all-linear LoRA, r = 256 trains 16–19% of the model's parameters; "
    "r = 512 and 1024 would have trained 29–77%, which is closer to partial full fine-tuning than to LoRA.",
    "Every rank fits a 48 GB L40S for all four models (adapter training state at r = 256 is 2.9–7.7 GB), so no "
    "80 GB GPUs are needed and all runs use one GPU type, as in v4.",
    "It removes the two slowest and most expensive rank settings, which also showed the most instability in v4 "
    "and E1.",
    "Dropping r = 1 and 2 is reasonable: these generative tasks very likely need more than rank 2.",
])
s.append(Paragraph("<b>Remaining limitations.</b> Trainable parameters of all-linear LoRA:", BODY))
s.append(table([
    ["Model", "r = 4", "r = 16", "r = 64", "r = 256", "Adapter training state at r = 256*"],
    ["Llama-3.2-1B", "2.8 M (0.2%)", "11 M (0.9%)", "45 M (3.6%)", "180 M (15%)", "2.9 GB"],
    ["Qwen2.5-1.5B", "4.6 M (0.3%)", "18 M (1.2%)", "74 M (4.8%)", "295 M (19%)", "4.7 GB"],
    ["Granite-3.3-2B", "7.0 M (0.3%)", "28 M (1.1%)", "113 M (4.5%)", "451 M (18%)", "7.2 GB"],
    ["Qwen2.5-3B", "7.5 M (0.2%)", "30 M (1.0%)", "120 M (3.9%)", "479 M (16%)", "7.7 GB"],
], [3.0, 2.3, 2.3, 2.3, 2.5, 4.6]))
s.append(Paragraph("* fp32 adapter weights + gradients + two Adam moments (16 bytes per parameter); excludes the "
                   "frozen model and activations. Percentages are of the total model size.", NOTE))
s += bullets([
    "<b>The upper edge may be open.</b> If accuracy is still rising at r = 256 for a task, the result can only say "
    "'256 or above'. Decision rule: if a model's best rank is 256, add r = 512 for that model only (about +15% of "
    "its cost) before writing conclusions.",
    "<b>The H-LoRA region is no longer tested.</b> Their 'two-thirds of hidden width' (≈1,024–1,365 here) is out of "
    "range, so the study cannot confirm or reject it directly. v4 already showed that region gives no benefit on "
    "classification.",
    "<b>Some matrices saturate at r = 256 in Qwen.</b> Qwen's k and v projections are only 256 wide, so at r = 256 "
    "they are already at full rank. Record the effective rank per matrix, as v4 did.",
    "<b>The lower edge is open.</b> If r = 4 is already inside the window, the result can only say '4 or below'.",
])

# ---------------------------------------------------------------------------
s.append(Paragraph("3. Cost estimate with 10 seeds", H2))
s.append(Paragraph(
    "Scaled from measured v4 run times. The 3 generative tasks total about 0.85× the training tokens of the earlier "
    "5-task plan (4,000 examples × ~210 tokens each), plus about 4 minutes of text-generation evaluation. All-linear "
    "LoRA adds about 30%, and r = 256 about 12% more. All runs on L40S at $2.17/h including CPU and memory.", BODY))
s.append(table([
    ["Model", "Per run", "Reported: 7 ranks × 10 seeds", "LR search: 3 ranks × 4 LRs × 2 seeds", "Total"],
    ["Llama-3.2-1B", "~$0.70", "~$50", "~$14", "~$64"],
    ["Qwen2.5-1.5B", "~$1.05", "~$75", "~$20", "~$95"],
    ["Granite-3.3-2B", "~$1.75", "~$125", "~$34", "~$159"],
    ["Qwen2.5-3B", "~$1.75", "~$125", "~$34", "~$159"],
    ["All four models", "", "", "", "~$477"],
], [3.3, 2.7, 4.2, 4.6, 2.2], bold_last=True))
s.append(Spacer(1, 6))
s.append(table([
    ["Schedule against $180 per month", "Work", "Estimated cost"],
    ["Month 1", "Pilot (~$5), epoch check (~$15), Llama-3.2-1B", "~$84"],
    ["Month 2", "Qwen2.5-1.5B", "~$95"],
    ["Month 3", "Granite-3.3-2B", "~$159"],
    ["Month 4", "Qwen2.5-3B", "~$159"],
    ["Optional", "r = 512 for a model whose best rank is 256 (+~15% of that model); MNLI control at 3 ranks "
     "(+~15%); second task order (+60–70%); WebNLG 4th task (+~30%)", "—"],
], [4.4, 9.0, 3.6]))
s.append(Spacer(1, 6))
s.append(Paragraph(
    "Uncertainty: treat every figure as ±40%. Generation length is the least certain part; a ~$5 pilot on one "
    "model at r = 16 and r = 256 should come first. Months 1 and 2 together (~$179) could fit one month's $180, but "
    "with no safety margin, so the plan keeps them separate.", NOTE))

# ---------------------------------------------------------------------------
s.append(Paragraph("4. Hyperparameters that need recalibrating", H2))
s.append(table([
    ["Hyperparameter", "Recalibrate?", "Reason"],
    ["Learning rate", "Yes (required)",
     "It was the strongest driver of forgetting in every experiment. The loss (text generation), tasks and target "
     "modules all change. Search 4 LRs × 2 seeds at r = 4, 32 and 256 (both ends of the grid and the middle): E1 "
     "showed high ranks can need about half the LR. Use the lowest learning rate within one standard error of the "
     "best, and interpolate for the ranks in between."],
    ["Epochs per task", "Yes (quick check)",
     "3 epochs was confirmed for classification only. One check at r = 16 comparing 2 / 3 / 4 epochs with 5 seeds, "
     "about $15."],
    ["Max sequence length", "Yes", "Raise from 256 to 512 tokens; SQuAD, SAMSum and GSM8K exceed 256."],
    ["Evaluation decoding", "Set, not tuned",
     "Greedy decoding; maximum new tokens per task: 32 for SQuAD, 96 for SAMSum, 256 for GSM8K."],
    ["Classifier head learning rate", "No longer needed", "The generative format has no classifier heads."],
    ["α = 32 (scale α/r), AdamW (0.9, 0.999, 1e-8), batch 16, warmup 6% + cosine, clipping 1.0, weight decay 0, "
     "dropout 0", "No", "Conventions or settings validated in v4; re-tuning them adds cost without clear benefit."],
    ["Target modules / seeds", "No (decided)",
     "All linear layers; 10 reported seeds, with 2 separate tuning seeds."],
], [4.4, 3.0, 9.6]))

s.append(Paragraph("Sources", H2))
s.append(Paragraph(
    "Measured costs and timings: results-v4/analysis (runs.csv, compute_by_model_and_stage.json). Findings: "
    "docs/FINDINGS_V4.md, sections 1–6 (E1: high ranks need a lower LR; E2: all-linear LoRA; E3: 3 epochs). Parameter "
    "counts are computed from each model's layer sizes. Intrinsic dimension: Aghajanyan, Zettlemoyer &amp; Gupta "
    "(2021), ACL. Pricing: Modal list prices on 25 October 2026.", NOTE))

OUT.parent.mkdir(parents=True, exist_ok=True)
SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                  bottomMargin=1.6 * cm, title="Next phase plan: tasks, ranks, 10-seed costs and recalibration",
                  author="Feasible Rank Window project").build(s)
print(OUT)
