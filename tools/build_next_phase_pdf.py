"""Build docs/next-phase-plan.pdf: proposed tasks, 10-seed cost estimates and recalibration plan."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parents[1] / "docs" / "next-phase-plan.pdf"
INK, MUTED, RULE, HEAD = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#d9d8d4"), colors.HexColor("#f0efec")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=18, leading=22, alignment=TA_LEFT, textColor=INK, spaceAfter=4)
SUB = ParagraphStyle("sub", parent=ss["Normal"], fontSize=9.5, textColor=MUTED, spaceAfter=14)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13, leading=16, textColor=INK, spaceBefore=12, spaceAfter=6)
BODY = ParagraphStyle("body", parent=ss["Normal"], fontSize=9.5, leading=13.5, textColor=INK, spaceAfter=6)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.5, leading=11, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")
NOTE = ParagraphStyle("note", parent=BODY, fontSize=8.5, leading=11.5, textColor=MUTED)


def table(rows, widths, bold_last=False):
    data = [[Paragraph(str(c), CELLB if i == 0 or (bold_last and i == len(rows) - 1) else CELL) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


s = []
s.append(Paragraph("Next phase plan: tasks, 10-seed costs and recalibration", H1))
s.append(Paragraph("Feasible Rank Window project · prepared 25 October 2026 · based on the frw-v4 results "
                   "(1,088 runs, docs/FINDINGS_V4.md)", SUB))

s.append(Paragraph("Why change the tasks", H2))
s.append(Paragraph(
    "On RTE, MNLI and SST-2, LoRA rank 1 already learned every task to within about 1 point of the best rank "
    "(all four models), so accuracy was flat across ranks. These classification tasks have a very low intrinsic "
    "dimension (Aghajanyan et al., ACL 2021). A rank window can only show real edges on tasks that need more "
    "adaptation capacity, so the next phase mixes low- and high-capacity NLP tasks in one sequence.", BODY))

s.append(Paragraph("1. Proposed task sequence", H2))
s.append(table([
    ["#", "Task", "Dataset", "Capacity needed", "Train / eval", "Metric"],
    ["1", "Topic classification", "AG News (4 classes)", "low", "4,000 / 500", "accuracy"],
    ["2", "Natural language inference", "MNLI (link to current results)", "low to medium", "4,000 / 500", "accuracy"],
    ["3", "Extractive question answering", "SQuAD v1.1", "medium", "4,000 / 500", "exact match / F1"],
    ["4", "Dialogue summarisation", "SAMSum", "high", "4,000 / 500", "ROUGE-L"],
    ["5", "Maths word problems (step by step)", "GSM8K", "high", "4,000 / 500", "exact match (final answer)"],
], [0.7, 4.0, 4.2, 2.4, 2.3, 3.4]))
s.append(Spacer(1, 6))
for t in [
    "<b>One format for every task:</b> the model generates the answer as text through its own output layer. "
    "This removes the per-task classifier heads, which caused most of the measured forgetting in v4.",
    "<b>Equal data:</b> the same 4,000 training examples per task; a fixed 500-example evaluation subset per task "
    "keeps the cost of text generation under control.",
    "<b>General-knowledge check:</b> a fixed 500-question MMLU subset (answer-likelihood scoring) after each task, "
    "to measure forgetting of pretrained knowledge as well as of earlier tasks.",
    "<b>Two task orders:</b> easy to hard, and the reverse.",
    "<b>Hypothesis to test:</b> the window stays at r ≈ 4–32 for AG News and MNLI but moves upward for SAMSum and "
    "GSM8K, i.e. the feasible window depends on the type of task.",
]:
    s.append(Paragraph("• " + t, BODY))

s.append(Paragraph("2. Cost estimate with 10 seeds (Modal L40S)", H2))
s.append(Paragraph(
    "Scaled from measured v4 run times: about 7× the training tokens per run (5 tasks × 4,000 examples at ~150 "
    "tokens versus 3 × 2,490 at ~55), about 5 minutes of text-generation evaluation per run, and about 30% extra "
    "for LoRA on all linear layers. L40S at $2.17 per hour including CPU and memory.", BODY))
s.append(table([
    ["Model", "Per run", "10 seeds × 6 ranks (1, 4, 16, 64, 256, 1024)", "LR search (3 ranks × 4 LRs × 2 seeds)", "Total"],
    ["Llama-3.2-1B", "~$0.80", "~$48", "~$15", "~$63"],
    ["Qwen2.5-1.5B", "~$1.20", "~$72", "~$23", "~$95"],
    ["Granite-3.3-2B", "~$2.00", "~$120", "~$38", "~$158"],
    ["Qwen2.5-3B", "~$2.00", "~$120", "~$38", "~$158"],
    ["All four models", "", "", "", "~$475"],
], [3.4, 2.0, 5.2, 4.6, 1.8], bold_last=True))
s.append(Spacer(1, 6))
s.append(table([
    ["Schedule against $180 per month", "Models", "Estimated cost"],
    ["Month 1", "Llama-3.2-1B + Qwen2.5-1.5B", "~$158"],
    ["Month 2", "Granite-3.3-2B", "~$158"],
    ["Month 3", "Qwen2.5-3B", "~$158"],
    ["Optional", "Second task order", "+60–70%"],
    ["One-time extras", "Epoch check (~$15); frozen-model, head-only baseline (under $5); pilot (~$5)", "~$25"],
], [4.0, 9.0, 4.0]))
s.append(Spacer(1, 6))
s.append(Paragraph(
    "Uncertainty: treat every figure as ±40%. Run a ~$5 pilot first to measure the real time per run, exactly as the "
    "GPU benchmark was done before v4. The six accounts keep the same safeguards: admission cap below $30, a hard stop "
    "on the real bill at $29, and an end-of-month deadline.", NOTE))

s.append(Paragraph("3. Will the hyperparameters need recalibrating?", H2))
s.append(Paragraph("Yes, but only a few, and cheaply. The costs above already include the learning-rate search.", BODY))
s.append(table([
    ["Hyperparameter", "Recalibrate?", "Reason"],
    ["Learning rate", "Yes (required)",
     "The strongest driver of forgetting in every experiment. The loss changes (generating text instead of "
     "classifying), the tasks change, and the all-linear selection already differed from q/v. Search at 3 ranks "
     "(4, 64, 1024), 4 LRs, 2 seeds, and reuse it for the other ranks: v4 showed the selected LR barely depends on "
     "rank when α is fixed."],
    ["Epochs per task", "Yes (quick check)",
     "3 epochs was confirmed for classification only (E3). Generative tasks may need a different number: one check "
     "at r = 16 comparing 2 / 3 / 4 epochs with 5 seeds, about $15."],
    ["Max sequence length", "Yes",
     "Raise from 256 to 512 tokens: SQuAD, SAMSum and GSM8K inputs plus answers exceed 256."],
    ["Evaluation decoding", "Set, not tuned",
     "Greedy decoding with a fixed maximum length per task (e.g. 8 tokens for classification, 64 for summaries, "
     "256 for maths)."],
    ["Classifier head learning rate", "No longer needed", "The generative format has no classifier heads."],
    ["α = 32 (scale α/r), AdamW (0.9, 0.999, 1e-8), batch 16, warmup 6% + cosine, clipping 1.0, weight decay 0, "
     "dropout 0", "No",
     "Conventions or settings validated in v4. Re-tuning them adds cost without clear benefit."],
    ["Target modules", "No (decided)", "LoRA on all linear layers, as agreed after the E2 experiment."],
    ["Seeds", "No (decided)", "10 seeds per reported setting; 2 tuning seeds kept separate from reported seeds."],
], [4.3, 3.0, 9.7]))

s.append(Paragraph("Sources", H2))
s.append(Paragraph(
    "Measured costs and timings: results-v4/analysis (runs.csv, compute_by_model_and_stage.json). Findings: "
    "docs/FINDINGS_V4.md, sections 1–6. Intrinsic dimension: Aghajanyan, Zettlemoyer &amp; Gupta (2021), "
    "\"Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning\", ACL. Pricing: Modal "
    "list prices on 25 October 2026 (L40S $1.95/h plus CPU and memory).", NOTE))

OUT.parent.mkdir(parents=True, exist_ok=True)
SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                  bottomMargin=1.6 * cm, title="Next phase plan: tasks, 10-seed costs and recalibration",
                  author="Feasible Rank Window project").build(s)
print(OUT)
