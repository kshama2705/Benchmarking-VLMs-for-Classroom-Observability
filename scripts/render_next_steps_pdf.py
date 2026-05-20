"""Render BMVC_next_steps.md as a multi-table PDF for sharing."""

import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether,
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "BMVC_next_steps.pdf")

styles = getSampleStyleSheet()
title = ParagraphStyle("title", parent=styles["Heading1"], fontSize=13, leading=16,
                      spaceAfter=2, textColor=colors.black)
sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=8.5, leading=11,
                     textColor=colors.grey, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=10.5, leading=13,
                    spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#1a3a6c"))
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=8.5, leading=11,
                      spaceAfter=3, alignment=0)
cell = ParagraphStyle("cell", parent=body, fontSize=7.5, leading=9.5)
cell_b = ParagraphStyle("cell_b", parent=cell, fontName="Helvetica-Bold")


def P(text):
    return Paragraph(text, body)


def H(text):
    return Paragraph(text, h2)


def make_cells(rows, header):
    return [[Paragraph(c, cell_b) for c in header]] + \
           [[Paragraph(c, cell) for c in row] for row in rows]


def make_table(rows, header, widths):
    data = make_cells(rows, header)
    tbl = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6ecf5")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f7")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return tbl


story = []
story.append(Paragraph("BMVC 2026 — Divide-and-Conquer Plan", title))
story.append(Paragraph("Today: 2026-05-07 · Abstract: 2026-05-22 (15 days) · Paper: 2026-05-29 (22 days)", sub))
story.append(P("Two tracks run in parallel for ~10 days, then we converge. Track A is compute-heavy (existing infrastructure on Aman's machine: DAiSEE frames, MPS, Ollama, OpenAI key). Track B is manuscript-heavy and can start day 1 without waiting on experiments."))

# Track A
story.append(H("Track A — Experiments (suggested owner: Aman)"))
a_rows = [
    ["A1", "Full 1,784-clip CLIP zero-shot, P1/P2/P3 (1 run each)", "results/clip_full_p{N}.csv + bootstrap CIs", "~1 hr", "P0", "reuse 03_clip_inference.py, swap input CSV"],
    ["A2", "Full 1,784-clip LLaVA-1.5 zero-shot, P1/P2/P3", "results/llava_full_p{N}.csv", "~4-6 hr (overnight)", "P0", "Ollama; reuse 06_llava_inference.py"],
    ["A3", "Full 1,784-clip GPT-4o zero-shot, P1/P2/P3 + refusal flag", "results/gpt4o_full_p{N}.csv with refusal column", "~3 hr, ~$27", "P0", "drop P3 from headline if refusal &gt; 50%"],
    ["A4", "Full 1,784-clip Qwen2.5-VL zero-shot, P1/P2/P3 (single-frame t=5s)", "results/qwen_full_p{N}.csv", "~6 hr", "P0", "local; if RAM-limited use Together AI (~$10)"],
    ["A5", "Multi-frame Qwen2.5-VL (t=2,5,8 averaged), P1/P2/P3", "results/qwen_multiframe_full_p{N}.csv", "~6 hr", "P0", "closes 'single-frame' reviewer attack"],
    ["A6", "Class-prior calibration on all 4 models' logits (post-hoc)", "results/calibration_deltas.csv", "~2 hr", "P0", "uses existing logits; no rerun"],
    ["A7", "Self-consistency rerun for LLaVA on N=300 (T=0.7 × 3 seeds)", "results/llava_consistency.csv", "~2 hr", "P1", "audit existing, may already be done"],
    ["A8", "(optional) Face-cropped CLIP probe", "features/clip_face_features.npz + probe results", "~6 hr", "P2", "only if Track B is ahead by day 10"],
]
story.append(make_table(a_rows, ["#", "Experiment", "Output", "Effort", "Pri", "Notes"],
                        [0.25 * inch, 2.0 * inch, 1.7 * inch, 0.85 * inch, 0.3 * inch, 2.0 * inch]))
story.append(P("<b>Critical path A1–A6</b> by <b>day 10 (2026-05-17)</b>."))

# Track B
story.append(H("Track B — Manuscript, figures, bibliography (suggested owner: co-author)"))
b_rows = [
    ["B1", "Clone BMVC 2026 template, port paper skeleton", "paper-bmvc/main.tex", "~3 hr", "P0", "—"],
    ["B2", "Strip all SCB content (sections, fig3, refs)", "smaller paper-bmvc/", "~2 hr", "P0", "B1"],
    ["B3", "Draft new intro around 'Engagement Ceiling' thesis", "sec/intro.tex", "~4 hr", "P0", "brief + probe table"],
    ["B4", "Update related work: add Vedernikov 2025, OrdinalCLIP, DINOv2, ViBED-Net, calibration refs; remove SCB", "sec/related.tex, refs.bib", "~4 hr", "P0", "—"],
    ["B5", "Ethics paragraph: DAiSEE consent, minor faces, GPT-4o refusal as research finding", "sec/ethics.tex", "~2 hr", "P0", "—"],
    ["B6", "Insert probe results into results section (table + interpretation)", "sec/results.tex", "~2 hr", "P0", "available now"],
    ["B7", "Regenerate figures: drop fig3, update fig1/2/4 with CIs, add ceiling figure (κ across encoders × frames × readouts)", "figures/*.pdf", "~5 hr", "P1", "A1-A6"],
    ["B8", "Draft abstract (BMVC format, ~250 words)", "sec/abstract.tex", "~2 hr", "P0", "B3 first"],
]
story.append(make_table(b_rows, ["#", "Task", "Output", "Effort", "Pri", "Depends"],
                        [0.25 * inch, 2.4 * inch, 1.5 * inch, 0.7 * inch, 0.3 * inch, 1.95 * inch]))
story.append(P("<b>Critical path B1–B6, B8</b> by <b>day 13 (2026-05-20)</b>."))

# Track C
story.append(H("Track C — Joint integration (days 11-22)"))
c_rows = [
    ["C1", "Integrate A1-A6 results into Table 1 + Table 2; CI columns everywhere", "both", "by 2026-05-19"],
    ["C2", "Methods section: A drafts experimental setup; B drafts probe methodology", "both", "by 2026-05-19"],
    ["C3", "Discussion + conclusion: ceiling interpretation, future work, limitations", "both", "by 2026-05-20"],
    ["C4", "Internal cross-review of full draft", "both", "2026-05-20 to 21"],
    ["C5", "Abstract submission (HARD)", "Aman", "2026-05-22"],
    ["C6", "Polish pass + supplementary zip (anonymized prompts, configs, code)", "B drafts, A reviews", "by 2026-05-27"],
    ["C7", "Final review pass", "both", "2026-05-28"],
    ["C8", "Paper submission (HARD)", "Aman", "2026-05-29"],
]
story.append(make_table(c_rows, ["#", "Task", "Owner", "When"],
                        [0.25 * inch, 4.4 * inch, 1.0 * inch, 1.5 * inch]))

# Daily cadence
story.append(H("Suggested daily cadence"))
story.append(P("<b>Days 1-3 (08-10 May):</b> A1, A2 launched; B1, B2 done; B4 in progress."))
story.append(P("<b>Days 4-7 (11-14 May):</b> A3, A4 done; A5, A6 in progress; B3, B5, B6 done."))
story.append(P("<b>Days 8-10 (15-17 May):</b> A5, A6 wrapped; B7 figures regenerated; B8 abstract draft."))
story.append(P("<b>Days 11-15 (18-22 May):</b> C1-C5; abstract submission on day 15."))
story.append(P("<b>Days 16-22 (23-29 May):</b> C6-C8; final submission on day 22."))

# Risks
story.append(H("Risk register"))
risk_rows = [
    ["GPT-4o refusal rate on full test set is very high", "Document as research finding; drop from headline metrics; report alongside"],
    ["Qwen2.5-VL local install / RAM limits", "Switch to Together AI API (~$10)"],
    ["Multi-frame Qwen2.5-VL too slow", "Fall back to 8-frame uniform via API or use 3-frame avg"],
    ["Co-author template migration &gt; 3 hr", "Swap roles: Aman handles template, co-author takes A6 calibration"],
    ["Ceiling finding contested by reviewer ('only ViTs tested')", "A8 face-cropped probe pre-empts; cite ViBED-Net for ResNet-based context"],
]
story.append(make_table(risk_rows, ["Risk", "Mitigation"],
                        [3.0 * inch, 4.2 * inch]))

story.append(H("Open coordination questions"))
story.append(P("1. <b>Compute access:</b> does co-author have GPU/RAM to take any of A2/A4? If yes, parallelize."))
story.append(P("2. <b>Writing voice:</b> single-pass merge or section-by-section drafting? Recommend section-by-section with tight ownership; merge in C4."))
story.append(P("3. <b>A7 / A8 inclusion:</b> P1/P2 — only if Track B is ahead by day 10. Don't block on these."))


doc = SimpleDocTemplate(OUT, pagesize=letter,
                        leftMargin=0.45 * inch, rightMargin=0.45 * inch,
                        topMargin=0.4 * inch, bottomMargin=0.4 * inch)
doc.build(story)
print(f"Wrote {OUT}")
