"""Render BMVC_pivot_brief.md as a single-page PDF for sharing with co-author."""

import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether,
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "BMVC_pivot_brief.pdf")

styles = getSampleStyleSheet()
title = ParagraphStyle("title", parent=styles["Heading1"], fontSize=13, leading=16,
                      spaceAfter=2, textColor=colors.black)
sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=8.5, leading=11,
                     textColor=colors.grey, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=10, leading=12,
                    spaceBefore=6, spaceAfter=2, textColor=colors.HexColor("#1a3a6c"))
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=8.5, leading=11,
                      spaceAfter=3, alignment=0)
quote = ParagraphStyle("quote", parent=body, fontSize=8.5, leading=11,
                       leftIndent=12, rightIndent=12, textColor=colors.HexColor("#333"),
                       fontName="Helvetica-Oblique", spaceBefore=2, spaceAfter=4)


def P(text):
    return Paragraph(text, body)


def H(text):
    return Paragraph(text, h2)


story = []
story.append(Paragraph("BMVC 2026 Pivot Brief — Engagement VLM Paper", title))
story.append(Paragraph("Date: 2026-05-06 · Deadlines: Abstract 22 May, Paper 29 May (no extensions)", sub))

story.append(H("Where we are"))
story.append(P("<b>CV4Edu (CVPR Workshop):</b> Accepted to non-archival track. Doesn't bar later peer-reviewed submission."))
story.append(P("<b>CV4Edu archival track:</b> Rejected. Reviewer cLFD (confidence 5) flagged: single-frame protocol invalidates temporal task; N=300 sampling biased; SCB scene-level reformulation not comparable to prior work; GPT-4o L2 fallback corrupts metrics; \"pure benchmarking without methodological innovation is weak.\""))
story.append(P("<b>Plan:</b> Resubmit a substantially strengthened version to BMVC 2026."))

story.append(H("Original BMVC plan (Framing A — \"Diagnose-and-Adapt\")"))
story.append(P("Reviewer signal said benchmark-only is dead. The plan kept the diagnostic half and added an adaptation half: linear probing, class-prior calibration, multi-frame inference, ordinal CLIP. Goal: <i>\"the signal is in the encoder, the language readout is broken — here's how to fix it.\"</i>"))
story.append(P("The paper hinged on one experiment: <b>subject-disjoint linear probe on CLIP features.</b>"))
story.append(P("• κ_quad ≥ 0.25 → encoder has signal → Framing A works"))
story.append(P("• κ_quad &lt; 0.15 → encoder lacks signal → Framing A fails, pivot needed"))
story.append(P("Decision: drop SCB; run on DAiSEE only (cleaner ordinal task, official subject-disjoint splits, comparable to prior literature)."))

story.append(H("What we ran"))
story.append(P("Linear probes on the <b>full 1,784-clip DAiSEE test set</b>, trained on the full official train split (5,358 clips, subject-disjoint). Three independent variations to rule out single-encoder artifacts:"))

table_data = [
    ["Probe", "κ_quad (95% CI)"],
    ["CLIP ViT-B/32, single frame, LogReg (balanced)", "0.055 [0.014, 0.091]"],
    ["CLIP ViT-B/32, single frame, Ridge (ordinal)", "0.098 [0.062, 0.134]"],
    ["CLIP ViT-B/32, mean of 3 frames, LogReg", "0.101 [0.068, 0.133]"],
    ["CLIP ViT-B/32, mean of 3 frames, Ridge", "0.103 [0.061, 0.140]"],
    ["DINOv2 ViT-B/14, single frame, LogReg", "0.107 [0.072, 0.143]"],
    ["DINOv2 ViT-B/14, single frame, Ridge", "0.090 [0.042, 0.133]"],
    ["Best zero-shot (LLaVA P3) — for reference", "~0.10"],
]
tbl = Table(table_data, colWidths=[3.6 * inch, 2.0 * inch], hAlign="LEFT")
tbl.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6ecf5")),
    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
    ("LINEBELOW", (0, -2), (-1, -2), 0.3, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f7")]),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ("TOPPADDING", (0, 0), (-1, -1), 2),
    ("LEFTPADDING", (0, 0), (-1, -1), 4),
]))
story.append(tbl)

story.append(H("What this tells us"))
story.append(P("<b>A κ ≈ 0.10 ceiling holds across three orthogonal axes:</b>"))
story.append(P("(1) Encoder family — CLIP (vision-language) ≈ DINOv2 (vision-only). "
               "(2) Temporal context — 1 frame ≈ 3 frames. "
               "(3) Readout — categorical ≈ ordinal regression."))
story.append(P("The best probe is statistically tied with the best zero-shot prompt. None reach 0.15. "
               "Framing A's premise (signal in encoder, broken readout) is <b>decisively rejected</b>."))

story.append(H("Proposed new direction (Pivot 1 — \"The Engagement Ceiling\")"))
story.append(P("The negative result is stronger than the original benchmark because it pre-empts every reviewer attack:"))
story.append(Paragraph(
    "<i>\"Frozen vision features — whether CLIP, DINOv2, or zero-shot VLMs; whether single-frame or temporal; "
    "whether read out via prompts or supervised heads — encode an upper bound of κ ≈ 0.10 on individual "
    "classroom engagement. The bottleneck is representational, not in the prompt or language readout.\"</i>",
    quote))
story.append(P("<b>Survives:</b> \"you didn't try adaptation\" (we did), \"single-frame is the problem\" "
               "(multi-frame also fails), \"CLIP-specific quirk\" (DINOv2 also fails)."))
story.append(P("<b>Implies:</b> the field needs face/AU-aware encoders or end-to-end fine-tuning, not better "
               "prompts. Direct contribution to BMVC's explainable-AI track or Brave New Ideas track."))

story.append(H("Remaining work (~21 days)"))
story.append(P("(1) Full 1,784-clip zero-shot reruns with bootstrap CIs (CLIP, LLaVA, GPT-4o, Qwen2.5-VL). "
               "(2) Multi-frame Qwen2.5-VL inference (closes temporal-handicap critique reviewer-side). "
               "(3) Class-prior calibration on existing logits (sanity check). "
               "(4) Drop SCB; switch to BMVC template; rewrite intro around the ceiling thesis. "
               "(5) Ethics paragraph (DAiSEE consent / IRB)."))

story.append(H("Questions for you"))
story.append(P("(1) On board with Pivot 1, or push on additional probes (face-cropped CLIP, deeper probe) "
               "before locking framing?"))
story.append(P("(2) Title preference: <i>\"The Engagement Ceiling: Why Frozen Vision Features Cannot "
               "Recover Individual Classroom Engagement\"</i> — or alternatives?"))
story.append(P("(3) Anything to keep from the SCB scene-level work as a side analysis, or drop entirely?"))


doc = SimpleDocTemplate(OUT, pagesize=letter,
                        leftMargin=0.55 * inch, rightMargin=0.55 * inch,
                        topMargin=0.45 * inch, bottomMargin=0.45 * inch)
doc.build(story)
print(f"Wrote {OUT}")
