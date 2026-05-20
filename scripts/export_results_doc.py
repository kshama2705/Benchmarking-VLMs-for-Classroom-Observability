"""Export all DAiSEE experiment results to a Word document with explanations."""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(BASE_DIR, "DAiSEE_Experiment_Results.docx")


def set_cell(cell, text, bold=False, align="left", bg_color=None):
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(str(text))
    run.font.size = Pt(9)
    if bold:
        run.bold = True
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "right":
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    if bg_color:
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), bg_color)
        cell._tc.get_or_add_tcPr().append(shading)


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, h in enumerate(headers):
        set_cell(table.rows[0].cells[i], h, bold=True, align="center", bg_color="4472C4")
        table.rows[0].cells[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)

    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            align = "left" if c_idx == 0 else "center"
            set_cell(table.rows[r_idx + 1].cells[c_idx], val, align=align)

    return table


def main():
    doc = Document()

    # Title
    title = doc.add_heading("DAiSEE Experiment Results", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph(
        "Benchmarking Vision-Language Models as Classroom Observers\n"
        "CV4Edu Workshop @ CVPR 2026"
    ).alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph("")

    # ── Section 1: Experiment Overview ──
    doc.add_heading("1. Experiment Overview", level=1)
    doc.add_paragraph(
        "We evaluated 4 Vision-Language Models (VLMs) on the DAiSEE dataset for student "
        "engagement classification. DAiSEE contains 9,068 video clips of students watching "
        "educational videos, each labeled with an engagement level from 0 (not engaged) to "
        "3 (highly engaged). We sampled 300 clips using stratified sampling to ensure "
        "representation of all 4 engagement levels, extracted the middle frame (at 5 seconds) "
        "from each 10-second clip, and ran zero-shot classification with each VLM."
    )

    doc.add_heading("Models Tested", level=2)
    bullets = [
        "CLIP (ViT-B-32): Discriminative model. Computes cosine similarity between image embedding and text descriptions of each engagement level. Deterministic output.",
        "BLIP-VQA (blip-vqa-base): Discriminative VQA model. Given an image and a question about engagement, produces a short text answer (e.g., 'not very', 'yes') which is mapped to a numeric level. Deterministic output.",
        "GPT-4o (OpenAI API): Generative model. Receives the image and a text prompt, generates a free-text response. We parse the response to extract the engagement level. Non-deterministic with temperature > 0.",
        "LLaVA-1.5-7B (via Ollama, 4-bit quantized): Open-source generative model. Same setup as GPT-4o but runs locally. Non-deterministic with temperature > 0.",
    ]
    for b in bullets:
        doc.add_paragraph(b, style="List Bullet")

    doc.add_heading("Prompt Variants", level=2)
    doc.add_paragraph(
        "Each model was tested with 3 prompt variants to measure prompt sensitivity:"
    )
    prompts = [
        "P1 (Minimal): A simple instruction asking the model to rate engagement on a 0-3 scale with just the number.",
        "P2 (Rubric): A detailed rubric describing behavioral cues for each engagement level (e.g., 'looking away' = 0, 'leaning in' = 3).",
        "P3 (Chain-of-Thought): A step-by-step prompt asking the model to first describe the student's expression, then posture, then give a rating. This tests whether structured reasoning improves classification.",
    ]
    for p in prompts:
        doc.add_paragraph(p, style="List Bullet")

    doc.add_heading("Exact Prompts Used", level=2)
    doc.add_paragraph(
        "Below are the exact prompts sent to each model. For CLIP, prompts are text "
        "templates for each engagement level (used for cosine similarity). For BLIP-VQA, "
        "prompts are questions. For GPT-4o and LLaVA, prompts are identical instructions "
        "sent alongside the image."
    )

    # CLIP prompts
    doc.add_heading("CLIP Prompts (text templates per class)", level=3)
    clip_prompts = {
        "P1 (minimal)": [
            "a student who is not engaged at all",
            "a student who is barely engaged",
            "a student who is engaged",
            "a student who is highly engaged",
        ],
        "P2 (behavioral)": [
            "a student looking away from screen, distracted, not paying attention",
            "a student sitting passively, neutral expression, not focused",
            "a student paying attention, looking at screen, following along",
            "a student actively engaged, leaning forward, taking notes, alert",
        ],
        "P3 (emotional)": [
            "a bored, disinterested student, sleepy, yawning",
            "an indifferent student with blank expression",
            "an attentive, interested student with focused expression",
            "an excited, curious, enthusiastic student, smiling while learning",
        ],
    }
    for pname, templates in clip_prompts.items():
        p = doc.add_paragraph()
        run = p.add_run(f"{pname}:")
        run.bold = True
        for i, t in enumerate(templates):
            doc.add_paragraph(f"Level {i}: \"{t}\"", style="List Bullet 2")

    # BLIP-VQA prompts
    doc.add_heading("BLIP-VQA Prompts (questions)", level=3)
    blip_prompts = {
        "P1 (direct)": "How engaged is this student?",
        "P2 (attention)": "Is this student paying attention?",
        "P3 (behavioral)": "Is this student interested in the lesson?",
    }
    for pname, question in blip_prompts.items():
        p = doc.add_paragraph()
        run = p.add_run(f"{pname}: ")
        run.bold = True
        p.add_run(f"\"{question}\"")

    doc.add_paragraph(
        "\nBLIP-VQA answer mapping: Short answers are mapped to engagement levels. "
        "For example, 'not at all' → 0, 'not very' / 'not much' → 1, "
        "'yes' / 'moderately' → 2, 'very' / 'extremely' → 3."
    )

    # GPT-4o / LLaVA prompts (shared)
    doc.add_heading("GPT-4o & LLaVA Prompts (identical for both)", level=3)

    gpt_prompts = {
        "P1 (minimal)": (
            "Rate this student's engagement level:\n"
            "0 = not engaged\n"
            "1 = barely engaged\n"
            "2 = engaged\n"
            "3 = highly engaged\n\n"
            "Answer with just the number (0, 1, 2, or 3)."
        ),
        "P2 (rubric)": (
            "You are an educational observer. Assess this student's engagement "
            "based on their facial expression and body language:\n\n"
            "0 = Distracted, looking away, disinterested, sleepy\n"
            "1 = Passively present but not focused, neutral expression\n"
            "2 = Attentive and following along, maintaining eye contact with screen\n"
            "3 = Actively focused, leaning in, alert, showing curiosity\n\n"
            "Respond with only the number (0, 1, 2, or 3)."
        ),
        "P3 (chain-of-thought)": (
            "Analyze this student watching an educational video.\n\n"
            "Step 1: Describe the student's facial expression in one sentence.\n"
            "Step 2: Describe their body posture in one sentence.\n"
            "Step 3: Based on your observations, rate their engagement level:\n"
            "  0 = not engaged at all\n"
            "  1 = barely engaged\n"
            "  2 = engaged\n"
            "  3 = highly engaged\n\n"
            "Format your response as:\n"
            "Expression: ...\n"
            "Posture: ...\n"
            "Rating: [0-3]"
        ),
    }
    for pname, prompt_text in gpt_prompts.items():
        p = doc.add_paragraph()
        run = p.add_run(f"{pname}:")
        run.bold = True
        # Add prompt in a styled block
        prompt_para = doc.add_paragraph()
        prompt_run = prompt_para.add_run(prompt_text)
        prompt_run.font.size = Pt(9)
        prompt_run.font.name = "Courier New"
        prompt_para.paragraph_format.left_indent = Inches(0.5)

    doc.add_paragraph(
        "\nNote: GPT-4o P3 triggered safety refusals in 98% of cases because the prompt "
        "asks to 'describe the student's facial expression,' which GPT-4o interprets as "
        "facial analysis of a real person. LLaVA had no such issue with the same prompt."
    )

    doc.add_paragraph("")

    doc.add_heading("Sample Distribution", level=2)
    doc.add_paragraph(
        "Our stratified sample of 300 clips has the following ground truth distribution:"
    )
    add_table(doc,
        ["Engagement Level", "Label", "Count", "Percentage"],
        [
            ["0", "Not engaged", "4", "1.3%"],
            ["1", "Barely engaged", "84", "28.0%"],
            ["2", "Engaged", "106", "35.3%"],
            ["3", "Highly engaged", "106", "35.3%"],
        ]
    )
    doc.add_paragraph(
        "\nNote: Level 0 has only 4 samples because it is extremely rare in DAiSEE's test set "
        "(only 4 out of 1,784 clips). We included all of them. Levels 1-3 were randomly sampled "
        "to reach 300 total."
    )

    doc.add_paragraph("")

    # ── Section 2: Table 1 — Best Performance ──
    doc.add_heading("2. Table 1: Best Classification Performance per Model", level=1)
    doc.add_paragraph(
        "This table shows the best-performing prompt variant for each model, compared against "
        "supervised baselines from the literature. The 'best' prompt is selected by highest accuracy. "
        "For GPT-4o, the best run (run 2, temp=0.7) is shown since it outperformed the temp=0 run."
    )

    add_table(doc,
        ["Model", "Best Prompt", "MSE ↓", "Acc ↑", "Kappa (QW) ↑", "F1 Macro ↑"],
        [
            ["CLIP (ViT-B-32)", "P1 minimal", "1.000", "37.0%", "0.036", "0.224"],
            ["BLIP-VQA", "P3 behavioral", "0.687", "35.3%", "0.000", "0.131"],
            ["GPT-4o", "P2 rubric (run 2)", "1.507", "32.3%", "0.075", "0.192"],
            ["LLaVA-1.5-7B", "P3 CoT (run 3)", "0.720", "39.0%", "0.101", "0.210"],
            ["CavT (SOTA baseline)", "supervised", "0.038", "—", "—", "—"],
            ["ResNet+TCN (baseline)", "supervised", "0.042", "58.8%", "—", "—"],
        ]
    )

    doc.add_paragraph("")
    doc.add_heading("How to read this table:", level=3)
    bullets = [
        "MSE (Mean Squared Error): Lower is better. Measures average squared difference between predicted and true engagement levels. CavT achieves 0.038; our best VLM (BLIP-VQA) gets 0.687 — about 18× worse.",
        "Accuracy: Percentage of samples where the predicted level exactly matches ground truth. Best VLM is LLaVA at 39.0%, vs. supervised ResNet+TCN at 58.8%.",
        "Cohen's Kappa (Quadratic Weighted): Measures agreement beyond chance. Values near 0 mean the model is no better than random guessing. All VLMs have kappa < 0.11, indicating near-random performance.",
        "F1 Macro: Average F1 score across all 4 classes. Low values (< 0.25) indicate the model fails to correctly identify most classes.",
    ]
    for b in bullets:
        doc.add_paragraph(b, style="List Bullet")

    doc.add_paragraph(
        "\nKey takeaway: All VLMs perform dramatically worse than supervised baselines. "
        "The best VLM (LLaVA with chain-of-thought prompting) achieves only 39.0% accuracy "
        "and a kappa of 0.101, which is still near-random agreement. Supervised methods are "
        "15-40× better on MSE."
    )

    doc.add_paragraph("")

    # ── Section 3: Table 2 — All Prompt Variants ──
    doc.add_heading("3. Table 2: Results Across All 12 Prompt Variants", level=1)
    doc.add_paragraph(
        "This table shows results for every model × prompt combination. It reveals how "
        "sensitive each model is to prompt design. The 'Pred Distribution' column shows "
        "the predicted class breakdown — this is critical for understanding class collapse."
    )

    add_table(doc,
        ["Model", "Prompt", "MSE", "Acc", "Kappa", "Pred Distribution"],
        [
            ["CLIP", "P1 minimal", "1.000", "37.0%", "0.036", "{0:4, 1:75, 2:214, 3:7}"],
            ["CLIP", "P2 behavioral", "1.387", "31.3%", "0.068", "{0:1, 1:216, 2:83}"],
            ["CLIP", "P3 emotional", "1.210", "35.0%", "-0.042", "{1:117, 2:167, 3:16}"],
            ["BLIP-VQA", "P1 direct", "2.390", "25.3%", "-0.018", "{0:55, 1:245}"],
            ["BLIP-VQA", "P2 attention", "0.940", "32.7%", "0.058", "{0:16, 1:24, 2:260}"],
            ["BLIP-VQA", "P3 behavioral", "0.687", "35.3%", "0.000", "{2:300}"],
            ["GPT-4o", "P1 minimal", "1.667", "27.7%", "0.067", "{0:12, 1:256, 2:32}"],
            ["GPT-4o", "P2 rubric", "1.580", "28.7%", "0.083", "{0:10, 1:246, 2:44}"],
            ["GPT-4o", "P3 CoT*", "0.690", "36.0%", "0.039", "{0:1, 1:4, 2:294, 3:1}"],
            ["LLaVA", "P1 minimal", "4.073", "5.3%", "0.037", "{0:254, 2:46}"],
            ["LLaVA", "P2 rubric", "2.837", "15.7%", "0.046", "{0:116, 1:182, 2:1, 3:1}"],
            ["LLaVA", "P3 CoT", "0.763", "37.7%", "0.009", "{1:22, 2:277, 3:1}"],
        ]
    )

    doc.add_paragraph(
        "\n*GPT-4o P3 (Chain-of-Thought): 98% of responses were safety refusals "
        "('I'm sorry, I can't help with that.') because the prompt asked to 'describe "
        "the student's facial expression,' which triggered GPT-4o's safety guardrails. "
        "These refusals defaulted to level 2 in our parsing, artificially inflating "
        "accuracy for the majority class."
    )

    doc.add_paragraph("")
    doc.add_heading("How to read this table:", level=3)
    doc.add_paragraph(
        "The Prediction Distribution column is the most revealing. Compare it against the "
        "ground truth distribution {0:4, 1:84, 2:106, 3:106}:"
    )
    bullets = [
        "BLIP-VQA P3 predicts ALL 300 samples as level 2 — complete class collapse. Its 35.3% accuracy simply reflects that 35.3% of the dataset happens to be level 2.",
        "LLaVA P1 predicts 254/300 as level 0 ('not engaged') — the opposite extreme. The same model with P3 predicts 277/300 as level 2. Same images, completely different predictions.",
        "GPT-4o consistently predicts mostly level 1 (barely engaged) across P1 and P2, suggesting a systematic bias toward low engagement ratings.",
        "No model ever predicts level 3 (highly engaged) in meaningful quantities, despite 35% of ground truth being level 3.",
    ]
    for b in bullets:
        doc.add_paragraph(b, style="List Bullet")

    doc.add_paragraph("")

    # ── Section 4: Table 3 — Prompt Sensitivity ──
    doc.add_heading("4. Table 3: Prompt Sensitivity", level=1)
    doc.add_paragraph(
        "This table quantifies how much each model's performance changes across the 3 "
        "prompt variants. It reports the difference between the best and worst prompt "
        "(max − min) for each metric. Higher values indicate greater sensitivity — meaning "
        "the model's output depends heavily on how you phrase the question, not on the actual image content."
    )

    add_table(doc,
        ["Model", "ΔMSE", "ΔAccuracy", "ΔKappa"],
        [
            ["CLIP", "0.387", "5.7%", "0.110"],
            ["BLIP-VQA", "1.703", "10.0%", "0.076"],
            ["GPT-4o", "0.977", "8.3%", "0.044"],
            ["LLaVA", "3.310", "32.3%", "0.037"],
        ]
    )

    doc.add_paragraph("")
    doc.add_heading("How to read this table:", level=3)
    bullets = [
        "LLaVA has the highest prompt sensitivity: accuracy swings by 32.3 percentage points (from 5.3% to 37.7%) and MSE swings by 3.31 just by changing the prompt. This means the prompt matters far more than the image content.",
        "CLIP is the most stable model with only 5.7% accuracy variation — but it's still performing poorly across all prompts.",
        "BLIP-VQA has high MSE sensitivity (1.703) because its predictions shift from 'mostly level 0-1' (P1) to 'all level 2' (P3).",
        "This has major implications: if a VLM's output changes dramatically based on prompt wording, it cannot be trusted as a reliable classroom observer.",
    ]
    for b in bullets:
        doc.add_paragraph(b, style="List Bullet")

    doc.add_paragraph("")

    # ── Section 5: Table 4 — Self-Consistency ──
    doc.add_heading("5. Table 4: Self-Consistency", level=1)
    doc.add_paragraph(
        "Self-consistency measures whether a model gives the same answer when asked the "
        "same question multiple times. We ran the best prompt 3 times for each model. "
        "CLIP and BLIP-VQA are deterministic (temperature=0, no randomness), so they "
        "always give the same answer. GPT-4o and LLaVA were tested with temperature=0.7 "
        "to measure variability."
    )

    add_table(doc,
        ["Model", "Prompt", "Full Agree", "Pairwise", "Acc Range", "Kappa Range"],
        [
            ["CLIP", "P1", "100%", "100%", "37.0% (det.)", "0.036"],
            ["BLIP-VQA", "P2", "100%", "100%", "32.7% (det.)", "0.058"],
            ["GPT-4o", "P2", "68.0%", "78.7%", "28.7–32.3%", "0.075–0.099"],
            ["LLaVA", "P3", "73.3%", "85.7%", "37.7–39.0%", "0.009–0.101"],
        ]
    )

    doc.add_paragraph("")
    doc.add_heading("How to read this table:", level=3)
    bullets = [
        "Full Agreement: The percentage of samples where all 3 runs gave the exact same prediction. GPT-4o only agrees with itself 68% of the time — for 32% of images, it gives different answers on different runs.",
        "Pairwise Agreement: Average agreement between any 2 of the 3 runs. More lenient than full agreement.",
        "GPT-4o is less self-consistent than LLaVA (68% vs 73.3%), and also produced some safety refusals inconsistently in run 3 that didn't appear in run 1.",
        "LLaVA's kappa range (0.009–0.101) is notable: the model can appear to have near-zero agreement with ground truth (run 1) or slight agreement (run 3), depending on random sampling during generation.",
    ]
    for b in bullets:
        doc.add_paragraph(b, style="List Bullet")

    doc.add_paragraph("")

    # ── Section 6: Key Findings ──
    doc.add_heading("6. Key Findings for the Paper", level=1)

    findings = [
        (
            "All VLMs fail at engagement classification",
            "The best kappa achieved is 0.101 (LLaVA), which represents near-random agreement. "
            "For reference, kappa > 0.6 is typically considered 'substantial agreement' and "
            "kappa > 0.8 is 'almost perfect.' Our best model doesn't even reach 'slight agreement' "
            "(0.0–0.20). Supervised baselines achieve 15-40× better MSE."
        ),
        (
            "Universal class collapse",
            "Every model collapses its predictions to 1-2 dominant classes, regardless of prompt. "
            "Models never learn to distinguish all 4 engagement levels — they essentially pick a "
            "default and assign it to most images. The 'best' accuracy scores (~37%) are artifacts "
            "of predicting the majority class correctly."
        ),
        (
            "Extreme prompt sensitivity",
            "LLaVA's predictions swing from 85% 'not engaged' (P1) to 92% 'engaged' (P3) on "
            "the exact same 300 images. This demonstrates that VLMs are responding to prompt "
            "framing rather than visual content. This is a fundamental reliability concern."
        ),
        (
            "GPT-4o safety guardrails block educational use",
            "GPT-4o's chain-of-thought prompt triggered safety refusals in 98% of cases because "
            "it asked to 'describe the student's facial expression.' Even the rubric prompt with "
            "temperature=0.7 occasionally triggered refusals. This is an important finding for "
            "anyone considering deploying GPT-4o in educational observation contexts."
        ),
        (
            "Open-source slightly outperforms proprietary",
            "LLaVA-1.5-7B (open, 4-bit quantized, running locally) achieved the best accuracy "
            "(39.0%) and kappa (0.101), slightly outperforming GPT-4o (32.3%, 0.099). This "
            "suggests that model scale and proprietary training don't provide an advantage for "
            "this task — the fundamental limitation is the zero-shot approach itself."
        ),
        (
            "Low self-consistency in generative models",
            "GPT-4o and LLaVA change their predictions for 20-30% of images across runs with "
            "temperature=0.7. A reliable classroom observation tool should give consistent "
            "assessments."
        ),
    ]

    for title_text, body_text in findings:
        p = doc.add_paragraph()
        run = p.add_run(f"{title_text}: ")
        run.bold = True
        run.font.size = Pt(11)
        p.add_run(body_text).font.size = Pt(11)

    doc.add_paragraph("")

    # ── Section 7: Methodology Notes ──
    doc.add_heading("7. Methodology Notes for Co-Author", level=1)

    doc.add_heading("Dataset", level=2)
    doc.add_paragraph(
        "DAiSEE (Dataset for Affective States in E-Environments) contains 9,068 video "
        "clips from 112 subjects. Each 10-second clip is labeled for Boredom, Engagement, "
        "Confusion, and Frustration on a 0-3 scale. We focus only on Engagement. The test "
        "set has 1,784 clips across 21 subjects. The dataset is heavily imbalanced: "
        "level 0 has only 4 clips, level 1 has 84, while levels 2 and 3 have ~850 each."
    )

    doc.add_heading("Frame Extraction", level=2)
    doc.add_paragraph(
        "We extracted the middle frame (at t=5s) from each 10-second video using ffmpeg. "
        "This gives us a single representative image per clip. We used stratified sampling: "
        "all 4 level-0 clips, all 84 level-1 clips, and 106 each from levels 2 and 3 "
        "(randomly selected), totaling 300 frames."
    )

    doc.add_heading("LLaVA Hardware Note", level=2)
    doc.add_paragraph(
        "LLaVA-1.5-7B was run via Ollama (4-bit quantized) on a 16GB M2 MacBook. "
        "The original HuggingFace float16 version (~14GB) could not fit in memory for "
        "inference. We used 4-bit quantization through Ollama, which reduces the model "
        "to ~4GB and runs at ~9 seconds per image. This should be noted in the paper as "
        "it may slightly affect results compared to full-precision inference."
    )

    doc.add_heading("GPT-4o Configuration", level=2)
    doc.add_paragraph(
        "GPT-4o was accessed via the OpenAI API with images sent as base64-encoded JPEG "
        "at 'low' detail (512×512 tokens). Rate limiting of 0.5s between requests was "
        "applied. Temperature was 0.0 for run 1 and 0.7 for runs 2-3."
    )

    doc.add_heading("Response Parsing", level=2)
    doc.add_paragraph(
        "For generative models (GPT-4o, LLaVA), we parse responses as follows: "
        "(1) Look for 'Rating: X' pattern for CoT prompts, "
        "(2) Find the last standalone digit 0-3 in the response, "
        "(3) Default to level 2 if no valid digit found (this includes safety refusals). "
        "For BLIP-VQA, short answers like 'not very' are mapped to levels via a lookup table."
    )

    doc.add_paragraph("")

    # ── Section 8: File Inventory ──
    doc.add_heading("8. Result Files", level=1)
    doc.add_paragraph("All result CSVs are in the results/ directory:")

    files = [
        ("clip_prompt{1,2,3}_run1.csv", "CLIP predictions for each prompt variant"),
        ("blip_vqa_prompt{1,2,3}_run1.csv", "BLIP-VQA predictions"),
        ("gpt4o_prompt{1,2,3}_run1.csv", "GPT-4o predictions (temp=0)"),
        ("gpt4o_prompt2_run{2,3}.csv", "GPT-4o consistency runs (temp=0.7)"),
        ("llava_prompt{1,2,3}_run1.csv", "LLaVA predictions (temp=0)"),
        ("llava_prompt3_run{2,3}.csv", "LLaVA consistency runs (temp=0.7)"),
        ("*_raw.csv", "Raw model responses (for debugging/qualitative analysis)"),
    ]
    for fname, desc in files:
        doc.add_paragraph(f"{fname} — {desc}", style="List Bullet")

    doc.add_paragraph(
        "\nEach CSV has columns: clip_id, predicted, ground_truth. "
        "Raw CSVs additionally have: raw_answer (the full model response text)."
    )

    # Save
    doc.save(OUTPUT_PATH)
    print(f"Document saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
