"""Day 2 report: PPEP across encoders + qualitative cross-encoder findings.
Honest negative-result summary, with the qualitative observations that ARE
worth keeping in the paper."""

import os, json
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "BMVC_day2_report.docx")

doc = Document()
style = doc.styles["Normal"]
style.font.name = "Calibri"; style.font.size = Pt(11)


def H1(t): return doc.add_heading(t, level=1)
def H2(t): return doc.add_heading(t, level=2)
def H3(t): return doc.add_heading(t, level=3)


def P(t=""): return doc.add_paragraph(t)


def P_runs(*runs):
    p = doc.add_paragraph()
    for text, kw in runs:
        r = p.add_run(text)
        r.bold = kw.get("bold", False); r.italic = kw.get("italic", False)
    return p


def load_json(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path))


# === Header ===
doc.add_heading("BMVC paper — Day 2 progress report", level=0)
sub = doc.add_paragraph()
sub.add_run("Date: 2026-05-11 · 48h TIPS-inspired exploration").italic = True

H2("Summary")
P_runs(
    ("Day 2 outcome: ", {"bold": True}),
    ("PPEP (Patch-Pooled Engagement Probe) is a ", {}),
    ("negative result on absolute κ", {"italic": True}),
    (" — face-region pooling never beats the encoder's own CLS on standard contrastive backbones, and TIPS's spatially-aware patches don't break the κ ≈ 0.10–0.16 ceiling. SigLIP-L CLS at κ = 0.199 remains the strongest frozen-feature baseline we have.", {}),
)
P()
P_runs(
    ("Day 2 still produced four ", {}),
    ("qualitative findings", {"bold": True}),
    (" that are paper-worthy and complement the entanglement story — see Section 4 below.", {}),
)

# === All PPEP results table ===
H2("1. All PPEP results across 4 encoder/grid configurations")
P("Subject-disjoint, full 1,784-clip DAiSEE test set. Best LR and best Ridge ordinal per pooling per encoder.")

table = doc.add_table(rows=1, cols=5)
table.style = "Light Grid Accent 1"
for i, h in enumerate(["Encoder (grid)", "Pooling", "LR κ_q [95% CI]", "Ridge κ_q [95% CI]", "Baseline CLS κ_q (from earlier)"]):
    table.rows[0].cells[i].text = h
    for run in table.rows[0].cells[i].paragraphs[0].runs:
        run.bold = True

ppep_sources = [
    ("DINOv2 ViT-B/14 (16×16)", os.path.join(BASE, "results", "ppep", "ppep_results.json"), "LR 0.107"),
    ("TIPSv2-B14 @ 224 (16×16)", os.path.join(BASE, "results", "ppep", "ppep_results_tipsv2.json"), "(TIPS CLS not tuned)"),
    ("TIPSv2-B14 @ 448 (32×32)", os.path.join(BASE, "results", "ppep", "ppep_results_tipsv2_448.json"), "(TIPS CLS not tuned)"),
    ("CLIP ViT-L/14 (16×16)", os.path.join(BASE, "results", "ppep", "ppep_results_more_encoders.json"), "LR 0.128 / Ridge 0.189"),
]
for enc_label, path, baseline_str in ppep_sources:
    d = load_json(path)
    if d is None:
        continue
    # For "more_encoders" file, key is "CLIP-L/14"
    if enc_label.startswith("CLIP ViT-L"):
        d = d.get("CLIP-L/14", d)
    for kind in ["face_feat", "bg_feat", "cls_feat", "mean_feat"]:
        if kind not in d:
            continue
        lr = d[kind]["logreg"]
        rd = d[kind]["ridge_ordinal"]
        lr_str = f"{lr['kappa_q']:.3f} [{lr['kappa_q_ci'][0]:.3f}, {lr['kappa_q_ci'][1]:.3f}]"
        rd_str = f"{rd['kappa_q']:.3f} [{rd['kappa_q_ci'][0]:.3f}, {rd['kappa_q_ci'][1]:.3f}]"
        row = table.add_row().cells
        row[0].text = enc_label
        row[1].text = kind
        row[2].text = lr_str
        row[3].text = rd_str
        row[4].text = baseline_str

P()
P_runs(
    ("The strongest PPEP result is ", {}),
    ("CLIP-L/14 cls_feat Ridge = 0.158 [0.116, 0.198]", {"bold": True}),
    (", which is essentially identical to our earlier CLIP-L Ridge baseline (0.189) — i.e. the result was already in our overnight runs. ", {}),
    ("The face_feat poolings top out at 0.118 (CLIP-L Ridge), still below the encoder's own CLS.", {}),
)

# === Concat experiments (DINOv2) ===
H2("2. Concatenation experiments (DINOv2 ViT-B/14)")
P("Testing whether patch info is complementary to CLS, even if it can't replace CLS.")
concat = load_json(os.path.join(BASE, "results", "ppep", "ppep_concat_results.json"))
if concat:
    table = doc.add_table(rows=1, cols=3)
    table.style = "Light Grid Accent 1"
    for i, h in enumerate(["Config", "LR κ_q [95% CI]", "Ridge κ_q [95% CI]"]):
        table.rows[0].cells[i].text = h
        for run in table.rows[0].cells[i].paragraphs[0].runs:
            run.bold = True
    for cfg in ["cls_only", "face_only", "bg_only", "face+bg", "face+cls", "bg+cls", "face+bg+cls", "face+bg+cls+mean"]:
        if cfg not in concat:
            continue
        lr = concat[cfg]["logreg"]; rd = concat[cfg]["ridge_ordinal"]
        row = table.add_row().cells
        row[0].text = cfg
        row[1].text = f"{lr['kappa_q']:.3f} [{lr['kappa_q_ci'][0]:.3f}, {lr['kappa_q_ci'][1]:.3f}]"
        row[2].text = f"{rd['kappa_q']:.3f} [{rd['kappa_q_ci'][0]:.3f}, {rd['kappa_q_ci'][1]:.3f}]"
    P_runs(
        ("face+cls Ridge gives κ_q = 0.116 vs cls-only Ridge 0.090 (+29% rel) but LR shows the opposite pattern (face+cls 0.085 vs cls 0.107). ", {}),
        ("Inconsistent across probe families; single seed; not a clean method finding.", {"italic": True}),
    )

# === Qualitative findings ===
H2("3. Qualitative findings worth keeping in the paper")
items = [
    ("CLS-vs-patch preference flips with pretraining objective.",
     "Contrastive encoders (CLIP, DINOv2) → CLS attention integration beats patch pooling. Spatially-aware MIM encoders (TIPS) → patches beat CLS. On TIPS-B14, face Ridge 0.094 vs CLS Ridge 0.038. On CLIP-L, CLS Ridge 0.158 vs face Ridge 0.118."),
    ("Face localization does not concentrate the engagement signal.",
     "Background patches give comparable κ to face patches on all encoders we tested. Engagement variance is not face-region-specific in frozen features."),
    ("Encoder scale dominates pooling strategy.",
     "CLIP-L face_feat (0.118) > DINOv2 face_feat (0.067). Bigger encoder helps more than better pooling. Consistent with our overnight finding that SigLIP-L CLS (0.199) breaks the small-encoder ceiling."),
    ("TIPS at 32×32 fine grid breaks its CLS.",
     "On TIPS-448, CLS Ridge κ_q = -0.108 (worse than random). The 448 input pushes the model off-distribution. Underscores that 'denser patches' is not a free lunch."),
]
for title, body in items:
    p = doc.add_paragraph(style="List Number")
    p.add_run(title).bold = True
    p.add_run(" — " + body)

# === What this means for the paper ===
H2("4. What this means for the BMVC paper")
P_runs(
    ("PPEP doesn't anchor a methods contribution. ", {}),
    ("The κ ≈ 0.10–0.16 patch-pooled ceiling is roughly the same as the linear-probe ceiling we documented for small encoders. SigLIP-L CLS (0.199) is still the strongest frozen-feature baseline.", {}),
)
P()
P_runs(
    ("What it DOES contribute: ", {"bold": True}),
    ("evidence for the 'exhaustive method search' story. The Day-2 negative result on patch-pooling is a clean addition to the existing list of method families that hit the ceiling. The 'CLS-vs-patch preference flips with pretraining' qualitative finding is genuinely novel and can be a paragraph in the discussion section.", {}),
)
P()
P_runs(
    ("Two paths forward: ", {"bold": True}),
)
P("Path A — Accept the negative result. The paper's narrative (diagnose + mechanism + exhaustive method search + scale-escape with SigLIP-L) now includes PPEP as one more method family that hits the ceiling. Add the qualitative cross-encoder finding as a discussion paragraph. Total of ~10 method families tested.")
P("Path B — One more push: learnable attention pooling over patches. Train a small attention head subject-disjointly that learns which patches to weight for engagement. If this exceeds 0.20 it would be the methods contribution. Estimated cost: ~6-8 hours.")

H2("5. Next-step suggestions")
P("• Get co-author's input on the IDEP/SIEP/PPEP arc and lock the paper framing.")
P("• Decide A vs B above.")
P("• Begin BMVC template + drafting in parallel — the data picture is now essentially complete.")

doc.save(OUT)
print(f"Wrote {OUT}")
