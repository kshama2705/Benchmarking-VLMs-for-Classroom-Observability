"""Brief progress doc v2: 4-column table (Status / Item / Dataset / Result)."""
import os
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "BMVC_progress_brief_2026-05-16.docx")

d = Document()

# Wider page margins so the 4-col table fits
for sec in d.sections:
    sec.left_margin = Inches(0.6)
    sec.right_margin = Inches(0.6)

t = d.add_heading("BMVC 2026 — Progress Brief (2026-05-16)", 0)
t.alignment = WD_ALIGN_PARAGRAPH.CENTER

d.add_paragraph(
    "Headline: 90+ method variants run over the past week. Frozen-feature ceiling on DAiSEE engagement "
    "holds at κ_q ≈ 0.23–0.24. Paper draft underway: diagnostic + DREAM (negative) + one robust adaptation "
    "(per-clip + per-subject blend, +0.014 on average). Abstract due 2026-05-22; paper 2026-05-29."
)

tbl = d.add_table(rows=1, cols=4)
tbl.style = "Light Grid Accent 1"
hdr = tbl.rows[0].cells
hdr[0].text = "Status"
hdr[1].text = "Experiment"
hdr[2].text = "Dataset"
hdr[3].text = "Result"

rows = [
    # ---------------- COMPLETED ----------------
    ("Completed", "Subject-disjoint linear probes (CLIP-B/32, DINOv2)",
     "DAiSEE (5358 / 1429 / 1784)",
     "CLIP κ_q = 0.055 [0.014, 0.091]; DINOv2 κ_q = 0.107 [0.072, 0.143]"),
    ("Completed", "Tier-2 encoder probes (CLIP-L/14, SigLIP-L, SigLIP-SO400M)",
     "DAiSEE (full subject-disjoint splits)",
     "CLIP-L Ridge 0.189 [0.143, 0.230]; SigLIP-L LR 0.199 [0.156, 0.242]; SO400M LR 0.165 [0.119, 0.213]"),
    ("Completed", "Bagged SigLIP-L LR + ordinal threshold (headline)",
     "DAiSEE (full splits)",
     "κ_q = 0.238 [0.191, 0.284]; threshold tuning +0.025 over raw"),
    ("Completed", "Full 1,784-clip zero-shot rerun (A1 reviewer attack)",
     "DAiSEE test (1,784 clips, CLIP-B/32)",
     "P1 κ = 0.014; P2 κ = 0.028; P3 κ = 0.000 — zero-shot is noise at full test scale"),
    ("Completed", "Class-prior calibration (CBU, Menon class-prior, SLD-EM)",
     "DAiSEE (CLIP-B/32 + SigLIP-L)",
     "Best CBU+P3 κ = 0.104; SLD-EM collapses with rare L0; threshold tuning is the only robust lift"),
    ("Completed", "FER2013 cross-task positive control",
     "FER2013 (28,709 train / 7,178 test)",
     "CLIP κ = 0.554 (62.8% acc); DINOv2 κ = 0.544 — same encoders work on emotion when not subject-correlated"),
    ("Completed", "Subject-ID probe (smoking gun)",
     "DAiSEE Train within-subject 80/20",
     "CLIP test_acc = 99.8%; DINOv2 99.6%; SigLIP-L 99.5% — features encode identity at near-perfect linear recoverability"),
    ("Completed", "IDEP entanglement curve (k = 0…69)",
     "DAiSEE (CLIP, DINOv2, SigLIP-L)",
     "Top-2 directions sweet spot; engagement collapses past k≈10; full LEACE destroys both signals"),
    ("Completed", "SIEP adversarial subject-invariant MLP (multi-seed)",
     "DAiSEE (CLIP-B/32, DINOv2)",
     "Mean κ = 0.115 ± 0.047 — within noise of linear; single 0.184 result was lucky seed"),
    ("Completed", "Within-subject ranking analysis",
     "DAiSEE test (per-subject)",
     "Within-subject Spearman ρ ≈ 0.05 (zero); within-subject AUC for L=0 detection 0.80 — model captures subject priors, not per-clip variation"),
    ("Completed", "DREAM personalization (anchor-modulation, cat/sub/FiLM variants)",
     "DAiSEE (SigLIP-L + MediaPipe anchor)",
     "Best DREAM-cat K=5 κ_q = 0.222 (vs baseline 0.232) — constructive negative for paper"),
    ("Completed", "Per-subject baseline assignment",
     "DAiSEE (SigLIP-L bag)",
     "Single best bag κ = 0.242 [0.199, 0.281]; 5-bag mean 0.226 ± 0.012"),
    ("Completed", "Per-clip + per-subject E[y] blend, α=0.5",
     "DAiSEE (SigLIP-L bag, K-fold-CV tuned)",
     "Best single bag κ = 0.250 [0.205, 0.296]; 5-bag mean 0.228 ± 0.018 (clip-only 0.214 ± 0.019, +0.014 robust lift)"),
    ("Completed", "Bagging variants (RSB frac sweep, subject-bootstrap, super-bag K=50×10, PCA, TTA)",
     "DAiSEE (SigLIP-L)",
     "Best RSB-45 κ = 0.220; super_bag mean 0.219 ± threshold 0.245; TTA hurts (crops OOD)"),
    ("Completed", "Non-linear probes (MLP focal, CORN, adapter, MC dropout, mixup, soft-kappa loss)",
     "DAiSEE (SigLIP-L 1024-d)",
     "Best non-linear (adapter lat64+ord) κ = 0.234 — comparable to LR; CORN 0.117 (worst); no non-linear lift over linear"),
    ("Completed", "Fusion / stacking (heterogeneous, rank/geom/power-mean, Dirichlet, OOF-stacking)",
     "DAiSEE multi-encoder + multi-method",
     "Pair fusion 0.223; 3-way 0.213; OOF-stacking 0.226; val-stacking catastrophic overfit 0.108"),
    ("Completed", "Concat features (3-encoder, +face/gaze/pose, +FER2013 emotion soft labels)",
     "DAiSEE (SigLIP-L + CLIP-L + DINOv2 etc.)",
     "3-enc concat κ = 0.151 (overfits 2560-d); +explicit signals 0.193; +emotion 0.193 — all worse than SigLIP-L alone"),
    ("Completed", "Comprehensive negative-results catalogue (34 methods tested)",
     "DAiSEE",
     "Calibration, pseudo-labeling, hard mining, LDA, stratified bagging, logit shift, NN-anchor, hierarchical, etc. — all ≤ 0.23"),
    ("Completed", "Paper draft scaffolding (8 sections, figures)",
     "—",
     "paper_bmvc/main.tex + sec/*.tex + figures/fig_*.pdf"),

    # ---------------- IN PROGRESS ----------------
    ("In progress", "DREAM probe variant runs",
     "DAiSEE (SigLIP-L + anchor)",
     "Running (script 131_dream_probe; PID 98239); >3h elapsed"),
    ("In progress", "VideoMAE-base encoding",
     "DAiSEE raw clips (kinetics-400 backbone)",
     "5,883 / 8,571 = 69% complete (test = 100%, train = 64%); single-LR κ = 0.016 (essentially noise)"),
    ("In progress", "VideoMAE bagged probe + per-subject",
     "DAiSEE VideoMAE features (partial)",
     "Running (script 151; PID 12551); bagged result pending"),

    # ---------------- NEXT ----------------
    ("Next", "SigLIP-L last-block fine-tune (Plan B if VideoMAE negative)",
     "DAiSEE (SigLIP-L + last block + head, low LR)",
     "Script 152 READY; ETA 3-4 h on MPS"),
    ("Next", "Multi-frame SigLIP-L headline number",
     "DAiSEE 3-frame mean features",
     "Pending — quick run on existing manifest"),
    ("Next", "Within-subject ρ histogram figure",
     "DAiSEE test per-subject",
     "Use real per-subject ρ values; replace synthetic placeholder"),
    ("Next", "Reviewer-attack reruns (A2-A8): GPT-4o refusal annotation, CavT scale fix, LLaVA precision-match",
     "DAiSEE",
     "Pending — most are quick column additions"),
    ("Next", "Replace placeholder BMVC LaTeX template with the official one",
     "—",
     "github.com/lwpyh/BMVCTemplate2026"),
    ("Next", "Co-author review of paper draft + framing decision",
     "—",
     "Decide whether the diagnostic + DREAM + blend story is enough, or fine-tune is required"),
]

for s, item, ds, res in rows:
    r = tbl.add_row().cells
    r[0].text = s
    r[1].text = item
    r[2].text = ds
    r[3].text = res

d.save(OUT)
print(f"Saved: {OUT}")
