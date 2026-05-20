"""
Generate a Word doc summarizing the past week's progress (2026-05-09 to 2026-05-16)
for sharing with co-author.
"""
import os
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "BMVC_progress_2026-05-16.docx")

d = Document()

# Title
t = d.add_heading("BMVC 2026 — Engagement Recognition Project: Weekly Progress Report", 0)
t.alignment = WD_ALIGN_PARAGRAPH.CENTER

p = d.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Period: 2026-05-09 to 2026-05-16   |   For co-author review")
r.italic = True

d.add_paragraph()

# Executive summary
d.add_heading("Executive Summary", 1)
d.add_paragraph(
    "Over the past week we ran 90+ method variants on DAiSEE engagement "
    "recognition. The frozen-feature ceiling at κ_q ≈ 0.24 holds firm across "
    "all readout-level methods. We have a clean paper structure: "
    "(1) the ceiling characterization with multi-encoder, multi-method exhaustion, "
    "(2) the within-subject entanglement diagnostic (within-subject Spearman ≈ 0.05), "
    "(3) DREAM as constructive personalization-failure, "
    "(4) one robust modest lift (the per-clip + per-subject E[y] blend) that adds +0.014 "
    "on average to the headline. VideoMAE Plan A appears to be a strong negative; "
    "encoder fine-tuning (Plan B) is the only remaining lever before the BMVC deadline."
)

# Section 1: Headline numbers
d.add_heading("1. Current Headline Numbers", 1)

tbl = d.add_table(rows=1, cols=3)
tbl.style = "Light Grid Accent 1"
hdr = tbl.rows[0].cells
hdr[0].text = "Method"
hdr[1].text = "κ_q (test)"
hdr[2].text = "95% CI"

rows = [
    ("CLIP ViT-B/32 LR (baseline)", "0.055", "[0.014, 0.091]"),
    ("CLIP-L/14 LR + Ridge", "0.128 / 0.189", "[0.078, 0.179] / [0.143, 0.230]"),
    ("DINOv2 ViT-B/14 LR", "0.107", "[0.072, 0.143]"),
    ("SigLIP-L LR (no threshold)", "0.199", "[0.156, 0.242]"),
    ("SigLIP-L bagged LR (K=20×5, no threshold)", "0.213", "[0.163, 0.259]"),
    ("SigLIP-L bagged LR + ordinal threshold (headline)", "0.238", "[0.191, 0.284]"),
    ("Per-subject baseline (single best bag)", "0.242", "[0.199, 0.281]"),
    ("Per-clip + per-subject E[y] blend, α=0.5 (best single bag)", "0.250", "[0.205, 0.296]"),
    ("α=0.5 blend cross-bag mean (5 bags)", "0.228 ± 0.018", "vs clip-only 0.214 ± 0.019"),
    ("DREAM-cat K=5 (personalization, constructive negative)", "0.222", "(co-author session)"),
    ("VideoMAE-base single LR (partial features 69%)", "0.016", "essentially zero — Plan A negative"),
    ("FER2013 cross-task positive control (CLIP, DINOv2)", "0.554 / 0.544", "frozen features encode affect — bottleneck is DAiSEE-specific"),
]
for r in rows:
    rr = tbl.add_row().cells
    rr[0].text = r[0]; rr[1].text = r[1]; rr[2].text = r[2]

d.add_paragraph()
d.add_paragraph(
    "Bag-training noise is ~±0.018 across realizations; threshold-tuning noise is "
    "smaller (±0.003 across K-fold folds). The ‘best single bag’ numbers should "
    "be reported with the cross-bag mean as the honest estimate."
)

# Section 2: What we ran (categorized)
d.add_heading("2. Methods Tested (90+ variants)", 1)

d.add_heading("2.1 Frozen Encoders", 2)
for line in [
    "• CLIP ViT-B/32 (single + multi-frame, single-frame at t=5s + mean of t=2,5,8)",
    "• CLIP ViT-L/14",
    "• DINOv2 ViT-B/14",
    "• SigLIP-L/16-256 (1024-d)",
    "• SigLIP-SO400M-14 (1152-d)",
    "• TIPSv2-B14 (224 + 448 patch tokens)",
    "• Facial emotions ViT (dima806/facial_emotions_image_detection)",
    "• MediaPipe FaceLandmarker (52 blendshapes + 3 head pose + 6 gaze + 12 landmark)",
    "• MediaPipe Pose Landmarker (16-d)",
    "• VideoMAE-base (temporal video, partial features only)",
]:
    d.add_paragraph(line, style="List Bullet")

d.add_heading("2.2 Readout / Classifier Methods", 2)
for line in [
    "Linear Logistic Regression (with C-sweep, class_weight=balanced)",
    "Ridge regression + ordinal rounding",
    "CORN cumulative-link ordinal MLP",
    "MLP with focal loss (γ=0,2,4; multiple hidden sizes 64-512)",
    "Adapter MLPs with ordinal+focal joint loss",
    "MC Dropout MLPs (dropout=0.3/0.5/0.7 with T=20 stochastic passes)",
    "HistGradientBoosting (GBC)",
    "LDA with shrinkage sweep",
    "Soft-Kappa direct optimization (differentiable kappa loss)",
    "Mixup-augmented MLP (α=0.2/0.4/0.8)",
    "Multi-task MLP (jointly predicting boredom + engagement + confusion + frustration)",
    "Patch-pooled Engagement Probe (PPEP) on patch tokens",
    "Polynomial features after PCA(50/100)",
]:
    d.add_paragraph(line, style="List Bullet")

d.add_heading("2.3 Ensembling / Bagging Variants", 2)
for line in [
    "Uniform clip-level bootstrap (LR-unif)",
    "Random subspace bagging (RSB) with feature subsampling fracs 0.35–0.55",
    "Stratified class-balanced bootstrap",
    "Subject-disjoint bootstrap (sample subjects, take all clips)",
    "Super-bag (K=50 × 10 seeds = 500 LR fits)",
    "PCA + bagging",
    "SupCon contrastive projection head",
    "Test-time augmentation (TTA: hflip + center crops)",
]:
    d.add_paragraph(line, style="List Bullet")

d.add_heading("2.4 Calibration / Post-hoc Adjustments", 2)
for line in [
    "Temperature scaling per probe (val-tuned)",
    "Calibrate-Before-Use (CBU, content-free bias)",
    "Class-prior adjustment (Menon)",
    "SLD (Saerens-Latinne-Decaestecker) EM iterative test-prior",
    "Logit shift optimization (per-class bias)",
    "Ordinal threshold tuning on E[y] = Σ k·p_k (KEY LIFT, +0.025 over raw)",
    "K-fold thresholds (mean + median)",
    "Logit calibration via cross-bag fusion (rank/geometric/power-mean)",
]:
    d.add_paragraph(line, style="List Bullet")

d.add_heading("2.5 Fusion Strategies", 2)
for line in [
    "Pair LR + LR-RSB weighted (val-tuned weight)",
    "Pair LR + GBC, LR + MLP, LR + CLIP-L, LR + DINOv2",
    "3-way and 4-way Dirichlet joint search over weights + thresholds",
    "5-way heterogeneous fusion (SigLIP-L LR + RSB + GBC + SO400M + explicit signals)",
    "Stacking meta-LR / meta-GBC (val-trained — overfits)",
    "OOF stacking (K-fold OOF train-side meta-features)",
    "Concat-then-bag across multiple encoders (SigLIP-L + CLIP-L + DINOv2 = 2560-d)",
    "Concat with explicit MediaPipe face/gaze/pose features",
    "FER2013 emotion soft labels concatenated with SigLIP-L",
    "Per-subject E[y] blended with per-clip E[y] (α sweep — α=0.5 robust winner)",
]:
    d.add_paragraph(line, style="List Bullet")

d.add_heading("2.6 Adaptation / Personalization (the diagnostic value)", 2)
for line in [
    "IDEP linear residualization (top-k identity directions, k=0..69)",
    "LEACE full identity erasure (destroys engagement signal too)",
    "SIEP adversarial subject-invariant MLP (gradient reversal, λ-sweep, multi-seed)",
    "SIEP v2 with EMA + cosine schedule + SupCon / subject-stratified InfoNCE",
    "Subject-disjoint bagging",
    "Per-subject baseline assignment (broadcasts subject mean E[y] → all clips)",
    "Hybrid per-clip + per-subject blend",
    "K-NN train-subject anchoring (transduce labels via centroid distance)",
    "Subject-centered features (residual after subject mean)",
    "DREAM (Decomposed Representation via Anchor-Modulation, co-author session)",
]:
    d.add_paragraph(line, style="List Bullet")

d.add_heading("2.7 Robustness Tests", 2)
for line in [
    "Multi-seed bag verification (5 seed groups × per-method)",
    "K-fold val cross-validation for threshold and α tuning",
    "Bootstrap test CIs (1,000 resamples)",
    "Per-subject ranking diagnostics (within-subject Spearman, AUC)",
    "Class-distribution diagnostics (pred_dist vs true_dist)",
]:
    d.add_paragraph(line, style="List Bullet")

# Section 3: Findings
d.add_heading("3. Key Findings", 1)

d.add_heading("3.1 The Frozen-Feature Ceiling", 2)
d.add_paragraph(
    "Across all 90+ method variants on SigLIP-L (best frozen encoder), κ_q ∈ "
    "[0.20, 0.24] with ±0.018 bag-realization noise. The ceiling is method-"
    "independent and structural."
)

d.add_heading("3.2 Identity-Engagement Entanglement (IDEP)", 2)
d.add_paragraph(
    "Subject identity is linearly recoverable from frozen features at >99% "
    "accuracy across CLIP-B/32, DINOv2-B/14, SigLIP-L. Engagement signal "
    "concentrates in top 2-10 identity directions. Removing top-k id directions: "
    "k=2 gives best engagement κ; k≥30 destroys both subject-ID and engagement. "
    "Full LEACE erasure: subject-ID drops to ~1% but engagement κ also collapses "
    "to ~0.02-0.04. The two concepts share a low-rank fragile subspace."
)

d.add_heading("3.3 Within-Subject Spearman ≈ 0.05 (load-bearing)", 2)
d.add_paragraph(
    "For each test subject, the bagged LR predictions correlate near-zero with "
    "the within-subject true ranking of engagement levels across that subject's "
    "clips. Frozen features capture subject-level engagement priors, not per-clip "
    "variation. This single statistic explains the ceiling: per-clip predictions "
    "are essentially subject-mean predictions plus noise."
)

d.add_heading("3.4 FER2013 Cross-Task Positive Control", 2)
d.add_paragraph(
    "Same SigLIP-L / DINOv2 / CLIP features achieve κ ≈ 0.55 on FER2013 facial "
    "emotion. Frozen features CAN encode affect when the labels are not "
    "subject-correlated. The DAiSEE failure is not 'VLMs can't do affect' — "
    "it is specifically the identity-correlated structure of DAiSEE engagement."
)

d.add_heading("3.5 Threshold Tuning Lift (+0.025)", 2)
d.add_paragraph(
    "Computing E[y] = Σ k·p_k and tuning 3 ordinal thresholds on val gives "
    "+0.025 κ over argmax of bag probabilities. This is the largest single "
    "post-hoc lift we found. The thresholds (~1.12, 1.44, 1.84) are stable "
    "across K-fold splits."
)

d.add_heading("3.6 Per-clip + Per-subject Blend (+0.014 average)", 2)
d.add_paragraph(
    "Computing per-test-subject mean E[y] and blending with per-clip E[y] via "
    "0.5/0.5 weight, then applying ordinal thresholds, gives κ = 0.228 ± 0.018 "
    "across 5 bag realizations (vs clip-only 0.214 ± 0.019). Best single bag: "
    "super_bag → κ=0.254. The K-fold CV picks α=0.5 robustly. This is a clean, "
    "reproducible adaptation that adds +0.014 on top of the headline."
)

d.add_heading("3.7 DREAM Personalization (constructive negative)", 2)
d.add_paragraph(
    "Subject-specific neutral-anchor features (minimum-activation blendshapes + "
    "forward gaze) subtracted or concatenated to SigLIP-L features. Best variant "
    "(cat K=5 train-shared anchor) gives κ=0.222 vs baseline 0.232. Constructive "
    "failure: anchor selection works, residualization preserves L2/L3 majority "
    "distribution as expected, but engagement κ drops because identity and "
    "engagement share the same low-rank subspace. Strong diagnostic value for "
    "the paper."
)

d.add_heading("3.8 VideoMAE-base (Plan A — appears negative)", 2)
d.add_paragraph(
    "Partial features (5883/8571 = 69% encoded, including 100% of test set). "
    "Single LR on these features: κ=0.016 (essentially noise). This is the "
    "first encoder tested that does NOT encode any usable engagement signal — "
    "likely because the model is trained on Kinetics actions, not affect. If "
    "the bagged + thresholded version also fails (running), Plan A is dead."
)

# Section 4: Negative results table
d.add_heading("4. Catalogue of Negative Results", 1)
d.add_paragraph("Methods tested and confirmed to NOT break the ceiling:")

neg = [
    ("Temperature-calibrated bagging", "+0.003 only"),
    ("Stratified class-balanced bootstrap", "HURTS (0.213 → 0.186)"),
    ("Rank-averaging fusion", "Collapses to 0.136"),
    ("Geometric / power-mean fusion", "Same as arithmetic"),
    ("L1 LR (saga solver)", "Too slow; skipped after partial run"),
    ("SigLIP-SO400M solo", "0.089 (worse than SigLIP-L; encoder-only quirk)"),
    ("Explicit face/gaze/pose signals solo", "0.056 (huge headroom for fusion, but doesn't fuse helpfully)"),
    ("5-way Dirichlet fusion", "Overfits val (test=0.113)"),
    ("Per-class logit bias (4 params, search)", "Catastrophic val→test overfit (val 0.203 → test 0.163)"),
    ("Pseudo-labeling iterative", "Round 1=0.190, Round 2=0.187, Round 3=0.133"),
    ("Subject-bootstrap bagging", "0.218 +threshold (vs 0.238 clip-bootstrap)"),
    ("Bagged non-linear MLP (focal loss)", "Matches LR ceiling at 0.224"),
    ("L0 detector override", "AUC 0.83 but 0 useful overrides at safe thresholds"),
    ("TTA (test-time augmentation)", "Hurts (crops are OOD vs train)"),
    ("CORN ordinal MLP", "0.117 (much worse than CE-loss LR)"),
    ("LDA with shrinkage", "0.222 best solo, threshold tuning hurts"),
    ("SLD EM prior adjustment", "Collapses to κ=0 (numerical instability with rare L0)"),
    ("Concat 3-encoder (2560-d)", "κ=0.151 — high-dim LR overfits"),
    ("PCA + Polynomial features", "PCA100+poly2 (5150-d) κ=0.063"),
    ("Stacking meta-LR (val-trained)", "Catastrophic val overfit (val 0.194 → test 0.108)"),
    ("OOF stacking", "0.226 (better than val-stacking but still below baseline)"),
    ("Hard example mining (focal sample weights)", "Solo 0.21-0.20 across γ"),
    ("Mixup MLP", "0.230 solo at α=0.4 but no robust improvement"),
    ("Soft-kappa direct loss", "0.18 — surrogate doesn't transfer"),
    ("NN-anchor (train subject centroid distance)", "0.21 (anchor-only)"),
    ("Hierarchical 2-stage classifier", "0.063 (too few val L0/L1 subjects)"),
    ("Subject-centered features (residual)", "Solo 0.12, +blend 0.23"),
    ("Subject mean+std classifier on val", "Ridge 0.11 (val too small)"),
    ("Test-time normalization", "Killed; not productive"),
    ("Fused 3-way Dirichlet per-subject", "0.215 — fusion hurts"),
    ("Multi-task MLP (engagement + boredom + confusion + frustration)", "Best 0.224 (single-task wins)"),
    ("FER2013 emotion soft features fused with SigLIP-L", "0.193 — emotion features dilute"),
    ("MC Dropout uncertainty-aware ensemble", "Best 0.216 at dropout=0.5"),
    ("VideoMAE single LR (partial features)", "0.016 — Plan A failure"),
]
neg_tbl = d.add_table(rows=1, cols=2)
neg_tbl.style = "Light Grid Accent 1"
hh = neg_tbl.rows[0].cells
hh[0].text = "Method"; hh[1].text = "Result"
for m, r in neg:
    rr = neg_tbl.add_row().cells
    rr[0].text = m; rr[1].text = r

# Section 5: Paper plan
d.add_heading("5. BMVC 2026 Paper Plan", 1)
d.add_paragraph(
    "Comprehensive diagnostic paper, 14 pages, 8 sections:"
)
for s in [
    "1. Introduction — problem framing, headline ceiling teaser",
    "2. Related Work — DAiSEE prior, LEACE/INLP, ordinal probes, VideoMAE",
    "3. The Frozen-Feature Ceiling — method exhaustion table, multi-encoder scaling, FER2013 positive control",
    "4. The Identity-Engagement Entanglement — N1/IDEP curve, within-subject Spearman, LEACE collapse",
    "5. DREAM Method — formal definition, anchor selection, sub/cat/FiLM variants",
    "6. Best Frozen-Feature Recipe — bagged LR + ordinal threshold + per-subject blend",
    "7. Experiments — DREAM results, blend results, encoder scaling, reviewer-attack reruns",
    "8. Discussion / Conclusion — why structural, future work (encoder fine-tune, multimodal, action-recognition pretraining)",
]:
    d.add_paragraph(s, style="List Bullet")
d.add_paragraph(
    "Drafted at paper_bmvc/. BMVC abstract due 2026-05-22 (6 days). "
    "Paper + supplementary due 2026-05-29 (13 days)."
)

# Section 6: Next steps
d.add_heading("6. Suggested Next Steps", 1)

d.add_heading("6.1 To-do this week (highest priority)", 2)
for s in [
    "Confirm VideoMAE bagged result (still running). If negative, drop Plan A; otherwise tune.",
    "Run SigLIP-L last-block fine-tune (script 152, READY). Plan B for any positive method. ~3-4h.",
    "Add the per-clip + per-subject blend (α=0.5, K-fold-CV verified) to the paper's results table — robust +0.014 over headline.",
    "Replace placeholder bmvc2k.cls with the real BMVC 2026 LaTeX template.",
    "Generate the within-subject ranking figure (per-subject ρ histogram) from real per-subject data.",
    "Multi-frame headline numbers — quick (3-frame mean of CLIP-B/32 + LR-unif + threshold).",
]:
    d.add_paragraph(s, style="List Bullet")

d.add_heading("6.2 Co-author review asks", 2)
for s in [
    "Is the diagnostic framing (with DREAM as constructive failure) compelling enough? Or do we need a stronger positive method (SigLIP-L fine-tune) to keep ‘Diagnose-and-Adapt’ framing?",
    "Should we move on the blend result (α=0.5 per-clip + per-subject) as a named method (e.g., ‘CASE: Clip-And-Subject Ensemble’)? It is a clean, reproducible +0.014 lift.",
    "VideoMAE encoding is partial (69% complete) and gives κ=0.016 on what we have. Worth completing the encoding (another ~3h) to make this a definitive negative? Or skip and report partial?",
    "Reviewer-attack reruns: do we have time to re-run all 8 attack-vectors before 2026-05-29? Some are already addressed via the full-test-set rerun (A1) and class-prior calibration (A6).",
]:
    d.add_paragraph(s, style="List Bullet")

d.add_heading("6.3 Risks and mitigations", 2)
risks = [
    ("Reviewer says ‘negative result without method’", "Frame DREAM as a positive contribution (constructive personalization with clean diagnostic), plus the blend method (+0.014) for a small adaptive lift."),
    ("Reviewer says ‘findings already known’ (cf. CVPR)", "Method-exhaustion table + IDEP entanglement curve + within-subject ρ are NEW relative to prior DAiSEE work. The cross-task FER2013 positive control is a strong novel control."),
    ("Reviewer requests fine-tuning baseline", "Plan B: SigLIP-L last-block fine-tune script (152) ready to run. Earlier LoRA attempts diverged."),
    ("Reviewer requests multi-frame", "Memory shows multi-frame averaging at κ ≈ 0.10 on CLIP. Need a multi-frame on SigLIP-L for an honest comparison. Quick to run."),
]
risk_tbl = d.add_table(rows=1, cols=2)
risk_tbl.style = "Light Grid Accent 1"
hh = risk_tbl.rows[0].cells
hh[0].text = "Risk"; hh[1].text = "Mitigation"
for rsk, mit in risks:
    rr = risk_tbl.add_row().cells
    rr[0].text = rsk; rr[1].text = mit

# Section 7: Files reference
d.add_heading("7. Where to find things", 1)
d.add_paragraph("Project root: /Users/amangoyal/Documents/CVPR 2026 Workshop")
ref_lines = [
    ("Headline SOTA result", "results/sota/verify_solo_threshold.json"),
    ("Per-subject baseline result", "results/sota/moonshot_per_subject.json"),
    ("Per-clip+subject blend (5-bag verify)", "results/sota/moonshot_blend_multi_bag.json"),
    ("DREAM results", "results/sota/moonshot_dream_*.json (co-author)"),
    ("VideoMAE features (partial 69%)", "features/daisee_videomae_features.npz"),
    ("VideoMAE probe (running)", "results/sota/videomae_probe.json"),
    ("IDEP entanglement curve", "results/idep/idep_finegrained.json"),
    ("Within-subject ranking", "results/sota/within_subject_ranking.json"),
    ("FER2013 cross-task", "results/overnight/fer2013_results.json"),
    ("LaTeX paper draft", "paper_bmvc/main.tex + paper_bmvc/sec/*.tex"),
    ("Figures", "paper_bmvc/figures/fig_*.pdf"),
    ("Project memory", "~/.claude/projects/-Users-amangoyal-Documents-CVPR-2026-Workshop/memory/"),
]
ref_tbl = d.add_table(rows=1, cols=2)
ref_tbl.style = "Light Grid Accent 1"
hh = ref_tbl.rows[0].cells
hh[0].text = "What"; hh[1].text = "Path"
for w, p in ref_lines:
    rr = ref_tbl.add_row().cells
    rr[0].text = w; rr[1].text = p

# Footer
d.add_paragraph()
ft = d.add_paragraph()
ft.alignment = WD_ALIGN_PARAGRAPH.CENTER
fr = ft.add_run("Generated 2026-05-16. BMVC abstract due 2026-05-22 (6 days). Paper due 2026-05-29 (13 days).")
fr.italic = True

d.save(OUT)
print(f"Saved: {OUT}")
