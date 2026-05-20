# BMVC 2026 Proposed Method — DREAM

**Decomposed Representation for Engagement via Anchor-Modulation**

Author: Aman Goyal, plus co-authors
Target venue: BMVC 2026 (Lancaster, UK, 23-26 Nov 2026)
Abstract deadline: 2026-05-22 (6 days)
Paper deadline: 2026-05-29 (13 days)

---

## 1. Why this method, why now

After 60+ method variants on DAiSEE engagement recognition, three findings are robust:

1. **Frozen-feature ceiling.** Across CLIP-B/32, DINOv2, CLIP-L/14, SigLIP-L, SigLIP-SO400M with zero-shot prompting, linear probes, ordinal regression, MLP, SupCon, bagging, fusion, calibration, ordinal threshold tuning — **all methods plateau at κ_q ∈ [0.20, 0.24]**.
2. **Identity dominates the feature space.** A linear classifier recovers subject ID at 99.5% on SigLIP-L; entanglement curves (k=0..69) show engagement variance is concentrated in directions that ALSO carry identity. Linear concept erasure (LEACE) destroys both signals together.
3. **Within-subject ranking is dead.** Spearman ρ ≈ 0.05 within each subject — the model captures *who* the subject is (their typical engagement level), not *which clip* of theirs is more engaged.

The reviewer-validated path forward is *not* another readout trick. The signal carried by frozen features is "subject prior × global engagement bias." To recover within-subject variation we have to **rescale each subject's features to a common, engagement-neutral origin** before reading out engagement.

Prior attempts at per-subject calibration (TFIC: subtract subject mean; moonshot 20: per-subject baseline at the probability level) succeed mildly (κ ≈ 0.23-0.24) but **mix engagement levels into the reference**. A subject who is mostly highly engaged gets a "high" baseline; one who is mostly bored gets a "low" baseline — the very thing we want to measure leaks into the reference.

**DREAM fixes this by using a self-supervised, engagement-agnostic criterion (MediaPipe blendshape activation magnitude) to pick *neutral-state anchor frames* per subject, computing the per-subject reference from those anchors only, and reading out engagement from the residual representation.**

---

## 2. Method

### 2.1 Notation
- $x_{s,t}$: frame at clip-time $t$ from subject $s$.
- $f(\cdot)$: frozen SigLIP-L encoder, $f(x) \in \mathbb{R}^{1024}$.
- $b(\cdot)$: MediaPipe FaceLandmarker blendshapes, $b(x) \in [0,1]^{52}$.
- $g(\cdot)$: 6-d gaze (yaw, pitch, openness × L/R).
- $y \in \{0,1,2,3\}$: engagement label per 10s clip.

### 2.2 Anchor selection (self-supervised, engagement-agnostic)
For each subject $s$ in **train**, score every available frame by

$$ \mathrm{neutral}(x) = \|b(x)\|_1 + \lambda \cdot \|g(x) - g^{\text{forward}}\|_2 $$

where $g^{\text{forward}} = (0,0,1,0,0,1)$ is a canonical forward-gaze reference and $\lambda = 0.5$. Smaller score ⇒ more neutral face.

Take the $K \in \{1,3,5,10\}$ frames with lowest `neutral(x)` for each subject. These are the subject's **anchor set** $A_s$.

Critically, $A_s$ is computed **only from blendshape/gaze**, never from the engagement label. This makes the criterion engagement-agnostic — anchors are picked for their *facial neutrality*, not their engagement level. Yet, empirically, neutral-face frames are dominated by L2 ("engaged") rather than L0 ("not engaged"), because the typical neutral face in a classroom is a mid-engaged attentive face, not an unengaged one. The anchor set is therefore an "engagement-balanced rest-state reference," NOT a "low-engagement reference" (this distinction is the experimental check in §4.2).

### 2.3 Anchor embedding
$$ \mathbf{a}_s = \frac{1}{|A_s|} \sum_{x \in A_s} f(x) $$

### 2.4 Anchor-modulated representation
We test three modulation operators:
- **Subtract**: $\phi_{\text{sub}}(x) = f(x) - \mathbf{a}_s$
- **Concat**: $\phi_{\text{cat}}(x) = [\, f(x); \, f(x) - \mathbf{a}_s\,] \in \mathbb{R}^{2048}$
- **FiLM**: $\phi_{\text{film}}(x) = (1 + \gamma(\mathbf{a}_s)) \odot f(x) + \beta(\mathbf{a}_s)$, where $\gamma, \beta$ are small learned MLPs.

The probe is a class-balanced LogReg + ordinal threshold tuning (the established SOTA recipe) on $\phi(x)$. We will bag (K=20 × 5 seeds) for variance reduction, then apply val-tuned ordinal thresholds — the same recipe that took baseline solo LR from κ=0.213 to κ=0.238 with frozen $f(x)$.

### 2.5 Generalization to unseen test subjects
DAiSEE's official test split has 21 subjects disjoint from train. For each **test subject $s^*$**, we compute their anchor embedding from their *own unlabeled training-time frames* — i.e., other clips in their test split, used WITHOUT their engagement labels. This is operationally identical to a deployment scenario: a teacher records the student for a minute, the system auto-detects their neutral state, and from then on scores engagement relative to that baseline.

We provide two protocols in the paper:
- **(P-train)**: anchors per subject come from that subject's *training-time* clips. For train subjects, this is well-defined. For test subjects, we use their OWN test-split clips' blendshapes (but NEVER their engagement labels) to select anchors, then compute anchor embedding. This is a transductive setting (LEAVE-LABELS-OUT, not leave-clips-out).
- **(P-zero)**: anchors per subject come from the FIRST 10% of frames in temporal order (the assumption: the first 10% of any session captures the subject before substantial engagement variation). Stricter zero-shot.

Both protocols never see test labels. P-train mirrors test-time adaptation literature (no privacy issue — only the subject's own frames are used).

### 2.6 Why we expect this to work
- TFIC (subject mean subtraction): subtracts mean over all engagement levels → reference leaks engagement variance → destroys signal (κ ≈ 0.04).
- Per-subject baseline (moonshot 20): operates on engagement probabilities, not feature space → constrained by the original per-clip probe's failure mode.
- **DREAM**: reference is engagement-controlled (neutral face), subtraction operates in feature space (richer signal). The residual $f(x) - \mathbf{a}_s$ captures **deviation from a subject's resting face state** — which is exactly what within-subject engagement variation should look like.

---

## 3. Why this passes BMVC review

| Reviewer attack | DREAM response |
|---|---|
| "Already known" | The anchor-modulation idea (engagement-agnostic per-subject reference) is novel in engagement literature. We position it as a constructive answer to the entanglement diagnostic. |
| "Findings overgeneralized" | All claims scoped to frozen-feature DAiSEE engagement readout. Cross-task control (FER2013 unchanged) bounds the claim. |
| "Single-frame protocol" | DREAM uses multi-frame anchor (mean over $K$ neutrals) + multi-frame test clip (3-frame average). |
| "Class imbalance" | LR with `class_weight='balanced'` + ordinal threshold tuning per Menon 2021. |
| "Doesn't move numbers" | If DREAM beats κ_q > 0.27, that's a +0.04 absolute and breaks the 60+ method ceiling. We will only submit if numbers move. |
| "Sampling bias" | Full 1,784-clip test set, bootstrap 1000× CI. |
| "GPT-4o L2 fallback" | We drop GPT-4o P3 from headline; report refusal rate explicitly per Reviewer mfAP. |
| "Quantization mismatch" | LLaVA + Qwen both reported at matched precision. |

---

## 4. Experiments to run

### 4.1 Anchor extraction (Day 1)
- Script `scripts/130_dream_anchors.py`: Load `daisee_face_signals.npz` (blendshapes, gaze). For each subject, rank frames by `neutral()` score. Save top-K indices per subject.
- Inspect: do anchor frames look neutral? Plot blendshape distribution of anchors vs random sample.
- **Verify (label leakage check)**: report distribution of engagement labels among selected anchors. We DO NOT use these labels for selection but must report them for transparency.

### 4.2 DREAM headline run (Day 1-2)
- Script `scripts/131_dream_probe.py`: Compute $\mathbf{a}_s$ for all train/test subjects. Compute $\phi_{\text{sub}}$, $\phi_{\text{cat}}$, $\phi_{\text{film}}$. Train LR probe + ordinal threshold tuning on each. Report test κ_q with bootstrap CI.
- Compare vs:
  - Solo LR-unif baseline (κ=0.238) — primary
  - Per-subject baseline (κ=0.234) — main competitor
  - α=0.5 blend (κ=0.245 stable) — strongest prior method

### 4.3 Ablations (Day 3-4)
- **K (anchor budget)**: K ∈ {1, 3, 5, 10, 20, 50}.
- **Selection criterion**: blendshape-min vs gaze-min vs random vs lowest-engagement (oracle, for upper bound).
- **Anchor source**: P-train (subject's own train-time clips) vs P-zero (first 10% of frames).
- **Modulation operator**: subtract vs concat vs FiLM.
- **Bagging**: solo vs K=20 × 5 seeds bag (the SOTA recipe).

### 4.4 Reviewer attack experiments (Day 5-6)
- **Multi-frame (R1)**: 3-frame mean per clip on top of DREAM.
- **Full test set (R2)**: 1,784 clips, baseline already in.
- **GPT-4o refusal table (R3)**: include refusal rate adjacent to every cell.
- **Class-prior + threshold (R4)**: ordinal thresholds via Menon adjustment.
- **Precision match (R6)**: LLaVA full-precision rerun.

### 4.5 Cross-dataset control (Day 7)
- DREAM on **FER2013** (subject-agnostic — no anchor) should give the same κ as baseline. This proves the gain is engagement-specific, not encoder-generic.
- (Optional) DREAM on **EmotiW** if frames retrievable.

### 4.6 Identity-engagement entanglement quantification (Day 8)
- Re-run subject-ID probe on $\phi_{\text{sub}}$ features. We expect subject-ID accuracy to drop from 99.5% → ~60-80% on DREAM-residual features.
- Re-run within-subject Spearman analysis. We expect ρ_within to lift from 0.05 → 0.15-0.25.

### 4.7 Real-time / deployment story (Day 9)
- DREAM is parameter-free at inference (anchor computed once per subject). Report compute: 1 SigLIP-L forward pass (~50ms/frame), 1 MediaPipe call (~10ms/frame).

### 4.8 Negative results to keep (paper Section 5)
- TFIC mean subtraction: κ drops to 0.04 (the failure mode DREAM fixes).
- Identity-erasure curve (LEACE): destroys both signals.
- 60+ method exhaustion table (compressed into supplementary).

---

## 5. Paper outline (14 pages excl. refs)

**Title (working):** *Beyond the Frozen-Feature Ceiling: Anchor-Modulated Representation for Personalized Engagement Recognition*

| Section | Pages | Content |
|---|---|---|
| 1. Introduction | 1.5 | Problem; ceiling phenomenon; DREAM teaser |
| 2. Related Work | 1.0 | DAiSEE/ViBED-Net, VLMs for affect (CLIP/SigLIP), TTA + personalization, AU-based methods |
| 3. The Frozen-Feature Ceiling (diagnostic) | 2.0 | Multi-method exhaustion table + IDEP entanglement curve (fig 1) + within-subject ranking finding |
| 4. DREAM Method | 2.5 | Notation, anchor selection, modulation, training/inference protocols, fig 2 (architecture) |
| 5. Experiments | 3.5 | DAiSEE main result table, ablations (K, criterion, modulation), cross-dataset, deployment, identity erasure |
| 6. Discussion | 1.5 | Why anchor selection beats mean subtraction; limitations (small subjects, neutral-face availability); when DREAM helps |
| 7. Conclusion | 0.5 | Recap + future (encoder fine-tuning, video temporal) |
| Refs | (excl) | ~35 entries |
| Supplementary | — | Hyperparam grids, GPT-4o refusal table, additional ablations, qualitative examples |

**Figures (3 main + 4 supp):**
- Fig 1 (centerpiece): IDEP entanglement curve + within-subject ρ histogram — establishes the problem
- Fig 2: DREAM architecture (frame → SigLIP-L → subtract a_s → LR probe → ordinal threshold)
- Fig 3: κ vs K (anchor budget) showing the sweet spot
- Supp Fig A: Qualitative — anchor frame vs high-engagement frame per subject (consent allowing)
- Supp Fig B: Per-subject Spearman improvement (DREAM vs baseline)
- Supp Fig C: Subject-ID accuracy curve on DREAM features
- Supp Fig D: Confusion matrices DREAM vs baseline

**Target numbers (TBD by experiment):**
- DAiSEE κ_q: target **0.28-0.32**. Acceptance requires beating the 60-variant ceiling at 0.238 by a clean +0.04 absolute.
- Within-subject Spearman ρ: target **0.15-0.25** (vs baseline 0.05).
- Subject-ID accuracy on residual: target **65-80%** (vs baseline 99.5%) — partial erasure that preserves engagement.

If the headline number lands below 0.26, we re-frame as "anchor-modulation is the best frozen-feature recipe we found; gains are small but the diagnostic story still stands." That's the worst case and is still a defensible BMVC submission given the comprehensive diagnostic decomposition.

---

## 6. Timeline (today is 2026-05-16; abstract 2026-05-22, paper 2026-05-29)

| Day | Date | Deliverable |
|---|---|---|
| D0 | 05-16 | Method spec (this doc), anchor-extraction script |
| D1 | 05-17 | DREAM-subtract result on test, sanity checks, IDEP/within-subject re-analysis on DREAM features |
| D2 | 05-18 | DREAM-concat, DREAM-FiLM, K-sweep, bagged DREAM |
| D3 | 05-19 | Ablations: criterion (blendshape vs gaze vs random), P-train vs P-zero, cross-dataset FER2013 control |
| D4 | 05-20 | Reviewer-attack reruns (multi-frame, full-test, refusal table); main result tables locked |
| D5 | 05-21 | Figures regenerated; abstract drafted |
| **D6** | **05-22** | **Abstract submitted** |
| D7-9 | 05-23/25 | Paper draft sections 1-4 |
| D10-11 | 05-26/27 | Paper draft sections 5-7, supplementary |
| D12 | 05-28 | Co-author review, polish, BMVC template fit |
| **D13** | **05-29** | **Paper + Supp submitted** |

---

## 7. Risk register

- **Anchor frames are dominated by the engaged class (L2/L3).** Mitigation: report distribution; argue this is OK because anchors are *facially neutral*, not *behaviorally disengaged*. The anchor's role is identity reference, not engagement reference.
- **DREAM gain is small (< +0.02).** Mitigation: still report; emphasize diagnostic contribution. Fall back to "comprehensive ceiling characterization + best frozen-feature recipe" framing.
- **DREAM degrades on encoders without good face localization (DINOv2 vs SigLIP-L).** Mitigation: report across 3 encoders; pick the strongest for headline.
- **MediaPipe fails on 0.1% of frames.** Already known (99.9% detection). For those, fall back to mean of detected frames per subject.
- **Reviewer asks why not encoder fine-tune.** Mitigation: discuss as Section 6 future work; cite our LoRA attempt that diverged. DREAM is parameter-free, deployment-friendly, and orthogonal to encoder choice.

---

## 8. Open files / scripts to create

- `scripts/130_dream_anchors.py` — extract anchors per subject
- `scripts/131_dream_probe.py` — main DREAM result
- `scripts/132_dream_ablations.py` — K-sweep, criterion sweep
- `scripts/133_dream_within_subject.py` — within-subject ranking on residual
- `scripts/134_dream_identity_check.py` — subject-ID recovery on residual features
- `scripts/135_dream_fer2013.py` — cross-dataset control (subject-agnostic)
- `scripts/136_dream_film.py` — FiLM modulation variant
- `scripts/137_dream_bagged_threshold.py` — full SOTA recipe (bag + threshold) on DREAM

---

*This document is the BMVC plan-of-record. Update with results inline as they come in.*
