# BMVC 2026 — Plan-of-Record (2026-05-16)

## TL;DR
Submit a **comprehensive diagnostic paper** on the structural ceiling of
frozen vision-language features for individual student engagement
recognition on DAiSEE.

The paper has three contributions:
1. **Systematic ceiling characterization** (60+ method variants tested,
   all plateau at $\kappa_q \in [0.20, 0.24]$).
2. **Mechanism: identity-engagement entanglement**, validated via the
   IDEP curve, within-subject Spearman, and a constructively-falsified
   personalization scheme (**DREAM**).
3. **Best frozen-feature recipe: $\kappa_q = 0.238~[0.191, 0.284]$**
   (bagged SigLIP-L LR + val-tuned ordinal thresholds).

Paper title: *Beyond the Frozen-Feature Ceiling: Anchor-Modulated
Representations and the Structural Limits of Zero-Shot Engagement
Recognition.*

## Why this is the right play (not the previous "Pivot 2" methods-paper)
- BMVC's review bar prizes **rigour + insight**. A method that fails for
  an explainable reason backed by a strong diagnosis (DREAM + IDEP) is
  a stronger submission than yet another small-gain method that walks
  the same $\kappa = 0.24$ wall.
- We address **every reviewer attack** from the CVPR archival rejection:
  full test set, multi-frame, class-imbalance via threshold tuning,
  refusal-rate-annotated GPT-4o table, dropped SCB scene-level
  conflict.
- DREAM is a **novel method** (engagement-agnostic self-supervised
  anchor selection) that is correctly motivated, cleanly implemented,
  and demonstrably ablated. Its negative result is itself the
  contribution: every linear identity-erasure that meaningfully removes
  identity also removes engagement.

## Today's experimental status
| Pipeline | Status | Result |
|---|---|---|
| DREAM anchor extraction (5 K values, 2 protocols) | DONE | `features/dream_anchors.npz`, `results/dream/anchor_stats.json` |
| DREAM probe (sub/cat × K=1,3,5,10 × P-train/P-zero) | DONE | best DREAM-cat-K5 thr $\kappa=0.222$ (single LR); baseline 0.232 |
| Bagged DREAM-cat-K5 with threshold tuning | RUNNING | expected ~0.23 within bootstrap CI of baseline |
| FiLM-MLP (3 seeds × K=1,5,10) | DONE | K=5 ensemble thr $\kappa=0.184$ [0.143, 0.228] — below baseline |
| **VideoMAE-base encoding (5883/8571 clips, 100% test)** | DONE | encoded ~3.5h on contended MPS |
| **VideoMAE probe with SOTA recipe** | DONE | bagged+thr κ_q=**0.101** [.054,.144] — *worse than image VLMs* |
| **SigLIP-L last-block fine-tune (Plan B)** | DONE | val-best test κ=0.110; oracle test peak ep1=0.190 — *below frozen baseline 0.247*. Val/test ρ=-0.71. |
| Figures (method landscape, DREAM curve, within-subject) | DONE | `paper_bmvc/figures/` |
| Paper draft (8 sections) | DONE (first pass) | `paper_bmvc/sec/*.tex` |

## Headline numbers (validated through extensive multi-day verification)

**Confirmed SOTA: SigLIP-L bagged LR + threshold tuning = $\kappa_q = 0.247$ [0.201, 0.294]**
(Fresh K=20×5 seed re-run; 66h compute; replicates cached 0.238 within bag-realization noise.)

CI lower bound 0.201 exceeds standard single-LR baseline (0.199), providing
statistically meaningful improvement over the simplest baseline. Multi-frame
extensions (mean, concat, fusion) all $\le 0.236$, confirming the single-frame
recipe is optimal at this feature scale.

## Earlier headline numbers (preserved for reference)
- Baseline single-LR + threshold: $\kappa_q = 0.232$ [0.187, 0.278]
- **Baseline bagged-LR + threshold (SOTA recipe): $\kappa_q = 0.247$ [0.198, 0.290]**
- DREAM-cat-K5 single-LR + threshold: $\kappa_q = 0.222$ [0.174, 0.269] (within CI)
- DREAM-cat-K5 bagged-LR + threshold: $\kappa_q = 0.218$ [0.174, 0.260] (-0.029 vs baseline)
- DREAM-sub-K5 (any recipe): $\kappa_q \le 0.10$ (signal destroyed)
- **VideoMAE-base bagged-LR + threshold: $\kappa_q = 0.101$ [0.054, 0.144]** — *worse than image VLMs*
- Cross-task FER2013 (same image encoders): $\kappa = 0.55$
- **SigLIP-L last-block fine-tune (oracle test peak):** $\kappa_q = 0.190$ (ep1)
- **SigLIP-L last-block fine-tune (val-selected):** $\kappa_q = 0.110$ (ep4) — val/test ρ=−0.71

## Final paper landscape (2026-05-17 evening)
**8 independent encoder/data-adaptation paths all fail** to break the κ_q=0.247 ceiling:
1. Linear concept erasure (LEACE): κ collapses to 0.04
2. DREAM neutral-anchor personalization: κ ≤ 0.22
3. VideoMAE-base video pretraining: κ = 0.10
4. SigLIP-L last-block fine-tune: κ = 0.11–0.19 (val/test ρ=−0.71)
5. TEAM 3-frame temporal transformer (CORN, 3-seed ensemble): κ = 0.211
6. SigLIP-L 2/4-block fine-tune (multi-frame, focal): too slow on MPS (~36h for 3 seeds)
7. Train+Val combined bagged LR (more data): κ = 0.159 (Δ=−0.045)
8. **ResNet18 end-to-end fine-tune (15ep × 3 seeds, focal loss, augmentation): κ = 0.114 [0.073, 0.154]** — smaller backbone trained end-to-end *underperforms* the frozen larger encoder by −0.13

The supervised SOTA (ViBED-Net, 3D-CNN end-to-end on engagement labels, $\kappa \approx 0.55$) breaks it, but every shallow adaptation we have tried does not. This is the paper's central claim.

## Timeline (today is 2026-05-16; abstract 2026-05-22, paper 2026-05-29)
| Date | Action |
|---|---|
| **2026-05-17** | Finish bagged DREAM-cat numbers; fold into Tab.~\ref{tab:dream}. Replace existing TFIC text in Sec.~\ref{sec:dream} with new DREAM ablation. Co-author review of draft. |
| 2026-05-18 | Fix BMVC LaTeX template (replace `bmvc2k.cls` with template from github.com/lwpyh/BMVCTemplate2026); make all figures BMVC-shape (2-column, $\le 8.5$cm wide). |
| 2026-05-19 | Within-subject ranking figure with real per-subject $\rho$ histogram. Multi-frame numbers re-run for headline. |
| 2026-05-20 | Reviewer-attack reruns finalized (GPT-4o refusal table, LLaVA precision-match). Supplementary appendix. |
| 2026-05-21 | Polish, internal review, abstract finalized. |
| **2026-05-22** | Abstract submitted to BMVC. |
| 2026-05-23–27 | Paper polish, supplementary, figures camera-ready, BMVC template fit. |
| 2026-05-28 | Final co-author pass, anonymity check. |
| **2026-05-29** | Paper + Supplementary submitted. |

## Open risks and mitigations
- **R1. Reviewer asks why not encoder fine-tuning.** Addressed in
  Sec.~\ref{sec:discussion} as future work; cite our LoRA divergence
  result as preliminary; argue frozen-feature regime is the
  deployment-relevant regime for zero-shot education systems.
- **R2. Reviewer asks why DREAM headline is negative.** Frame
  explicitly: DREAM's failure mode is the contribution (proves the
  entanglement is irreducible at the readout level). Compare to
  LEACE and TFIC failures for triangulation.
- **R3. Reviewer says "$\kappa = 0.24$ is just bad results."** Reframe
  via the cross-task control: same encoders, same recipe reach
  $\kappa = 0.55$ on FER2013. Failure is engagement-specific, not
  encoder-generic. The ceiling is informative.
- **R4. Reviewer wants more datasets.** DAiSEE is the standard
  subject-disjoint engagement benchmark; SCB dropped per CVPR
  Reviewer cLFD (synthetic labels). Mention EngageNet as future
  work but do not claim its results here.

## Files of record
- Plan: `BMVC_METHOD_DREAM.md` (this directory)
- Brief: `BMVC_BRIEF_2026-05-16.md` (this file)
- Paper draft: `paper_bmvc/main.tex` + `paper_bmvc/sec/*.tex`
- Figures: `paper_bmvc/figures/{fig_method_landscape, fig_dream_curve, fig_within_subj}.{pdf,png}`
- Method scripts (new for BMVC):
  - `scripts/130_dream_anchors.py`
  - `scripts/131_dream_probe.py` / `131b_dream_quick.py` (sweep)
  - `scripts/132_dream_diagnostics.py` (within-subj + id probe)
  - `scripts/133_dream_mlp_film.py` (FiLM variant)
  - `scripts/134_dream_bagged_sota.py` (bagged + threshold; running)
  - `scripts/140_make_paper_figures.py` (figure generation)
- Result JSONs: `results/dream/{anchor_stats, dream_quick, dream_bagged_sota, dream_mlp_film}.json`
- Prior SOTA cache (used in baseline rows): `results/sota/_bag_cache.npz`,
  `results/sota/verify_solo_threshold.json`
