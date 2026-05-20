# BMVC 2026 Pivot Brief — Engagement VLM Paper

**Date:** 2026-05-06 · **Deadlines:** Abstract 22 May, Paper 29 May (no extensions)

## Where we are
- **CV4Edu (CVPR Workshop):** Accepted to non-archival track. Doesn't bar later peer-reviewed submission.
- **CV4Edu archival track:** Rejected. Reviewer cLFD (confidence 5) flagged: single-frame protocol invalidates temporal task; N=300 sampling biased; SCB scene-level reformulation not comparable to prior work; GPT-4o L2 fallback corrupts metrics; "pure benchmarking without methodological innovation is weak."
- **Plan:** Resubmit a substantially strengthened version to BMVC 2026.

## Original BMVC plan (Framing A — "Diagnose-and-Adapt")
The reviewer signal said benchmark-only is dead. The plan was to keep the diagnostic half and add an adaptation half: linear probing, class-prior calibration, multi-frame inference, ordinal CLIP. Goal: *"the signal is in the encoder, the language readout is broken — here's how to fix it."*

The paper hinged on one experiment: **subject-disjoint linear probe on CLIP features**.
- κ_quadratic ≥ 0.25 → encoder has signal → Framing A works
- κ_quadratic < 0.15 → encoder lacks signal → Framing A fails, pivot needed

Decision was made to drop SCB and run on DAiSEE only (cleaner ordinal task, official subject-disjoint splits, comparable to prior literature).

## What we ran
Linear probes on the **full 1,784-clip DAiSEE test set**, trained on the full official train split (5,358 clips, subject-disjoint from test). Three independent variations to rule out single-encoder artifacts:

| Probe | κ_quadratic (95% CI bootstrap) |
|---|---|
| CLIP ViT-B/32, single frame, LogReg (balanced) | 0.055 [0.014, 0.091] |
| CLIP ViT-B/32, single frame, Ridge (ordinal) | 0.098 [0.062, 0.134] |
| **CLIP ViT-B/32, mean of 3 frames, LogReg** | **0.101 [0.068, 0.133]** |
| **CLIP ViT-B/32, mean of 3 frames, Ridge** | **0.103 [0.061, 0.140]** |
| **DINOv2 ViT-B/14, single frame, LogReg** | **0.107 [0.072, 0.143]** |
| DINOv2 ViT-B/14, single frame, Ridge | 0.090 [0.042, 0.133] |
| Best zero-shot (LLaVA P3) — for reference | ~0.10 |

## What this tells us
**A κ ≈ 0.10 ceiling holds across three orthogonal axes:**
1. Encoder family — CLIP (vision-language) ≈ DINOv2 (vision-only)
2. Temporal context — 1 frame ≈ 3 frames
3. Readout — categorical ≈ ordinal regression

The best probe is statistically tied with the best zero-shot prompt. None reach 0.15. Framing A's premise (signal in encoder, broken readout) is **decisively rejected**.

## Proposed new direction (Pivot 1 — "The Engagement Ceiling")
The negative result is stronger than the original benchmark because it pre-empts every reviewer attack:

> *Frozen vision features — whether CLIP, DINOv2, or zero-shot VLMs; whether single-frame or temporal; whether read out via prompts or supervised heads — encode an upper bound of κ ≈ 0.10 on individual classroom engagement. The bottleneck is representational, not in the prompt or language readout.*

**Survives:** "you didn't try adaptation" (we did), "single-frame is the problem" (multi-frame also fails), "CLIP-specific quirk" (DINOv2 also fails).

**Implies:** the field needs face/AU-aware encoders or end-to-end fine-tuning, not better prompts. Direct contribution to the explainable-AI track or Brave New Ideas track at BMVC.

## Remaining work (~21 days)
1. Full 1,784-clip zero-shot reruns with bootstrap CIs (CLIP, LLaVA, GPT-4o, Qwen2.5-VL)
2. Multi-frame Qwen2.5-VL inference (closes the temporal-handicap critique reviewer-side too)
3. Class-prior calibration on existing logits (quick sanity check)
4. Drop SCB content; switch to BMVC template; rewrite intro around the ceiling thesis
5. Ethics paragraph (DAiSEE consent / IRB)

## Questions for you
1. Are you on board with Pivot 1, or do you want to push on additional probes (e.g. face-cropped CLIP, fine-tuned probe with more capacity) before locking the framing?
2. Title preference between *"The Engagement Ceiling: Why Frozen Vision Features Cannot Recover Individual Classroom Engagement"* and alternatives?
3. Anything you want to keep from the SCB scene-level work as a side analysis, or drop entirely?
