# BMVC 2026 — Divide-and-Conquer Plan

**Today: 2026-05-07** · **Abstract: 2026-05-22 (15 days)** · **Paper: 2026-05-29 (22 days)**

Two tracks run in parallel for ~10 days, then we converge. Track A is compute-heavy (existing infrastructure on Aman's machine: DAiSEE frames, MPS, Ollama, OpenAI key). Track B is manuscript-heavy and can start day 1 without waiting on experiments.

---

## Track A — Experiments (suggested owner: Aman)

| # | Experiment | Output | Effort | Priority | Notes |
|---|---|---|---|---|---|
| A1 | Full 1,784-clip CLIP zero-shot, P1/P2/P3 (1 run each) | `results/clip_full_p{1,2,3}.csv` + bootstrap CIs | ~1 hr | **P0** | reuse `03_clip_inference.py`, swap input CSV |
| A2 | Full 1,784-clip LLaVA-1.5 zero-shot, P1/P2/P3 (1 run each) | `results/llava_full_p{N}.csv` | ~4-6 hr (overnight) | **P0** | Ollama; reuse `06_llava_inference.py` |
| A3 | Full 1,784-clip GPT-4o zero-shot, P1/P2/P3 + refusal flag | `results/gpt4o_full_p{N}.csv` with refusal column | ~3 hr, ~$27 API | **P0** | drop P3 from headline if refusal > 50%; report adjacent |
| A4 | Full 1,784-clip Qwen2.5-VL zero-shot, P1/P2/P3 (1 run each, single-frame t=5s) | `results/qwen_full_p{N}.csv` | ~6 hr | **P0** | local; if RAM-limited use Together AI (~$10) |
| A5 | Multi-frame Qwen2.5-VL (t=2,5,8 averaged), P1/P2/P3 | `results/qwen_multiframe_full_p{N}.csv` | ~6 hr | **P0** | closes the "single-frame invalidates" reviewer attack |
| A6 | Class-prior calibration on all 4 models' logits (post-hoc) | `results/calibration_deltas.csv` | ~2 hr | **P0** | reads existing model output probabilities; no rerun needed |
| A7 | Self-consistency rerun for LLaVA on N=300 (T=0.7 × 3 seeds) — *only if not already done with full coverage* | `results/llava_consistency.csv` | ~2 hr | **P1** | already partially in `results/`; just needs full audit |
| A8 | (optional) Face-cropped CLIP probe — tests if signal is recoverable from face region only | `features/clip_face_features.npz` + probe results | ~6 hr | P2 | only if Track B is ahead; pre-empts "you didn't isolate the face" |

**Critical path A1–A6** by **day 10 (2026-05-17)**.

---

## Track B — Manuscript, figures, bibliography (suggested owner: co-author)

| # | Task | Output | Effort | Priority | Depends on |
|---|---|---|---|---|---|
| B1 | Clone BMVC 2026 template (lwpyh/BMVCTemplate2026), port existing paper structure | `paper-bmvc/main.tex` skeleton | ~3 hr | **P0** | — |
| B2 | Strip all SCB content (sections 4.4, fig3, related-work mentions, refs) | smaller `paper-bmvc/` | ~2 hr | **P0** | B1 |
| B3 | Draft new intro around "Engagement Ceiling" thesis | `sec/intro.tex` | ~4 hr | **P0** | brief + probe table |
| B4 | Update related work: add Vedernikov 2025, OrdinalCLIP, DINOv2, ViBED-Net, Calibrate-Before-Use, logit adjustment; remove SCB-related | `sec/related.tex`, `refs.bib` | ~4 hr | **P0** | — |
| B5 | Ethics paragraph: DAiSEE consent provenance, minor faces, GPT-4o refusal as a research finding | `sec/ethics.tex` | ~2 hr | **P0** | — |
| B6 | Insert probe results into results section (table + 1-paragraph interpretation) | update `sec/results.tex` | ~2 hr | **P0** | available now from `probe_results*.json` |
| B7 | Regenerate figures: drop fig3 (cross-dataset); update fig1/2/4 with full-test-set numbers; new "ceiling figure" plotting κ across encoders × frames × readouts | `figures/*.pdf` | ~5 hr | **P1** | A1-A6 |
| B8 | Draft abstract (BMVC format, ~250 words) | `sec/abstract.tex` | ~2 hr | **P0** | B3 first |

**Critical path B1–B6, B8** by **day 13 (2026-05-20)**.

---

## Track C — Joint integration (days 11-22)

| # | Task | Owner | When |
|---|---|---|---|
| C1 | Integrate A1-A6 results into Table 1 + Table 2; add CI columns everywhere | both | by 2026-05-19 |
| C2 | Methods section: A drafts experimental setup; B drafts probe methodology | both | by 2026-05-19 |
| C3 | Discussion + conclusion: ceiling interpretation, what comes next, limitations | both | by 2026-05-20 |
| C4 | Internal cross-review of full draft | both | 2026-05-20 to 2026-05-21 |
| C5 | **Abstract submission (HARD)** | Aman | **2026-05-22** |
| C6 | Polish pass + supplementary zip (anonymized prompts, configs, probe code) | B drafts, A reviews | by 2026-05-27 |
| C7 | Final review pass | both | 2026-05-28 |
| C8 | **Paper submission (HARD)** | Aman | **2026-05-29** |

---

## Suggested daily cadence
- **Days 1-3 (08-10 May):** A1, A2 launched; B1, B2 done; B4 in progress.
- **Days 4-7 (11-14 May):** A3, A4 done; A5, A6 in progress; B3, B5, B6 done.
- **Days 8-10 (15-17 May):** A5, A6 wrapped; B7 figures regenerated; B8 abstract draft.
- **Days 11-15 (18-22 May):** C1-C5; abstract submission on day 15.
- **Days 16-22 (23-29 May):** C6-C8; final submission on day 22.

---

## Risk register

| Risk | Mitigation |
|---|---|
| GPT-4o refusal rate on full test set is very high | Document as research finding; drop from headline metrics; report alongside |
| Qwen2.5-VL local install / RAM limits | Switch to Together AI API (~$10) |
| Multi-frame Qwen2.5-VL too slow | Fall back to 8-frame uniform via API or use 3-frame avg already done for CLIP |
| Co-author template migration takes longer than 3 hr | Aman swaps in; co-author takes over A6 calibration (also writeable via Python) |
| Ceiling finding contested by reviewer ("only ViTs tested") | A8 face-cropped probe pre-empts; cite ResNet-based engagement SOTA (ViBED-Net) for context |

---

## Open coordination questions

1. **Compute access:** does co-author have GPU/RAM to take any of A2/A4 off Aman? If yes, parallelize.
2. **Writing voice:** single-pass merge or section-by-section drafting? Recommend section-by-section with tight ownership; merge in C4.
3. **A7 / A8 inclusion:** P1/P2 — only if Track B is ahead by day 10. Don't block on these.
