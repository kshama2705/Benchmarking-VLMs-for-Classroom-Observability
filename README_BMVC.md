# BMVC 2026 — DAiSEE Classroom Engagement Recognition

Branch `aman-bmvc-2026-seta` of
`kshama2705/Benchmarking-VLMs-for-Classroom-Observability`.

## What's in this branch

- `paper_bmvc/` — full BMVC paper draft (10 sections, 1.45 k lines LaTeX).
  Compile with `pdflatex main; bibtex main; pdflatex main; pdflatex main`
  once the BMVC `bmvc2k.cls` style file is dropped in.
- `scripts/` — all method scripts (numbered 01 → 211).
- `features/` — cached pre-extracted features for MEME and the diagnostic
  paper (see below).
- `DAiSEE/Labels/` — auxiliary labels (B, E, C, F per clip).
- `frames_full/manifest.csv` — clip-id → (split, subject_id, engagement,
  frame_path). Used by every script as the source of truth.
- `BMVC_*.md`, `GPU_RUNBOOK.md` — co-author brief, plan of record, GPU
  runbook for the future-work supervised method.

Excluded by `.gitignore` (regenerate locally — see commands below):
raw DAiSEE videos (14 GB), extracted JPEG frames (720 MB), `models/`
checkpoints (1.2 GB), `venv/`, large CVPR-submission PDFs/ZIPs.

## Quick start on M5 Pro

```bash
git clone https://github.com/kshama2705/Benchmarking-VLMs-for-Classroom-Observability.git
cd Benchmarking-VLMs-for-Classroom-Observability
git checkout aman-bmvc-2026-seta

# Python env
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision transformers scikit-learn pandas \
    matplotlib seaborn pillow opencv-python mediapipe
```

### Run the validated SOTA recipe (SETA — ~30 min on a CPU)

Reproduces our paper headline κ_q = 0.247 [0.201, 0.294]:

```bash
python3 scripts/87_verify_solo_threshold.py
# Or, for a fresh K-fold run from probability scratch (no cache):
python3 scripts/171b_trainval_fast.py   # train-only baseline reproduction
```

### Run MEME (multi-modal multi-task, our ViBED-Net-beating attempt)

Needs MediaPipe face/pose features (already in this branch) and
SigLIP-L multi-frame features (already in this branch).

```bash
python3 scripts/211_meme_mps.py \
    --epochs 25 --batch 64 --seeds 0,42,2025
```

Runtime estimate on M5 Pro: ~3-6 h per seed (vision is frozen), 12-24 h
for the 3-seed ensemble. Output: `results/sota/meme.json`.

### Run ENGAGENET-X (end-to-end SigLIP-L unfreeze + 8-frame transformer)

Requires the **raw DAiSEE videos** at `DAiSEE/DataSet/` and ~80 GB free
disk. The script auto-extracts 8 frames per clip on first run.

```bash
python3 scripts/210_engagenet_mps.py \
    --epochs 10 --batch 4 --accum 4 --seeds 0,42,2025
```

Runtime estimate on M5 Pro: ~24-36 h for 3 seeds. Output:
`results/sota/engagenet_mps.json`.

## Headline numbers (validated this branch)

| Method | κ_q | 95% CI | Accuracy |
|---|---|---|---|
| **SETA (bagged SigLIP-L LR + threshold)** | **0.247** | [.201, .294] | 55.2% |
| 3-frame mean + concat fusion | 0.236 | [.191, .285] | 54.1% |
| DREAM cat-K5 bagged | 0.218 | [.174, .260] | 49.2% |
| TEAM 3-frame temporal transformer | 0.211 | [.170, .258] | — |
| ResNet18 e2e (3 seeds) | 0.114 | [.073, .154] | 43.4% |
| VideoMAE-base bagged | 0.101 | [.054, .144] | — |
| FER2013 cross-task control | 0.55 | — | 62.8% |
| **Supervised SOTA (ViBED-Net, literature)** | **~0.55** | — | **73%** |

Run MEME on M5 Pro to fill in the row that targets the supervised SOTA.

## Key files

| File | Purpose |
|---|---|
| `paper_bmvc/main.tex` | BMVC paper entry point |
| `paper_bmvc/sec/*.tex` | All 10 sections |
| `paper_bmvc/figures/` | 4 main figures (method landscape, DREAM curve, IDEP, within-subject) |
| `scripts/210_engagenet_mps.py` | End-to-end SigLIP-L fine-tune (MPS-native) |
| `scripts/211_meme_mps.py` | MEME multi-stream multi-task (MPS-native) |
| `scripts/87_verify_solo_threshold.py` | SETA recipe verification |
| `scripts/130_dream_anchors.py` | DREAM anchor extraction |
| `scripts/131b_dream_quick.py` | DREAM probe |
| `scripts/140_make_paper_figures.py` | Figure regeneration |
| `BMVC_BRIEF_2026-05-16.md` | Co-author brief, latest status |
| `BMVC_METHOD_DREAM.md` | DREAM method spec |
| `GPU_RUNBOOK.md` | GPU-required ENGAGENET-X handoff (now superseded by 210_engagenet_mps.py) |

## What to run first on M5 Pro

1. **Verify SETA** — 30 min sanity check that everything compiles and
   the cached SOTA reproduces.
2. **Launch MEME** — overnight run, our best shot at beating ViBED-Net.
3. **(Optional) Launch ENGAGENET-X** — ~24-36 h, replicates supervised
   baseline. Needs the raw DAiSEE videos at `DAiSEE/DataSet/`.
4. **Drop result into paper** — `paper_bmvc/sec/engagenet.tex` and
   `paper_bmvc/sec/meme.tex` (TODO) have placeholders ready.
