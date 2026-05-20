# DAiSEE Experiment Guide — Adding a New Model

> Feed this file to Claude Code and ask it to implement a new model for the DAiSEE engagement classification benchmark. Everything needed to reproduce the existing pipeline and slot in a new model is documented here.

---

## Overview

We benchmark Vision-Language Models (VLMs) on **zero-shot engagement classification** using the **DAiSEE** dataset. The task: given a single video frame of a student, classify their engagement on a 4-point scale (0 = not engaged → 3 = highly engaged).

Existing models already run: **CLIP**, **BLIP-VQA**, **GPT-4o**, **LLaVA-1.5-7B**.

Your job: add **one new model** following the exact same pipeline.

---

## Directory Structure

```
CVPR 2026 Workshop/           ← project root (BASE_DIR)
├── DAiSEE/
│   ├── DataSet/
│   │   └── Test/
│   │       ├── <subject_id>/
│   │       │   └── <clip_id>/
│   │       │       └── <clip_id>.avi   (or .mp4)
│   ├── Labels/
│   │   └── TestLabels.csv              ← ground truth (ClipID, Boredom, Engagement, ...)
│   └── README.txt
├── sampled_frames/                     ← extracted JPG frames (already done)
│   └── <clip_id>.jpg
├── sampled_test.csv                    ← 300 sampled clips (already done)
│   columns: clip_id, engagement, frame_path
├── results/                            ← all prediction CSVs go here
│   ├── clip_prompt1_run1.csv
│   ├── blip_vqa_prompt1_run1.csv
│   ├── gpt4o_prompt2_run1.csv
│   ├── llava_prompt3_run3.csv
│   └── ...
├── scripts/
│   ├── 01_sample_and_extract_frames.py
│   ├── 02_evaluate.py                  ← reusable evaluator
│   ├── 03_clip_inference.py            ← reference implementation
│   ├── 04_blip_vqa_inference.py        ← reference implementation
│   ├── 05_gpt4o_inference.py           ← reference implementation
│   ├── 06_llava_inference.py           ← reference implementation
│   └── run_all.sh
└── venv/                               ← Python virtual environment
```

---

## Dataset: DAiSEE

- **9,068** 10-second video clips of students on webcam during e-learning
- **Labels:** 4-class engagement (0/1/2/3) in `DAiSEE/Labels/TestLabels.csv`
  - CSV columns: `ClipID, Boredom, Engagement, Confusion, Frustration`
  - We use only the **Engagement** column
- **Test set:** 1,784 clips. Heavily imbalanced — only 4 clips at level 0
- **Video path structure:**
  ```
  DAiSEE/DataSet/Test/<subject_id>/<clip_id_no_ext>/<clip_id>.avi
  ```
  (clip_id can be `.avi` or `.mp4`)

### The 300-clip stratified sample (already extracted)

`sampled_test.csv` contains 300 pre-sampled clips with frames already extracted:

| Level | Count | Description |
|-------|-------|-------------|
| 0 | 4 | Not engaged (all available — exhaustive) |
| 1 | 84 | Barely engaged (all available — exhaustive) |
| 2 | 106 | Engaged (random sample from ~850, seed=42) |
| 3 | 106 | Highly engaged (random sample from ~850, seed=42) |

**Do NOT re-run step 1.** The frames are already extracted at `sampled_frames/`. Use `sampled_test.csv` directly.

### Sampling Strategy (details from `scripts/01_sample_and_extract_frames.py`)

```python
random.seed(42)

TARGET_PER_LEVEL = {
    0: None,   # take all (only 4 available)
    1: None,   # take all (only 84 available)
    2: 106,    # random.sample(clips, 106)
    3: 106,    # random.sample(clips, 106)
}
```

**Exact procedure:**
1. Load `DAiSEE/Labels/TestLabels.csv` — use only the `Engagement` column
2. Group all test clips by engagement level
3. For levels 0 and 1: take all clips (too few to sample)
4. For levels 2 and 3: `random.sample(clips, 106)` with `random.seed(42)`
5. `random.shuffle(sampled)` — final list order is also shuffled (same seed)
6. For each clip: extract frame at **t=5s** (temporal midpoint) using:
   ```
   ffmpeg -y -ss 5 -i <video> -frames:v 1 -q:v 2 <output>.jpg
   ```
7. Save to `sampled_frames/<clip_id>.jpg` and write `sampled_test.csv`

**`sampled_test.csv` columns:** `clip_id, engagement, frame_path`

> ⚠️ `frame_path` values are **absolute paths** on the original machine. If running on a new machine, remap them:
> ```python
> frame_path = os.path.join("/your/path/to/sampled_frames", os.path.basename(row["frame_path"]))
> ```

---

## Experiment Design

### Three Prompt Variants

Every model is tested with **3 prompts** to measure prompt sensitivity:

| Variant | Name | Description |
|---------|------|-------------|
| P1 | Minimal | Just the 4-label scale, no context |
| P2 | Rubric-Anchored | Explicit behavioural anchors per level |
| P3 | Chain-of-Thought | Step-by-step: expression → posture → rating |

**The exact prompts used for GPT-4o and LLaVA** (copy these exactly for new generative models):

```
P1 (minimal):
Rate this student's engagement level:
0 = not engaged
1 = barely engaged
2 = engaged
3 = highly engaged

Answer with just the number (0, 1, 2, or 3).

---

P2 (rubric):
You are an educational observer. Assess this student's engagement
based on their facial expression and body language:

0 = Distracted, looking away, disinterested, sleepy
1 = Passively present but not focused, neutral expression
2 = Attentive and following along, maintaining eye contact with screen
3 = Actively focused, leaning in, alert, showing curiosity

Respond with only the number (0, 1, 2, or 3).

---

P3 (chain-of-thought):
Analyze this student watching an educational video.

Step 1: Describe the student's facial expression in one sentence.
Step 2: Describe their body posture in one sentence.
Step 3: Based on your observations, rate their engagement level:
  0 = not engaged at all
  1 = barely engaged
  2 = engaged
  3 = highly engaged

Format your response as:
Expression: ...
Posture: ...
Rating: [0-3]
```

> ⚠️ **GPT-4o warning:** P3 triggered safety refusals ~98% of the time (asking to "describe facial expression" of a real person). If testing GPT-class models, expect this. The parse fallback defaults to level 2 for any non-numeric response.

### Self-Consistency Runs

For **generative models** (temperature > 0): run each prompt 3 times.
- Run 1: `temperature = 0.0`
- Run 2 & 3: `temperature = 0.7`

For **deterministic models** (CLIP, BLIP-VQA): temperature is irrelevant; one run suffices.

---

## Output Format

Every inference script must produce a CSV at:
```
results/<model_name>_prompt<N>_run<M>.csv
```

**Required columns:**
```
clip_id,predicted,ground_truth
9289010118.avi,2,3
9403280234.avi,1,3
...
```

**Optional raw answers file** (for debugging):
```
results/<model_name>_prompt<N>_run<M>_raw.csv
```
Columns: `clip_id, raw_answer, predicted, ground_truth`

---

## Evaluation Metrics

Run `scripts/02_evaluate.py` on any output CSV. It computes:

| Metric | Direction | Notes |
|--------|-----------|-------|
| **Accuracy** | ↑ | 4-class exact match |
| **F1 Macro** | ↑ | Mean of per-class F1; punishes class collapse |
| **Cohen's κ (quadratic weighted)** | ↑ | >0.6 = substantial, <0.2 = near-random |
| **MSE** | ↓ | Ordinal distance; supervised CavT achieves 0.038 |

```bash
# Activate venv first
source venv/bin/activate

# Evaluate a single run
python scripts/02_evaluate.py --predictions results/yourmodel_prompt1_run1.csv

# Evaluate self-consistency (3 runs)
python scripts/02_evaluate.py --consistency \
  results/yourmodel_prompt2_run1.csv \
  results/yourmodel_prompt2_run2.csv \
  results/yourmodel_prompt2_run3.csv

# Evaluate prompt sensitivity
python scripts/02_evaluate.py --prompt-sensitivity \
  results/yourmodel_prompt1_run1.csv \
  results/yourmodel_prompt2_run1.csv \
  results/yourmodel_prompt3_run1.csv

# Full evaluation (all prompts + consistency for a model)
python scripts/02_evaluate.py --full --model yourmodel --results-dir results/
```

---

## Existing Results (for comparison)

### Best result per model

| Model | Best Prompt | Acc. | F1 Macro | κ | MSE |
|-------|------------|------|----------|---|-----|
| CLIP ViT-B/32 | P1 | 37.0% | 0.224 | 0.036 | 1.000 |
| BLIP-VQA | P3 | 35.3% | 0.131 | 0.000 | 0.687 |
| GPT-4o | P3† | 36.0% | 0.153 | 0.039 | 0.690 |
| **LLaVA-1.5-7B** | **P3** | **39.0%** | **0.210** | **0.101** | **0.720** |

† GPT-4o P3 had 98% safety refusals; accuracy is an artifact.

### All prompt variants — DAiSEE

| Model | P1 Acc | P2 Acc | P3 Acc | P1 κ | P2 κ | P3 κ |
|-------|--------|--------|--------|------|------|------|
| CLIP | 37.0% | 31.3% | 35.0% | 0.036 | 0.068 | −0.042 |
| BLIP-VQA | 25.3% | 32.7% | 35.3% | −0.018 | 0.058 | 0.000 |
| GPT-4o | 27.7% | 32.3% | 36.0% | 0.067 | 0.075 | 0.039 |
| LLaVA | 5.3% | 15.7% | 37.7% | 0.037 | 0.045 | 0.008 |

> The **current SOTA zero-shot result is κ = 0.101** (LLaVA P3 run 3). Beat this.

### Key failure modes observed
1. **Class collapse** — all models predict mostly level 2; no model meaningfully predicts level 3 (35% of GT)
2. **Prompt sensitivity** — LLaVA swings 32 pp accuracy across prompts on identical images
3. **Near-random κ** — best κ is 0.101; supervised models reach κ > 0.35

---

## How to Add a New Model

### Step 1: Create the inference script

Create `scripts/07_<modelname>_inference.py`. Model it on one of the existing scripts (e.g., `05_gpt4o_inference.py` for generative, `03_clip_inference.py` for embedding-based).

**Minimal template:**

```python
"""
Step 7: <ModelName> engagement classification on DAiSEE frames.

Usage:
  python 07_<modelname>_inference.py --prompt-variant 1 --run 1
  python 07_<modelname>_inference.py --prompt-variant 2 --run 1
  python 07_<modelname>_inference.py --prompt-variant 3 --run 1

Output: results/<modelname>_prompt{N}_run{M}.csv
"""

import argparse
import csv
import os
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLED_CSV = os.path.join(BASE_DIR, "sampled_test.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Copy the exact prompts from the section above
PROMPT_VARIANTS = {
    1: {"name": "minimal",        "prompt": "..."},
    2: {"name": "rubric",         "prompt": "..."},
    3: {"name": "chain_of_thought","prompt": "..."},
}

def load_samples():
    samples = []
    with open(SAMPLED_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                "clip_id": row["clip_id"],
                "engagement": int(row["engagement"]),
                "frame_path": row["frame_path"],
            })
    return samples

def parse_response(response_text):
    """Extract 0-3 from model output. Return 2 as fallback."""
    import re
    # For CoT prompts, look for "Rating: X" first
    m = re.search(r"Rating:\s*([0-3])", response_text)
    if m:
        return int(m.group(1))
    # Find last standalone digit 0-3
    digits = re.findall(r"\b([0-3])\b", response_text)
    if digits:
        return int(digits[-1])
    return 2  # fallback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-variant", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--run", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_slug = "<modelname>"   # e.g. "gemini", "internvl", "qwen_vl"
    output_file = os.path.join(RESULTS_DIR, f"{model_slug}_prompt{args.prompt_variant}_run{args.run}.csv")

    variant = PROMPT_VARIANTS[args.prompt_variant]
    samples = load_samples()
    print(f"Processing {len(samples)} frames with {args.prompt_variant} ({variant['name']})...")

    results = []
    for i, sample in enumerate(samples):
        frame_path = sample["frame_path"]
        if not os.path.exists(frame_path):
            print(f"  SKIP: {frame_path} not found")
            continue

        # ── YOUR MODEL CALL HERE ─────────────────────────────────
        # Load image, send to model with variant["prompt"], get response
        answer = YOUR_MODEL_CALL(frame_path, variant["prompt"])
        # ────────────────────────────────────────────────────────

        predicted = parse_response(answer)
        results.append({
            "clip_id": sample["clip_id"],
            "predicted": predicted,
            "ground_truth": sample["engagement"],
            "raw_answer": answer,
        })

        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(samples)} | last: '{answer[:60]}' -> {predicted}")

    # Save predictions CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "predicted", "ground_truth"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in ["clip_id", "predicted", "ground_truth"]})

    # Save raw answers for debugging
    raw_file = output_file.replace(".csv", "_raw.csv")
    with open(raw_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "raw_answer", "predicted", "ground_truth"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {output_file}")
    dist = Counter(r["predicted"] for r in results)
    print(f"Prediction distribution: {dict(sorted(dist.items()))}")

if __name__ == "__main__":
    main()
```

### Step 2: Run all 3 prompts

```bash
source venv/bin/activate
python scripts/07_<modelname>_inference.py --prompt-variant 1 --run 1
python scripts/07_<modelname>_inference.py --prompt-variant 2 --run 1
python scripts/07_<modelname>_inference.py --prompt-variant 3 --run 1
```

### Step 3: Run self-consistency (generative models only)

Run the best-looking prompt two more times at temperature 0.7:

```bash
python scripts/07_<modelname>_inference.py --prompt-variant <BEST> --run 2
python scripts/07_<modelname>_inference.py --prompt-variant <BEST> --run 3
```

### Step 4: Evaluate

```bash
# All prompts + consistency for the new model
python scripts/02_evaluate.py --full --model <modelname> --results-dir results/
```

---

## Environment Setup

```bash
# The venv is already set up at project root. Just activate it.
source venv/bin/activate

# Installed packages (as of March 2026):
# numpy 2.4.2, pandas 3.0.1, pillow 12.1.1
# torch 2.10.0, torchvision 0.25.0
# open_clip_torch 3.2.0
# transformers 5.2.0
# openai 2.23.0
# scikit-learn (for evaluation)
# matplotlib 3.10.8, seaborn

# Install any additional packages your model needs:
pip install <package>
```

**Hardware note:** LLaVA was run via **Ollama** (4-bit quantized) on a 16 GB M2 MacBook because the full float16 model (~14 GB) doesn't fit in memory. If your model is large, use Ollama or use `load_in_4bit=True` with bitsandbytes.

---

## Important Notes

1. **Do not re-extract frames.** `sampled_test.csv` and `sampled_frames/` are already ready. All `frame_path` values in the CSV point to absolute paths on the original machine — if paths don't exist, update them:
   ```python
   # At the top of your inference script, optionally remap paths:
   NEW_FRAMES_DIR = "/path/to/sampled_frames"
   # then: frame_path = os.path.join(NEW_FRAMES_DIR, os.path.basename(row["frame_path"]))
   ```

2. **Naming convention matters.** The evaluator discovers files by glob pattern `<model>_*.csv`. Use a consistent `model_slug` (lowercase, no spaces, e.g. `gemini`, `internvl2`, `qwen_vl`).

3. **All 300 samples must be included.** If your model skips any frames, note it in results. Missing predictions skew metrics.

4. **Fallback value is level 2.** If a response can't be parsed (refusal, timeout, empty), default to 2. This is what all existing models do to keep N=300 consistent.

5. **Temperature protocol:**
   - Run 1 of any prompt: `temperature = 0.0` (deterministic)
   - Runs 2 & 3 (consistency only): `temperature = 0.7`

6. **Prediction distribution is critical.** Always print and log what fraction of samples got each label. Class collapse (e.g., 90% of predictions = level 2) is the main failure mode to watch.

---

## Ground Truth Distribution (for reference)

| Level | Label | Count | % of 300 |
|-------|-------|-------|----------|
| 0 | Not engaged | 4 | 1.3% |
| 1 | Barely engaged | 84 | 28.0% |
| 2 | Engaged | 106 | 35.3% |
| 3 | Highly engaged | 106 | 35.3% |

**Majority-class baseline:** predicting level 2 for everything gives **35.3% accuracy** and **κ = 0.0**.
Any model below this in accuracy is worse than random. The F1 Macro and κ metrics are harder to fake.
