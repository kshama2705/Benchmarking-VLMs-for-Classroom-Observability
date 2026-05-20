"""
Step 8: Multi-frame temporal ablation — CLIP and LLaVA inference over 3 frames per clip.

For each clip, predictions are made at t=2s, t=5s, t=8s and aggregated via:
  - majority_vote: mode of 3 ordinal predictions (ties broken by t=5s frame)
  - score_avg:     mean of the 3 raw CLIP similarity softmax scores, then argmax

Results are saved alongside single-frame baselines for direct comparison.

Usage:
  # CLIP (all prompts, both aggregation strategies):
  python 08_multiframe_inference.py --model clip --prompt-variant 3

  # LLaVA (prompt 3 only; takes ~45 min for 300 clips × 3 frames):
  python 08_multiframe_inference.py --model llava --prompt-variant 3

Output: results/clip_multiframe_p{N}_majvote.csv
        results/clip_multiframe_p{N}_scoreavg.csv
        results/llava_multiframe_p{N}_majvote.csv
"""

import argparse
import base64
import csv
import json
import os
import re
import urllib.request
from collections import Counter
from statistics import mode, StatisticsError

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MULTIFRAME_CSV = os.path.join(BASE_DIR, "sampled_test_multiframe.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

FRAME_TIMES = [2, 5, 8]

# ── CLIP prompt templates (identical to 03_clip_inference.py) ────────────────

CLIP_PROMPTS = {
    1: {
        "name": "minimal",
        "templates": {
            0: "a photo of a student who is not engaged at all, completely distracted",
            1: "a photo of a student who is barely engaged, passively present",
            2: "a photo of a student who is engaged and attentive",
            3: "a photo of a student who is highly engaged, very focused and alert",
        },
    },
    2: {
        "name": "behavioral",
        "templates": {
            0: "a student looking away from the screen, yawning or distracted, showing no interest",
            1: "a student passively sitting, occasionally glancing at the screen, low attention",
            2: "a student watching the screen attentively, maintaining eye contact with the content",
            3: "a student leaning forward, deeply focused, actively concentrating on the content",
        },
    },
    3: {
        "name": "emotional",
        "templates": {
            0: "a bored and disinterested student with a blank or sleepy expression",
            1: "a slightly disengaged student with a neutral, unfocused expression",
            2: "an attentive student with an interested and alert facial expression",
            3: "a highly focused student showing curiosity and active concentration",
        },
    },
}

# ── LLaVA prompt templates (identical to 06_llava_inference.py) ──────────────

LLAVA_PROMPTS = {
    1: {
        "name": "minimal",
        "prompt": (
            "Rate this student's engagement level:\n"
            "0 = not engaged\n"
            "1 = barely engaged\n"
            "2 = engaged\n"
            "3 = highly engaged\n\n"
            "Answer with just the number (0, 1, 2, or 3)."
        ),
    },
    2: {
        "name": "rubric",
        "prompt": (
            "You are an educational observer. Assess this student's engagement "
            "based on their facial expression and body language:\n\n"
            "0 = Distracted, looking away, disinterested, sleepy\n"
            "1 = Passively present but not focused, neutral expression\n"
            "2 = Attentive and following along, maintaining eye contact with screen\n"
            "3 = Actively focused, leaning in, alert, showing curiosity\n\n"
            "Respond with only the number (0, 1, 2, or 3)."
        ),
    },
    3: {
        "name": "chain_of_thought",
        "prompt": (
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
    },
}


# ── Aggregation helpers ───────────────────────────────────────────────────────

def majority_vote(predictions, tiebreak_idx=1):
    """
    Return mode of predictions list.
    On tie (all different), fall back to the tiebreak_idx prediction (default t=5s = index 1).
    """
    try:
        return mode(predictions)
    except StatisticsError:
        return predictions[tiebreak_idx]


def score_avg_predict(sim_scores_per_frame):
    """
    Average raw similarity scores across frames, return argmax.
    sim_scores_per_frame: list of lists, each inner list has 4 similarity values (levels 0-3).
    """
    import torch
    stacked = torch.stack([torch.tensor(s) for s in sim_scores_per_frame])
    avg = stacked.mean(dim=0)
    return int(avg.argmax().item())


# ── Load multi-frame CSV ──────────────────────────────────────────────────────

def load_multiframe_samples():
    samples = []
    with open(MULTIFRAME_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                "clip_id": row["clip_id"],
                "engagement": int(row["engagement"]),
                "frame_t2": row["frame_t2"],
                "frame_t5": row["frame_t5"],
                "frame_t8": row["frame_t8"],
            })
    return samples


# ── CLIP inference ────────────────────────────────────────────────────────────

def run_clip(prompt_variant):
    import torch
    import open_clip
    from PIL import Image

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading CLIP ViT-B/32...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device)
    model.eval()

    variant = CLIP_PROMPTS[prompt_variant]
    text_labels = sorted(variant["templates"].keys())
    text_inputs = [variant["templates"][l] for l in text_labels]

    with torch.no_grad():
        text_tokens = tokenizer(text_inputs).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    samples = load_multiframe_samples()
    print(f"\nRunning CLIP on {len(samples)} clips × 3 frames (P{prompt_variant}: {variant['name']})...")

    majvote_rows = []
    scoreavg_rows = []

    for i, sample in enumerate(samples):
        frame_keys = [f"frame_t{t}" for t in FRAME_TIMES]
        per_frame_preds = []
        per_frame_sims = []

        for fk in frame_keys:
            fp = sample[fk]
            if not os.path.exists(fp):
                print(f"  SKIP: {fp} not found")
                per_frame_preds.append(2)
                per_frame_sims.append([0.25, 0.25, 0.25, 0.25])
                continue

            image = preprocess(Image.open(fp).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                img_feat = model.encode_image(image)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                sims = (img_feat @ text_features.T).squeeze(0)
                pred = text_labels[sims.argmax().item()]
            per_frame_preds.append(pred)
            per_frame_sims.append(sims.cpu().tolist())

        mv = majority_vote(per_frame_preds, tiebreak_idx=1)  # t=5s is index 1
        sa = score_avg_predict(per_frame_sims)

        majvote_rows.append({
            "clip_id": sample["clip_id"],
            "predicted": mv,
            "ground_truth": sample["engagement"],
            "pred_t2": per_frame_preds[0],
            "pred_t5": per_frame_preds[1],
            "pred_t8": per_frame_preds[2],
        })
        scoreavg_rows.append({
            "clip_id": sample["clip_id"],
            "predicted": sa,
            "ground_truth": sample["engagement"],
            "pred_t2": per_frame_preds[0],
            "pred_t5": per_frame_preds[1],
            "pred_t8": per_frame_preds[2],
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(samples)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    _save(majvote_rows, f"clip_multiframe_p{prompt_variant}_majvote.csv")
    _save(scoreavg_rows, f"clip_multiframe_p{prompt_variant}_scoreavg.csv")
    print(f"\nMajority-vote pred dist: {dict(sorted(Counter(r['predicted'] for r in majvote_rows).items()))}")
    print(f"Score-avg pred dist:     {dict(sorted(Counter(r['predicted'] for r in scoreavg_rows).items()))}")


# ── LLaVA inference ───────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"


def parse_llava_response(text):
    rating_match = re.search(r"Rating:\s*(\d)", text)
    if rating_match:
        val = int(rating_match.group(1))
        if 0 <= val <= 3:
            return val
    digits = re.findall(r"\b([0-3])\b", text)
    if digits:
        return int(digits[-1])
    return 2


def query_ollama(prompt, image_path, temperature=0.0, max_retries=3):
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "model": "llava:7b",
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 150},
    }
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                OLLAMA_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "").strip()
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(5)
            else:
                print(f"  WARNING: Failed after {max_retries} retries: {e}", flush=True)
                return ""


def run_llava(prompt_variant):
    variant = LLAVA_PROMPTS[prompt_variant]
    prompt_text = variant["prompt"]

    samples = load_multiframe_samples()
    print(f"Running LLaVA on {len(samples)} clips × 3 frames (P{prompt_variant}: {variant['name']})...")
    print(f"Estimated time: ~{len(samples) * 3 * 5 // 60} min at 5s/query")

    majvote_rows = []
    raw_rows = []

    for i, sample in enumerate(samples):
        frame_keys = [f"frame_t{t}" for t in FRAME_TIMES]
        per_frame_preds = []
        per_frame_raw = []

        for fk in frame_keys:
            fp = sample[fk]
            if not os.path.exists(fp):
                per_frame_preds.append(2)
                per_frame_raw.append("")
                continue
            answer = query_ollama(prompt_text, fp, temperature=0.0)
            pred = parse_llava_response(answer)
            per_frame_preds.append(pred)
            per_frame_raw.append(answer)

        mv = majority_vote(per_frame_preds, tiebreak_idx=1)

        majvote_rows.append({
            "clip_id": sample["clip_id"],
            "predicted": mv,
            "ground_truth": sample["engagement"],
            "pred_t2": per_frame_preds[0],
            "pred_t5": per_frame_preds[1],
            "pred_t8": per_frame_preds[2],
        })
        raw_rows.append({
            "clip_id": sample["clip_id"],
            "raw_t2": per_frame_raw[0],
            "raw_t5": per_frame_raw[1],
            "raw_t8": per_frame_raw[2],
            "pred_t2": per_frame_preds[0],
            "pred_t5": per_frame_preds[1],
            "pred_t8": per_frame_preds[2],
            "predicted_mv": mv,
            "ground_truth": sample["engagement"],
        })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(samples)} | Last preds: {per_frame_preds} -> {mv}", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    _save(majvote_rows, f"llava_multiframe_p{prompt_variant}_majvote.csv")

    raw_path = os.path.join(RESULTS_DIR, f"llava_multiframe_p{prompt_variant}_raw.csv")
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    print(f"\nMajority-vote pred dist: {dict(sorted(Counter(r['predicted'] for r in majvote_rows).items()))}")


# ── Save helper ───────────────────────────────────────────────────────────────

def _save(rows, filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows → {path}")


# ── Metrics summary ───────────────────────────────────────────────────────────

def print_metrics(csv_path, label):
    """Quick metrics printout after inference."""
    try:
        from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_squared_error
        rows = []
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                rows.append(r)
        y_true = [int(r["ground_truth"]) for r in rows]
        y_pred = [int(r["predicted"]) for r in rows]
        acc = accuracy_score(y_true, y_pred) * 100
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        mse = mean_squared_error(y_true, y_pred)
        print(f"  {label}: Acc={acc:.1f}% F1={f1:.3f} κ={kappa:.3f} MSE={mse:.3f}")
    except Exception as e:
        print(f"  {label}: metrics error — {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-frame temporal ablation")
    parser.add_argument("--model", choices=["clip", "llava"], required=True)
    parser.add_argument("--prompt-variant", type=int, choices=[1, 2, 3], required=True)
    args = parser.parse_args()

    if not os.path.exists(MULTIFRAME_CSV):
        print(f"ERROR: {MULTIFRAME_CSV} not found.")
        print("Run 07_extract_multiframes.py first.")
        return

    if args.model == "clip":
        run_clip(args.prompt_variant)
        p = args.prompt_variant
        print("\n── Metrics summary ──")
        print_metrics(os.path.join(RESULTS_DIR, f"clip_multiframe_p{p}_majvote.csv"), f"CLIP P{p} 3-frame majority-vote")
        print_metrics(os.path.join(RESULTS_DIR, f"clip_multiframe_p{p}_scoreavg.csv"), f"CLIP P{p} 3-frame score-avg")
        # Compare with single-frame baseline
        single = os.path.join(RESULTS_DIR, f"clip_prompt{p}_run1.csv")
        if os.path.exists(single):
            print_metrics(single, f"CLIP P{p} single-frame (baseline)")
    elif args.model == "llava":
        run_llava(args.prompt_variant)
        p = args.prompt_variant
        print("\n── Metrics summary ──")
        print_metrics(os.path.join(RESULTS_DIR, f"llava_multiframe_p{p}_majvote.csv"), f"LLaVA P{p} 3-frame majority-vote")
        single = os.path.join(RESULTS_DIR, f"llava_prompt{p}_run1.csv")
        if os.path.exists(single):
            print_metrics(single, f"LLaVA P{p} single-frame (baseline)")


if __name__ == "__main__":
    main()
