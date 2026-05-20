"""
Step 6: LLaVA-1.5 engagement classification on DAiSEE frames.

Uses LLaVA-1.5-7B via Ollama (4-bit quantized, optimized for Apple Silicon).

Usage:
  python 06_llava_inference.py --prompt-variant 1 --run 1

Output: results/llava_prompt{N}_run{M}.csv
"""

import argparse
import base64
import csv
import json
import os
import re
import urllib.request
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLED_CSV = os.path.join(BASE_DIR, "sampled_test.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

OLLAMA_URL = "http://localhost:11434/api/generate"

# Same prompts as GPT-4o for fair comparison
PROMPT_VARIANTS = {
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


def parse_response(response_text):
    """Extract engagement level from LLaVA response."""
    rating_match = re.search(r"Rating:\s*(\d)", response_text)
    if rating_match:
        val = int(rating_match.group(1))
        if 0 <= val <= 3:
            return val

    digits = re.findall(r"\b([0-3])\b", response_text)
    if digits:
        return int(digits[-1])

    return 2


def load_samples(csv_path=None):
    samples = []
    with open(csv_path or SAMPLED_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                "clip_id": row["clip_id"],
                "engagement": int(row["engagement"]),
                "frame_path": row["frame_path"],
            })
    return samples


def query_ollama(prompt, image_path, temperature=0.0, max_retries=3):
    """Send image + prompt to Ollama LLaVA model."""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": "llava:7b",
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 150,
        },
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
        except (TimeoutError, urllib.error.URLError) as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(5)
            else:
                print(f"  WARNING: Failed after {max_retries} retries: {e}", flush=True)
                return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-variant", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--input-csv", type=str, default=None,
                        help="Override input CSV (default sampled_test.csv)")
    parser.add_argument("--output-prefix", type=str, default="llava_prompt",
                        help="Output filename prefix (default llava_prompt)")
    parser.add_argument("--results-subdir", type=str, default=None,
                        help="Subdir under results/")
    args = parser.parse_args()

    out_dir = RESULTS_DIR if not args.results_subdir else os.path.join(RESULTS_DIR, args.results_subdir)
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, f"{args.output_prefix}{args.prompt_variant}_run{args.run}.csv")

    variant = PROMPT_VARIANTS[args.prompt_variant]
    print(f"Prompt variant {args.prompt_variant} ({variant['name']})")

    temperature = args.temperature
    if args.run > 1 and temperature == 0.0:
        temperature = 0.7
        print(f"  Auto-setting temperature={temperature} for run {args.run}")
    print(f"Temperature: {temperature}")

    samples = load_samples(args.input_csv)
    print(f"\nProcessing {len(samples)} frames...", flush=True)

    results = []
    for i, sample in enumerate(samples):
        frame_path = sample["frame_path"]
        if not os.path.exists(frame_path):
            continue

        answer = query_ollama(variant["prompt"], frame_path, temperature)
        predicted = parse_response(answer)

        results.append({
            "clip_id": sample["clip_id"],
            "predicted": predicted,
            "ground_truth": sample["engagement"],
            "raw_answer": answer,
        })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(samples)} | Last: '{answer[:60]}' -> {predicted}", flush=True)

    # Save results
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "predicted", "ground_truth"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in ["clip_id", "predicted", "ground_truth"]})

    raw_file = output_file.replace(".csv", "_raw.csv")
    with open(raw_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "raw_answer", "predicted", "ground_truth"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_file}")
    pred_dist = Counter(r["predicted"] for r in results)
    print(f"Prediction distribution: {dict(sorted(pred_dist.items()))}")


if __name__ == "__main__":
    main()
