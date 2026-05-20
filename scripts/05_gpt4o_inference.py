"""
Step 5: GPT-4o engagement classification on DAiSEE frames.

Uses OpenAI API to classify engagement level from frames.

Usage:
  export OPENAI_API_KEY=your_key_here
  python 05_gpt4o_inference.py --prompt-variant 1 --run 1

Output: results/gpt4o_prompt{N}_run{M}.csv
"""

import argparse
import csv
import os
import base64
import time
import re
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLED_CSV = os.path.join(BASE_DIR, "sampled_test.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

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


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_response(response_text):
    """Extract engagement level from GPT-4o response."""
    # Try to find a single digit 0-3
    # For chain-of-thought, look for "Rating: X"
    rating_match = re.search(r"Rating:\s*(\d)", response_text)
    if rating_match:
        val = int(rating_match.group(1))
        if 0 <= val <= 3:
            return val

    # Look for standalone digit
    digits = re.findall(r"\b([0-3])\b", response_text)
    if digits:
        return int(digits[-1])  # take last digit found

    # Fallback
    return 2


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


def main():
    from openai import OpenAI

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-variant", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    client = OpenAI()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_file = os.path.join(RESULTS_DIR, f"gpt4o_prompt{args.prompt_variant}_run{args.run}.csv")

    variant = PROMPT_VARIANTS[args.prompt_variant]
    prompt_text = variant["prompt"]
    print(f"Prompt variant {args.prompt_variant} ({variant['name']})")
    print(f"Temperature: {args.temperature}")

    # For self-consistency runs, use temperature > 0
    temperature = args.temperature
    if args.run > 1 and temperature == 0.0:
        temperature = 0.7
        print(f"  Auto-setting temperature={temperature} for run {args.run}")

    samples = load_samples()
    print(f"\nProcessing {len(samples)} frames...")

    results = []
    for i, sample in enumerate(samples):
        frame_path = sample["frame_path"]
        if not os.path.exists(frame_path):
            continue

        b64_image = encode_image(frame_path)

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}",
                                    "detail": "low",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=150,
                temperature=temperature,
            )
            answer = response.choices[0].message.content.strip()
            predicted = parse_response(answer)
        except Exception as e:
            print(f"  ERROR on {sample['clip_id']}: {e}")
            time.sleep(5)
            continue

        results.append({
            "clip_id": sample["clip_id"],
            "predicted": predicted,
            "ground_truth": sample["engagement"],
            "raw_answer": answer,
        })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(samples)} | Last: '{answer[:50]}' -> {predicted}")

        # Rate limiting
        time.sleep(0.5)

    # Save
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
