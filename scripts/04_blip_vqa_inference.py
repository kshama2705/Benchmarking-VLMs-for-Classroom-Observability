"""
Step 4: BLIP-VQA engagement classification on DAiSEE frames.

Uses Salesforce BLIP-VQA to classify engagement by asking questions
about the student's state.

Usage:
  python 04_blip_vqa_inference.py --prompt-variant 1 --run 1
  python 04_blip_vqa_inference.py --prompt-variant 2 --run 1
  python 04_blip_vqa_inference.py --prompt-variant 3 --run 1

Output: results/blip_vqa_prompt{N}_run{M}.csv
"""

import argparse
import csv
import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLED_CSV = os.path.join(BASE_DIR, "sampled_test.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# BLIP-VQA gives short answers, so we ask targeted questions
# and map answers to engagement levels
PROMPT_VARIANTS = {
    1: {
        "name": "direct",
        "question": "How engaged is this student? Answer: not engaged, barely engaged, engaged, or highly engaged.",
        "answer_map": {
            "not engaged": 0, "not at all": 0, "disengaged": 0, "distracted": 0, "bored": 0, "no": 0,
            "not very": 1, "not much": 1, "not really": 1,
            "barely engaged": 1, "barely": 1, "slightly": 1, "low": 1, "somewhat": 1, "not": 1,
            "engaged": 2, "attentive": 2, "focused": 2, "yes": 2, "moderate": 2,
            "highly engaged": 3, "highly": 3, "very engaged": 3, "very much": 3, "very": 3, "high": 3, "extremely": 3,
        },
    },
    2: {
        "name": "attention",
        "question": "Is this student paying attention to the screen? Answer: not at all, a little, yes, or very much.",
        "answer_map": {
            "not at all": 0, "no": 0, "none": 0,
            "not very": 1, "not much": 1, "not really": 1, "not": 1,
            "a little": 1, "little": 1, "slightly": 1, "barely": 1, "somewhat": 1,
            "yes": 2, "attentive": 2, "focused": 2, "paying attention": 2,
            "very much": 3, "very": 3, "extremely": 3, "absolutely": 3, "completely": 3,
        },
    },
    3: {
        "name": "behavioral",
        "question": "What is this student doing? Answer: sleeping, distracted, watching, or actively studying.",
        "answer_map": {
            "sleeping": 0, "sleep": 0, "asleep": 0, "napping": 0, "resting": 0,
            "distracted": 1, "looking away": 1, "bored": 1, "texting": 1, "phone": 1,
            "watching": 2, "looking": 2, "listening": 2, "sitting": 2, "reading": 2,
            "actively studying": 3, "studying": 3, "writing": 3, "working": 3, "typing": 3, "taking notes": 3,
        },
    },
}


def parse_answer(answer_text, answer_map):
    """Map BLIP's free-form answer to an engagement level."""
    answer_lower = answer_text.lower().strip()

    # Try exact match first
    if answer_lower in answer_map:
        return answer_map[answer_lower]

    # Try substring match (longest first)
    sorted_keys = sorted(answer_map.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key in answer_lower:
            return answer_map[key]

    # Default to level 2 (most common class) if unparseable
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-variant", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--model-name", type=str, default="Salesforce/blip-vqa-base")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_file = os.path.join(RESULTS_DIR, f"blip_vqa_prompt{args.prompt_variant}_run{args.run}.csv")

    # Load model
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading BLIP-VQA model: {args.model_name}...")
    processor = BlipProcessor.from_pretrained(args.model_name)
    model = BlipForQuestionAnswering.from_pretrained(args.model_name).to(device)
    model.eval()

    # Get prompt config
    variant = PROMPT_VARIANTS[args.prompt_variant]
    question = variant["question"]
    answer_map = variant["answer_map"]
    print(f"Prompt variant {args.prompt_variant} ({variant['name']}): {question}")

    # Load samples
    samples = load_samples()
    print(f"\nProcessing {len(samples)} frames...")

    results = []
    raw_answers = []
    unparsed_count = 0

    for i, sample in enumerate(samples):
        frame_path = sample["frame_path"]
        if not os.path.exists(frame_path):
            print(f"  SKIP: {frame_path} not found")
            continue

        image = Image.open(frame_path).convert("RGB")
        inputs = processor(image, question, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=20)
            answer = processor.decode(output[0], skip_special_tokens=True)

        predicted = parse_answer(answer, answer_map)
        raw_answers.append(answer)

        results.append({
            "clip_id": sample["clip_id"],
            "predicted": predicted,
            "ground_truth": sample["engagement"],
            "raw_answer": answer,
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(samples)} | Last answer: '{answer}' -> {predicted}")

    # Save results (without raw_answer in main CSV, save separately)
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

    print(f"\nResults saved to {output_file}")
    print(f"Raw answers saved to {raw_file}")

    # Stats
    pred_dist = Counter(r["predicted"] for r in results)
    answer_dist = Counter(raw_answers)
    print(f"Prediction distribution: {dict(sorted(pred_dist.items()))}")
    print(f"\nTop 10 raw answers:")
    for ans, count in answer_dist.most_common(10):
        print(f"  '{ans}': {count}")


if __name__ == "__main__":
    main()
