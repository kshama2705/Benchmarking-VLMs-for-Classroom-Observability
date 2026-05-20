"""
Step 3: CLIP zero-shot engagement classification on DAiSEE frames.

Uses open_clip to classify engagement level (0-3) by computing similarity
between frame embeddings and text template embeddings.

Usage:
  python 03_clip_inference.py --prompt-variant 1 --run 1
  python 03_clip_inference.py --prompt-variant 2 --run 1
  python 03_clip_inference.py --prompt-variant 3 --run 1

Output: results/clip_prompt{N}_run{M}.csv
"""

import argparse
import csv
import os
import torch
import open_clip
from PIL import Image
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLED_CSV = os.path.join(BASE_DIR, "sampled_test.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Three prompt variants with different text templates
PROMPT_VARIANTS = {
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


def load_samples():
    """Load the sampled test data."""
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
    parser.add_argument("--model-name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_file = os.path.join(RESULTS_DIR, f"clip_prompt{args.prompt_variant}_run{args.run}.csv")

    # Load model
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading CLIP model: {args.model_name} ({args.pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model = model.to(device)
    model.eval()

    # Prepare text embeddings
    variant = PROMPT_VARIANTS[args.prompt_variant]
    print(f"Prompt variant {args.prompt_variant} ({variant['name']}):")
    text_labels = []
    text_inputs = []
    for level in sorted(variant["templates"].keys()):
        text_labels.append(level)
        text_inputs.append(variant["templates"][level])
        print(f"  Level {level}: {variant['templates'][level]}")

    with torch.no_grad():
        text_tokens = tokenizer(text_inputs).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Load samples
    samples = load_samples()
    print(f"\nProcessing {len(samples)} frames...")

    results = []
    for i, sample in enumerate(samples):
        frame_path = sample["frame_path"]
        if not os.path.exists(frame_path):
            print(f"  SKIP: {frame_path} not found")
            continue

        image = preprocess(Image.open(frame_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ text_features.T).squeeze(0)
            predicted_idx = similarities.argmax().item()
            predicted_level = text_labels[predicted_idx]

        results.append({
            "clip_id": sample["clip_id"],
            "predicted": predicted_level,
            "ground_truth": sample["engagement"],
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(samples)}")

    # Save results
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "predicted", "ground_truth"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_file}")
    print(f"Predictions: {len(results)}")
    pred_dist = Counter(r["predicted"] for r in results)
    print(f"Prediction distribution: {dict(sorted(pred_dist.items()))}")


if __name__ == "__main__":
    main()
