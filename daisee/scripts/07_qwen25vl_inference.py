"""
Step 7: Qwen2.5-VL-7B-Instruct engagement classification on DAiSEE frames.

Usage:
  python scripts/07_qwen25vl_inference.py --prompt-variant 1 --run 1
  python scripts/07_qwen25vl_inference.py --prompt-variant 2 --run 1
  python scripts/07_qwen25vl_inference.py --prompt-variant 3 --run 1

Output: results/qwen25vl_prompt{N}_run{M}.csv

Notes:
  - Run 1 uses temperature=0.0 (deterministic)
  - Runs 2 & 3 use temperature=0.7 (self-consistency)
  - Requires: transformers>=5.0, accelerate, Pillow>=9.1, torch
  - Model is loaded once and reused across all samples (~16GB bfloat16,
    use --load-in-4bit if GPU memory is limited)
"""

import argparse
import csv
import os
import re
import time
from collections import Counter

import torch
from PIL import Image
from tqdm import tqdm

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    MODEL_CLASS = Qwen2_5_VLForConditionalGeneration
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    MODEL_CLASS = Qwen2VLForConditionalGeneration

# ---------------------------------------------------------------------------
# Paths — update BASE_DIR if running from a different working directory
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLED_CSV = os.path.join(BASE_DIR, "sampled_test.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# If the absolute frame_path values in sampled_test.csv don't exist on this
# machine, set NEW_FRAMES_DIR to your local sampled_frames/ location and the
# script will remap them automatically.
NEW_FRAMES_DIR = None  # e.g. "/path/to/sampled_frames"

# ---------------------------------------------------------------------------
# Prompts (copied exactly from experiment guide)
# ---------------------------------------------------------------------------
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
            "You are an educational observer. Assess this student's engagement\n"
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_samples():
    samples = []
    with open(SAMPLED_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_path = row["frame_path"]
            if NEW_FRAMES_DIR and not os.path.exists(frame_path):
                frame_path = os.path.join(NEW_FRAMES_DIR, os.path.basename(frame_path))
            samples.append({
                "clip_id": row["clip_id"],
                "engagement": int(row["engagement"]),
                "frame_path": frame_path,
            })
    return samples


def parse_response(text: str) -> int:
    """
    Extract engagement rating 0-3 from model output.
    For CoT (P3): looks for 'Rating: X' first.
    Falls back to last standalone digit 0-3.
    Returns 2 if nothing found.
    """
    if not text:
        return 2
    # CoT pattern
    m = re.search(r"Rating:\s*([0-3])", text)
    if m:
        return int(m.group(1))
    # Last standalone digit 0-3
    digits = re.findall(r"\b([0-3])\b", text)
    if digits:
        return int(digits[-1])
    return 2


@torch.inference_mode()
def run_inference(processor, model, image_path: str, prompt: str,
                  device: torch.device, max_new_tokens: int = 64,
                  temperature: float = 0.0) -> str:
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    if temperature > 0.0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature

    generated_ids = model.generate(**inputs, **gen_kwargs)
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs["input_ids"], generated_ids)
    ]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-variant", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--hf-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="Load in 4-bit quantisation (requires bitsandbytes)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    # Temperature: run 1 = deterministic, runs 2+ = 0.7
    temperature = 0.0 if args.run == 1 else 0.7

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_slug = "qwen25vl"
    output_file = os.path.join(
        RESULTS_DIR, f"{model_slug}_prompt{args.prompt_variant}_run{args.run}.csv"
    )
    raw_file = output_file.replace(".csv", "_raw.csv")

    variant = PROMPT_VARIANTS[args.prompt_variant]
    samples = load_samples()
    print(f"Loaded {len(samples)} samples from {SAMPLED_CSV}")
    print(f"Prompt variant: P{args.prompt_variant} ({variant['name']}), run {args.run}, temp={temperature}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.hf_model} on {device}...")

    processor = AutoProcessor.from_pretrained(args.hf_model)

    load_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        del load_kwargs["torch_dtype"]

    model = MODEL_CLASS.from_pretrained(args.hf_model, **load_kwargs)
    model.eval()

    # Inference
    results = []
    for i, sample in enumerate(tqdm(samples)):
        frame_path = sample["frame_path"]
        if not os.path.exists(frame_path):
            print(f"  SKIP: {frame_path} not found")
            results.append({
                "clip_id": sample["clip_id"],
                "predicted": 2,
                "ground_truth": sample["engagement"],
                "raw_answer": "SKIPPED: file not found",
            })
            continue

        try:
            raw = run_inference(
                processor, model, frame_path, variant["prompt"],
                device, args.max_new_tokens, temperature,
            )
        except Exception as e:
            raw = f"ERROR: {e}"

        predicted = parse_response(raw)
        results.append({
            "clip_id": sample["clip_id"],
            "predicted": predicted,
            "ground_truth": sample["engagement"],
            "raw_answer": raw,
        })

        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(samples)} | last: '{raw[:60]}' -> {predicted}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    # Save predictions CSV (required columns only)
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "predicted", "ground_truth"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in ["clip_id", "predicted", "ground_truth"]})

    # Save raw answers for debugging
    with open(raw_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["clip_id", "raw_answer", "predicted", "ground_truth"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {output_file}")
    print(f"Saved: {raw_file}")
    dist = Counter(r["predicted"] for r in results)
    print(f"Prediction distribution: {dict(sorted(dist.items()))}")
    gt_dist = Counter(r["ground_truth"] for r in results)
    print(f"Ground truth distribution: {dict(sorted(gt_dist.items()))}")


if __name__ == "__main__":
    main()
