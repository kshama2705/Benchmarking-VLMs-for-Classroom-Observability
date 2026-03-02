import os
import time
import argparse
from typing import List, Dict

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from transformers import CLIPProcessor, CLIPModel


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff")

# We define all 3 prompt variants here so you can run them via the CLI argument.
PROMPTS = {
    "P1": [
        "a classroom where the majority of students are not engaged at all",
        "a classroom where the majority of students are barely engaged",
        "a classroom where the majority of students are engaged",
        "a classroom where the majority of students are highly engaged",
    ],
    "P2": [
        "a classroom where students are distracted, bowing heads, using phones, or completely disengaged",
        "a classroom where students are passively present but not focused, leaning heavily on desks",
        "a classroom where students are attentive, reading, writing, or following along",
        "a classroom where students are actively focused, raising hands, showing curiosity",
    ],
    "P3": [
        "a classroom with bored, disinterested students showing disengaged posture",
        "a classroom with indifferent students showing neutral, passive posture",
        "a classroom with attentive, interested students showing focused posture",
        "a classroom with excited, curious students showing highly alert posture",
    ]
}


def resolve_image_path(images_dir: str, image_id: str) -> str:
    p = os.path.join(images_dir, image_id)
    if os.path.isfile(p):
        return p
    stem, _ = os.path.splitext(image_id)
    for ext in IMG_EXTS:
        cand = os.path.join(images_dir, stem + ext)
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"Image not found: {image_id} in {images_dir}")


@torch.inference_mode()
def clip_zeroshot_predict(
    processor: CLIPProcessor,
    model: CLIPModel,
    image_path: str,
    text_prompts: List[str],
    device: torch.device,
) -> Dict[str, object]:
    """
    Zero-shot CLIP classification:
      - encode image
      - encode 4 text prompts
      - forward pass directly handles cosine similarity * logit_scale
      - softmax -> probabilities
      - argmax -> class 0..3
    """
    image = Image.open(image_path).convert("RGB")

    # Prepare inputs
    inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # The forward pass automatically applies the learned logit_scale (temperature)
    outputs = model(**inputs)
    
    # Extract the scaled logits (shape 1, 4 -> 4,)
    logits = outputs.logits_per_image.squeeze(0)
    
    # Now softmax works correctly to give sharp probability distributions
    probs = torch.softmax(logits, dim=-1)
    pred = int(torch.argmax(probs).item())

    return {
        "pred": pred,
        "probs": probs.cpu().tolist(),
        "sims": logits.cpu().tolist(), # Storing the scaled logits instead of raw similarities
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--gt_csv", required=True)      # file in root/labels/
    ap.add_argument("--output_csv", required=True)
    
    # Added argument to select prompt variant (P1, P2, P3)
    ap.add_argument("--prompt", type=str, choices=["P1", "P2", "P3"], required=True)

    ap.add_argument("--hf_model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--print_every", type=int, default=25)

    args = ap.parse_args()

    images_dir = os.path.join(args.root, "images", args.split)
    gt_path = os.path.join(args.root, "labels", args.gt_csv)

    df = pd.read_csv(gt_path)
    if args.max_samples is not None:
        df = df.head(args.max_samples)

    # Fetch the exact prompt variant list
    text_prompts = PROMPTS[args.prompt]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(args.hf_model)
    model = CLIPModel.from_pretrained(args.hf_model).to(device)
    model.eval()

    rows = []
    for j, (_, row) in tqdm(enumerate(df.iterrows(), start=1)):
        image_id = str(row["image_id"])
        gt = row["ground_truth"]

        img_path = resolve_image_path(images_dir, image_id)

        try:
            out = clip_zeroshot_predict(processor, model, img_path, text_prompts, device)
            pred = int(out["pred"])
            probs = out["probs"]
            sims = out["sims"]
            err = ""
            code = 0
        except Exception as e:
            pred = 2  # Defaulting to 2 as per your methodology fallback
            probs = [None, None, None, None] # Fixed to None to prevent Pandas mixed-type columns
            sims = [None, None, None, None]
            err = f"{type(e).__name__}: {e}"
            code = 1

        rows.append({
            "image_id": os.path.basename(img_path),
            "image_path": img_path,
            "ground_truth": gt,
            "predicted": pred,
            "sim_0": sims[0],
            "sim_1": sims[1],
            "sim_2": sims[2],
            "sim_3": sims[3],
            "p_0": probs[0],
            "p_1": probs[1],
            "p_2": probs[2],
            "p_3": probs[3],
            "stderr": err,
            "returncode": code,
        })

        if args.print_every > 0 and (j % args.print_every == 0):
            print(f"Processed {j}/{len(df)}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print("Saved:", args.output_csv)


if __name__ == "__main__":
    main()