"""
Runs Qwen2.5-VL-7B-Instruct over an image folder for classroom engagement classification.

Example:
  python inference/qwen25vl_inference.py \
    --root /home/ubuntu/SCB-05-Dataset/5k_HRW_Dataset_combined \
    --split val \
    --gt_csv scb_ground_truth_val.csv \
    --prompt_file prompts/prompt1.txt \
    --output_csv SCB-05-Dataset/runs/output_files/qwen25vl_prompt1_run1.csv \
    --hf_model Qwen/Qwen2.5-VL-7B-Instruct \
    --temperature 0.0
"""

import os
import re
import time
import argparse

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    MODEL_CLASS = Qwen2_5_VLForConditionalGeneration
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    MODEL_CLASS = Qwen2VLForConditionalGeneration


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff")


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


def parse_prediction(text: str) -> int:
    """Extract last standalone digit 0-3. Default to 2."""
    if not text:
        return 2
    matches = re.findall(r"\b([0-3])\b", text)
    if matches:
        return int(matches[-1])
    # Fallback: any digit 0-3 anywhere
    matches = re.findall(r"([0-3])", text)
    if matches:
        return int(matches[-1])
    return 2


@torch.inference_mode()
def qwen_predict(
    processor: AutoProcessor,
    model,
    image_path: str,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
) -> str:
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    # Only pass temperature if sampling is enabled
    if temperature > 0.0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature

    generated_ids = model.generate(**inputs, **generate_kwargs)

    # Trim prompt tokens from output
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return output.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--gt_csv", required=True)      # file in root/labels/
    ap.add_argument("--prompt_file", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--hf_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--print_every", type=int, default=25)
    args = ap.parse_args()

    images_dir = os.path.join(args.root, "images", args.split)
    gt_path = os.path.join(args.root, "labels", args.gt_csv)

    df = pd.read_csv(gt_path)
    if args.max_samples is not None:
        df = df.head(args.max_samples)

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.hf_model} on {device}...")

    processor = AutoProcessor.from_pretrained(args.hf_model)
    model = MODEL_CLASS.from_pretrained(
        args.hf_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    rows = []
    for j, (_, row) in tqdm(enumerate(df.iterrows(), start=1)):
        image_id = str(row["image_id"])
        gt = row["ground_truth"]

        img_path = resolve_image_path(images_dir, image_id)

        try:
            raw = qwen_predict(
                processor, model, img_path, prompt_text,
                device, args.max_new_tokens, args.temperature,
            )
            err = ""
            code = 0
        except Exception as e:
            raw = ""
            err = f"{type(e).__name__}: {e}"
            code = 1

        pred = parse_prediction(raw)

        rows.append({
            "image_id": os.path.basename(img_path),
            "image_path": img_path,
            "ground_truth": gt,
            "predicted": pred,
            "raw_answer": raw,
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
