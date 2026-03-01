import os
import re
import time
import argparse
from typing import Tuple
from tqdm import tqdm

import pandas as pd
from PIL import Image

import torch
from transformers import BlipProcessor, BlipForQuestionAnswering


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
    """
    Try hard to extract a 0-3 rating from BLIP output.
    BLIP often answers in words ("highly engaged", "two", etc.) depending on question.
    We'll support both digit + common word mappings.
    Default: 2
    """
    if not text:
        return 2

    s = text.strip().lower()

    # 1) digits (standalone or embedded)
    m = re.findall(r"\b([0-3])\b", s)
    if m:
        return int(m[-1])
    m = re.findall(r"([0-3])", s)
    if m:
        return int(m[-1])

    # 2) common words
    word_map = {
        "zero": 0, "0": 0, "none": 0, "not engaged": 0, "no": 0,
        "one": 1, "1": 1, "barely": 1, "slightly": 1, "low": 1,
        "two": 2, "2": 2, "engaged": 2, "medium": 2, "moderate": 2,
        "three": 3, "3": 3, "highly": 3, "very": 3, "high": 3,
    }
    # crude keyword check
    for k, v in word_map.items():
        if k in s:
            return v

    return 2


def build_question(prompt_text: str) -> str:
    """
    BLIP VQA expects a *question*.
    If your prompt file is instruction-y, we convert it into a direct question.
    """
    p = prompt_text.strip()
    # If it's already a question, keep it.
    if "?" in p:
        return p

    # Strong question that tends to yield short answers
    return (
        "Rate the overall classroom engagement level based on the majority of students. "
        "Answer with only one number: 0, 1, 2, or 3."
    )


@torch.inference_mode()
def blip_vqa_answer(
    processor: BlipProcessor,
    model: BlipForQuestionAnswering,
    image_path: str,
    question: str,
    device: torch.device,
    max_new_tokens: int = 3,
) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)

    # generate short answer
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=3,
        do_sample=False,
    )
    ans = processor.decode(out_ids[0], skip_special_tokens=True)
    return ans.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--gt_csv", required=True)      # file in root/labels/
    ap.add_argument("--prompt_file", required=True)
    ap.add_argument("--output_csv", required=True)

    ap.add_argument("--hf_model", default="Salesforce/blip-vqa-base")
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
        prompt_text = f.read()

    question = build_question(prompt_text)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained(args.hf_model)
    model = BlipForQuestionAnswering.from_pretrained(args.hf_model).to(device)
    model.eval()

    rows = []
    for j, (_, row) in tqdm(enumerate(df.iterrows(), start=1)):
        image_id = str(row["image_id"])
        gt = row["ground_truth"]

        img_path = resolve_image_path(images_dir, image_id)

        try:
            raw = blip_vqa_answer(processor, model, img_path, question, device)
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