"""
Runs LLaVA (or other vision-capable Ollama models) over an image folder using the
Ollama HTTP API (NOT the CLI), so you can reliably pass images + temperature.

Example:
  python inference/llava_ollama_inference.py \
    --root /home/ubuntu/SCB-05-Dataset/5k_HRW_Dataset_combined \
    --split val \
    --gt_csv scb_ground_truth_val.csv \
    --prompt_file prompts/prompt1.txt \
    --output_csv SCB-05-Dataset/runs/output_files/llava_prompt1_run1.csv \
    --model llava:7b \
    --temperature 0.0 \
    --max_samples 50
"""

import os
import argparse
import time
import re
import json
import base64
import urllib.request
import urllib.error
from typing import Tuple, Optional, Dict, Any

import pandas as pd
from tqdm import tqdm

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff")


def resolve_image_path(images_dir: str, image_id: str) -> str:
    """Resolve image_id to an existing file path, trying common extensions."""
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
    matches = re.findall(r"\b[0-3]\b", text)
    return int(matches[-1]) if matches else 2


def file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ollama_generate(
    model: str,
    prompt: str,
    image_path: str,
    temperature: float,
    host: str = "http://localhost:11434",
    timeout_s: int = 600,
    extra_options: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, int]:
    """
    Call Ollama HTTP API /api/generate with base64 image(s).

    Returns: (stdout_text, stderr_text, returncode_like_int)
      - returncode 0 means success
      - returncode 1 means failure, stderr contains error
    """
    url = host.rstrip("/") + "/api/generate"

    img_b64 = file_to_b64(image_path)

    options: Dict[str, Any] = {"temperature": float(temperature)}
    if extra_options:
        options.update(extra_options)

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": options,
    }

    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        # /api/generate returns {"response": "...", ...}
        out = (data.get("response") or "").strip()
        return out, "", 0

    except urllib.error.HTTPError as e:
        # HTTP error with body
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return "", f"HTTPError {e.code}: {e.reason}\n{body}".strip(), 1

    except Exception as e:
        return "", f"{type(e).__name__}: {e}", 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--gt_csv", required=True)  # file in root/labels/
    ap.add_argument("--prompt_file", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--model", default="llava:7b")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--max_samples", type=int, default=None)

    # Ollama API settings
    ap.add_argument("--ollama_host", default="http://localhost:11434")
    ap.add_argument("--timeout_s", type=int, default=600)

    # Debug / resilience
    ap.add_argument("--print_every", type=int, default=25)
    ap.add_argument("--fail_fast", action="store_true", help="Stop on first API failure.")
    args = ap.parse_args()

    images_dir = os.path.join(args.root, "images", args.split)
    gt_path = os.path.join(args.root, "labels", args.gt_csv)

    df = pd.read_csv(gt_path)
    if args.max_samples is not None:
        df = df.head(args.max_samples)

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()
    rows = []
    for idx, row in tqdm(df.iterrows()):
        image_id = str(row["image_id"])
        gt = row["ground_truth"]

        img_path = resolve_image_path(images_dir, image_id)

        raw, err, code = ollama_generate(
            model=args.model,
            prompt=prompt_text,
            image_path=img_path,
            temperature=args.temperature,
            host=args.ollama_host,
            timeout_s=args.timeout_s,
        )

        pred = parse_prediction(raw)

        rows.append(
            {
                "index": int(idx),
                "image_id": os.path.basename(img_path),
                "image_path": img_path,
                "ground_truth": gt,
                "predicted": pred,
                "raw_answer": raw,
                "stderr": err,
                "returncode": code,
            }
        )

        if code != 0 and args.fail_fast:
            print(f"[FAIL_FAST] idx={idx} image={img_path}")
            print(err)
            break

        n_done = len(rows)
        if args.print_every > 0 and (n_done % args.print_every == 0):
            print(f"Processed {n_done}/{len(df)} (last: {os.path.basename(img_path)})")

        if args.sleep > 0:
            time.sleep(args.sleep)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print("Saved:", args.output_csv)


if __name__ == "__main__":
    main()