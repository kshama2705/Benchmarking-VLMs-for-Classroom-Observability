"""
Encode CLIP ViT-B/32 (laion2b_s34b_b79k) image features for ALL DAiSEE clips.

Matches the encoder used in 03_clip_inference.py so the linear probe
demonstrates "same encoder, different readout".

Output:
  features/clip_vitb32_features.npz
    keys: clip_id (str), split (str), subject_id (str),
          engagement (int), feat (float32, 512-d)
"""

import os
import csv
import numpy as np
import torch
import open_clip
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(BASE_DIR, "frames_full", "manifest.csv")
OUT_DIR = os.path.join(BASE_DIR, "features")
OUT_FILE = os.path.join(OUT_DIR, "clip_vitb32_features.npz")

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
BATCH = 64


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load manifest
    rows = []
    with open(MANIFEST) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["status"] in ("ok", "exists") and os.path.exists(r["frame_path"]):
                rows.append(r)
    print(f"Frames available: {len(rows)}")

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device).eval()

    feats = np.zeros((len(rows), 512), dtype=np.float32)
    clip_ids = []
    splits = []
    subjects = []
    engagements = []

    with torch.no_grad():
        for i in range(0, len(rows), BATCH):
            batch_rows = rows[i:i + BATCH]
            imgs = []
            for r in batch_rows:
                img = Image.open(r["frame_path"]).convert("RGB")
                imgs.append(preprocess(img))
            x = torch.stack(imgs).to(device)
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            feats[i:i + len(batch_rows)] = f.cpu().numpy()
            for r in batch_rows:
                clip_ids.append(r["clip_id"])
                splits.append(r["split"])
                subjects.append(r["subject_id"])
                engagements.append(int(r["engagement"]))
            if (i // BATCH) % 10 == 0:
                print(f"  encoded {i + len(batch_rows)}/{len(rows)}")

    np.savez_compressed(
        OUT_FILE,
        clip_id=np.array(clip_ids),
        split=np.array(splits),
        subject_id=np.array(subjects),
        engagement=np.array(engagements, dtype=np.int64),
        feat=feats,
    )
    print(f"\nSaved {feats.shape} features to {OUT_FILE}")
    print(f"Per-split counts:")
    from collections import Counter
    for s, c in Counter(splits).items():
        print(f"  {s}: {c}")


if __name__ == "__main__":
    main()
