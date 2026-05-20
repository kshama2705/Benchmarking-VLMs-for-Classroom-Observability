"""
Full 1,784-clip CLIP zero-shot inference (P1/P2/P3) with bootstrap CIs.

Output:
  results/full/clip_full_p{1,2,3}.csv     (per-clip predictions)
  results/full/clip_full_metrics.json     (metrics + bootstrap CIs)
"""

import os, csv, json
import numpy as np
import torch
import open_clip
from PIL import Image
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_squared_error

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(BASE, "full_test.csv")
OUT = os.path.join(BASE, "results", "full")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)

PROMPTS = {
    1: {  # minimal
        0: "a photo of a student who is not engaged at all, completely distracted",
        1: "a photo of a student who is barely engaged, passively present",
        2: "a photo of a student who is engaged and attentive",
        3: "a photo of a student who is highly engaged, very focused and alert",
    },
    2: {  # behavioral
        0: "a student looking away from the screen, yawning or distracted, showing no interest",
        1: "a student passively sitting, occasionally glancing at the screen, low attention",
        2: "a student watching the screen attentively, maintaining eye contact with the content",
        3: "a student leaning forward, deeply focused, actively concentrating on the content",
    },
    3: {  # emotional
        0: "a bored and disinterested student with a blank or sleepy expression",
        1: "a slightly disengaged student with a neutral, unfocused expression",
        2: "an attentive student with an interested and alert facial expression",
        3: "a highly focused student showing curiosity and active concentration",
    },
}


def metrics(yt, yp):
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "kappa_quadratic": float(cohen_kappa_score(yt, yp, weights="quadratic")),
        "kappa_linear": float(cohen_kappa_score(yt, yp, weights="linear")),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "mse": float(mean_squared_error(yt, yp)),
        "pred_dist": {int(k): int((yp == k).sum()) for k in range(4)},
        "true_dist": {int(k): int((yt == k).sum()) for k in range(4)},
    }


def bootstrap(yt, yp, n=1000):
    out = {"accuracy": [], "kappa_quadratic": [], "f1_macro": [], "mse": []}
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        a, b = yt[idx], yp[idx]
        out["accuracy"].append(accuracy_score(a, b))
        out["kappa_quadratic"].append(cohen_kappa_score(a, b, weights="quadratic"))
        out["f1_macro"].append(f1_score(a, b, average="macro", zero_division=0))
        out["mse"].append(mean_squared_error(a, b))
    return {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
            for k, v in out.items()}


def main():
    rows = []
    with open(INPUT) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"Loaded {len(rows)} test clips")

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    # Encode images once
    print("Encoding images...")
    feats = np.zeros((len(rows), 512), dtype=np.float32)
    BATCH = 64
    with torch.no_grad():
        for i in range(0, len(rows), BATCH):
            batch = rows[i:i + BATCH]
            imgs = torch.stack([preprocess(Image.open(r["frame_path"]).convert("RGB")) for r in batch]).to(device)
            f = model.encode_image(imgs)
            f = f / f.norm(dim=-1, keepdim=True)
            feats[i:i + len(batch)] = f.cpu().numpy()
    print("Image encoding done.")

    all_metrics = {}
    for prompt_id, templates in PROMPTS.items():
        texts = [templates[k] for k in sorted(templates.keys())]
        with torch.no_grad():
            tt = tokenizer(texts).to(device)
            tf = model.encode_text(tt)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            tf_np = tf.cpu().numpy()

        sims = feats @ tf_np.T
        preds = sims.argmax(axis=1)
        truth = np.array([int(r["engagement"]) for r in rows])

        out_csv = os.path.join(OUT, f"clip_full_p{prompt_id}.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip_id", "predicted", "ground_truth"])
            for r, p in zip(rows, preds):
                w.writerow([r["clip_id"], int(p), r["engagement"]])
        m = metrics(truth, preds)
        ci = bootstrap(truth, preds)
        m["ci95"] = ci
        all_metrics[f"P{prompt_id}"] = m
        print(f"\nP{prompt_id}: acc={m['accuracy']:.3f}  κ_q={m['kappa_quadratic']:.3f} "
              f"[{ci['kappa_quadratic'][0]:.3f},{ci['kappa_quadratic'][1]:.3f}]  "
              f"F1={m['f1_macro']:.3f}  pred_dist={m['pred_dist']}")

    with open(os.path.join(OUT, "clip_full_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics: {os.path.join(OUT, 'clip_full_metrics.json')}")


if __name__ == "__main__":
    main()
