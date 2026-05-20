"""
Encode DAiSEE single t=5s frames with DINOv2 ViT-B/14 (vision-only encoder),
then run the same subject-disjoint linear probe used for CLIP.

Tests whether the negative result is CLIP-specific or generalizes across encoders.

Output:
  features/dinov2_vitb14_features.npz
  results/linear_probe/probe_results_dinov2.json
"""

import os
import csv
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score, mean_squared_error,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(BASE_DIR, "frames_full", "manifest.csv")
FEAT_FILE = os.path.join(BASE_DIR, "features", "dinov2_vitb14_features.npz")
OUT_DIR = os.path.join(BASE_DIR, "results", "linear_probe")
RNG = np.random.default_rng(42)
BATCH = 32


def encode_features():
    if os.path.exists(FEAT_FILE):
        print(f"Features already exist at {FEAT_FILE}, skipping encoding.")
        return

    rows = []
    with open(MANIFEST) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["status"] in ("ok", "exists") and os.path.exists(r["frame_path"]):
                rows.append(r)
    print(f"Encoding {len(rows)} frames with DINOv2 ViT-B/14...")

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", source="github", trust_repo=True)
    model = model.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    feats = np.zeros((len(rows), 768), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(rows), BATCH):
            batch = rows[i:i + BATCH]
            imgs = [preprocess(Image.open(r["frame_path"]).convert("RGB")) for r in batch]
            x = torch.stack(imgs).to(device)
            f = model(x)
            feats[i:i + len(batch)] = f.cpu().numpy()
            if (i // BATCH) % 20 == 0:
                print(f"  {i + len(batch)}/{len(rows)}")

    os.makedirs(os.path.dirname(FEAT_FILE), exist_ok=True)
    np.savez_compressed(
        FEAT_FILE,
        clip_id=np.array([r["clip_id"] for r in rows]),
        split=np.array([r["split"] for r in rows]),
        subject_id=np.array([r["subject_id"] for r in rows]),
        engagement=np.array([int(r["engagement"]) for r in rows], dtype=np.int64),
        feat=feats,
    )
    print(f"Saved {feats.shape} -> {FEAT_FILE}")


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa_quadratic": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "kappa_linear": float(cohen_kappa_score(y_true, y_pred, weights="linear")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "pred_dist": {int(k): int((y_pred == k).sum()) for k in range(4)},
    }


def bootstrap_ci(y_true, y_pred, n=1000):
    accs, kqs, f1s, mses = [], [], [], []
    nt = len(y_true)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        yt, yp = y_true[idx], y_pred[idx]
        accs.append(accuracy_score(yt, yp))
        kqs.append(cohen_kappa_score(yt, yp, weights="quadratic"))
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
        mses.append(mean_squared_error(yt, yp))

    def ci(a):
        return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]

    return {
        "accuracy_ci95": ci(accs),
        "kappa_quadratic_ci95": ci(kqs),
        "f1_macro_ci95": ci(f1s),
        "mse_ci95": ci(mses),
    }


def fit_logreg(Xtr, ytr, Xva, yva, Xte, yte):
    best = None
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        score = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or score > best["v"]:
            best = {"C": C, "v": float(score), "clf": clf}
    yhat_te = best["clf"].predict(Xte)
    return {"method": "logreg_balanced", "best_C": best["C"],
            "test_metrics": metrics(yte, yhat_te),
            "test_metrics_ci": bootstrap_ci(yte, yhat_te)}


def fit_ridge(Xtr, ytr, Xva, yva, Xte, yte):
    best = None
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(Xtr, ytr.astype(np.float32))
        yhat = np.clip(np.round(reg.predict(Xva)), 0, 3).astype(int)
        score = cohen_kappa_score(yva, yhat, weights="quadratic")
        if best is None or score > best["v"]:
            best = {"a": alpha, "v": float(score), "reg": reg}
    yhat_te = np.clip(np.round(best["reg"].predict(Xte)), 0, 3).astype(int)
    return {"method": "ridge_ordinal", "best_alpha": best["a"],
            "test_metrics": metrics(yte, yhat_te),
            "test_metrics_ci": bootstrap_ci(yte, yhat_te)}


def main():
    encode_features()
    data = np.load(FEAT_FILE, allow_pickle=True)
    splits, y, X, subj = data["split"], data["engagement"], data["feat"], data["subject_id"]
    tr, va, te = splits == "Train", splits == "Validation", splits == "Test"
    print(f"Train: {tr.sum()}  Val: {va.sum()}  Test: {te.sum()}")
    print(f"Subject overlap train∩test: {len(set(subj[tr]) & set(subj[te]))}")

    print("\n=== DINOv2 LogReg ===")
    res_lr = fit_logreg(X[tr], y[tr], X[va], y[va], X[te], y[te])
    print(json.dumps(res_lr, indent=2))

    print("\n=== DINOv2 Ridge ===")
    res_rd = fit_ridge(X[tr], y[tr], X[va], y[va], X[te], y[te])
    print(json.dumps(res_rd, indent=2))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "probe_results_dinov2.json"), "w") as f:
        json.dump({"logreg": res_lr, "ridge_ordinal": res_rd}, f, indent=2)
    print(f"\nSaved: {os.path.join(OUT_DIR, 'probe_results_dinov2.json')}")


if __name__ == "__main__":
    main()
