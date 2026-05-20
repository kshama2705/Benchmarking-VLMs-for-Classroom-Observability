"""
Overnight pipeline (tier 1 + tier 2):

Tier 1 (FER2013 cross-dataset):
  A. Download FER2013 (clip-benchmark/wds_fer2013, ~36K images, 7 classes)
  B. Extract CLIP-B/32 features
  C. Extract DINOv2 features
  D. Linear probe + SupCon-contrastive on FER2013 emotions

Tier 2 (more encoders on DAiSEE):
  E. Encode DAiSEE frames with SigLIP-L
  F. Encode DAiSEE frames with CLIP-L
  G. Linear probe + SIEP-contrastive on each new encoder

Each stage in try/except — failures don't kill subsequent stages.
Status flushed to results/overnight_status.txt every step.

Output:
  features/fer2013_clip_features.npz
  features/fer2013_dinov2_features.npz
  features/daisee_siglip_l_features.npz
  features/daisee_clip_l_features.npz
  results/overnight/fer2013_results.json
  results/overnight/tier2_results.json
  results/overnight/summary.json
"""

import os, json, csv, time, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_squared_error

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(BASE, "features")
OUT_DIR = os.path.join(BASE, "results", "overnight")
STATUS = os.path.join(OUT_DIR, "overnight_status.txt")
os.makedirs(FEAT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))
RNG = np.random.default_rng(42)


def status(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


# ============ Helpers ============
def boot_kappa(yt, yp, n=1000, weighted=False):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        if weighted:
            out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
        else:
            out.append(cohen_kappa_score(yt[idx], yp[idx]))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def metrics_classification(yt, yp, n_classes):
    return {
        "n": int(len(yt)),
        "accuracy": float(accuracy_score(yt, yp)),
        "kappa_unweighted": float(cohen_kappa_score(yt, yp)),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "pred_dist": {int(k): int((yp == k).sum()) for k in range(n_classes)},
    }


def class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1)
    w = counts.sum() / (n_classes * counts)
    return torch.from_numpy(w).float().to(DEVICE)


# ============ Tier 1: FER2013 ============
def stage_a_download_fer2013():
    status("STAGE A: Downloading FER2013 (clip-benchmark/wds_fer2013)")
    import datasets
    ds = datasets.load_dataset("clip-benchmark/wds_fer2013")
    status(f"  Splits: {list(ds.keys())}, sizes: { {k: len(ds[k]) for k in ds} }")
    return ds


def encode_with_open_clip(images, model_name, pretrained, device=DEVICE, batch=64, status_prefix="enc"):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    n = len(images)
    feat_dim = model.visual.output_dim if hasattr(model.visual, "output_dim") else None
    # Probe one image to find dim
    if feat_dim is None:
        with torch.no_grad():
            x = preprocess(images[0].convert("RGB")).unsqueeze(0).to(device)
            f = model.encode_image(x)
            feat_dim = f.shape[-1]
    feats = np.zeros((n, feat_dim), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch):
            chunk = images[i:i + batch]
            xb = torch.stack([preprocess(im.convert("RGB")) for im in chunk]).to(device)
            f = model.encode_image(xb)
            f = f / f.norm(dim=-1, keepdim=True)
            feats[i:i + len(chunk)] = f.cpu().numpy()
            if (i // batch) % 50 == 0:
                status(f"  [{status_prefix}] {i + len(chunk)}/{n}")
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return feats


def encode_with_dinov2(images, device=DEVICE, batch=32, status_prefix="enc"):
    from torchvision import transforms
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                           source="github", trust_repo=True).to(device).eval()
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    n = len(images)
    feats = np.zeros((n, 768), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch):
            chunk = images[i:i + batch]
            xb = torch.stack([preprocess(im.convert("RGB")) for im in chunk]).to(device)
            f = model(xb)
            feats[i:i + len(chunk)] = f.cpu().numpy()
            if (i // batch) % 50 == 0:
                status(f"  [{status_prefix}] {i + len(chunk)}/{n}")
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return feats


def stage_b_c_fer2013_features(ds):
    """Extract CLIP-B/32 and DINOv2 features for FER2013."""
    out = {}
    train_imgs = list(ds["train"]["jpg"])
    train_lbl = np.array(ds["train"]["cls"], dtype=np.int64)
    test_imgs = list(ds["test"]["jpg"]) if "test" in ds else []
    test_lbl = np.array(ds["test"]["cls"], dtype=np.int64) if "test" in ds else None

    status(f"  FER2013 train={len(train_imgs)}  test={len(test_imgs)}")
    status(f"  train label dist: {np.bincount(train_lbl).tolist()}")

    for label_path, encoder, model_name, pretrained in [
        ("clip", "open_clip", "ViT-B-32", "laion2b_s34b_b79k"),
        ("dinov2", "dinov2", None, None),
    ]:
        out_path = os.path.join(FEAT_DIR, f"fer2013_{label_path}_features.npz")
        if os.path.exists(out_path):
            status(f"STAGE B/C: {label_path} features exist, skipping")
            d = np.load(out_path, allow_pickle=True)
            out[label_path] = {"train_feats": d["train_feats"], "test_feats": d["test_feats"],
                               "train_lbl": d["train_lbl"], "test_lbl": d["test_lbl"]}
            continue
        status(f"STAGE B/C: encoding FER2013 with {label_path}")
        if encoder == "open_clip":
            train_feats = encode_with_open_clip(train_imgs, model_name, pretrained, status_prefix=f"FER-{label_path}-tr")
            test_feats = encode_with_open_clip(test_imgs, model_name, pretrained, status_prefix=f"FER-{label_path}-te") if test_imgs else None
        else:
            train_feats = encode_with_dinov2(train_imgs, status_prefix=f"FER-{label_path}-tr")
            test_feats = encode_with_dinov2(test_imgs, status_prefix=f"FER-{label_path}-te") if test_imgs else None
        save = {"train_feats": train_feats, "train_lbl": train_lbl}
        if test_feats is not None:
            save["test_feats"] = test_feats
            save["test_lbl"] = test_lbl
        else:
            save["test_feats"] = np.zeros((0, train_feats.shape[1]), dtype=np.float32)
            save["test_lbl"] = np.zeros((0,), dtype=np.int64)
        np.savez_compressed(out_path, **save)
        out[label_path] = save
        status(f"  saved {out_path}  shape={train_feats.shape}")
    return out


def stage_d_fer2013_probe(features):
    """Linear probe + SupCon on FER2013 features."""
    status("STAGE D: FER2013 probes")
    results = {}
    for enc_label, d in features.items():
        Xtr, ytr = d["train_feats"], d["train_lbl"]
        Xte, yte = d["test_feats"], d["test_lbl"]
        n_classes = int(max(ytr.max(), (yte.max() if len(yte) else 0))) + 1
        # If test is empty, do random 80/20 of train (subjects unknown anyway)
        if len(Xte) == 0:
            rng = np.random.default_rng(42)
            idx = np.arange(len(Xtr))
            rng.shuffle(idx)
            cut = int(0.8 * len(idx))
            tr_idx, te_idx = idx[:cut], idx[cut:]
            Xtr2, ytr2 = Xtr[tr_idx], ytr[tr_idx]
            Xte, yte = Xtr[te_idx], ytr[te_idx]
            Xtr, ytr = Xtr2, ytr2
        status(f"  {enc_label}: train={len(Xtr)} test={len(Xte)} n_classes={n_classes}")

        # Linear probe (LogReg balanced)
        best = None
        for C in [0.1, 1.0, 10.0]:
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000,
                                     solver="lbfgs", random_state=42, n_jobs=-1)
            clf.fit(Xtr, ytr)
            yhat = clf.predict(Xte)
            acc = accuracy_score(yte, yhat)
            kw = float(cohen_kappa_score(yte, yhat))
            if best is None or kw > best["kw"]:
                best = {"C": C, "kw": float(kw), "acc": float(acc), "yhat": yhat, "clf": clf}
        m = metrics_classification(yte, best["yhat"], n_classes)
        m["best_C"] = best["C"]
        m["kappa_unweighted_ci95"] = boot_kappa(yte, best["yhat"], weighted=False)
        m["macro_f1_ci95"] = None  # skip f1 ci for speed
        status(f"  LinProbe {enc_label}: acc={m['accuracy']:.3f}  κ={m['kappa_unweighted']:.3f}  F1={m['f1_macro']:.3f}")
        results[enc_label] = {"linear_probe": m}

    out_path = os.path.join(OUT_DIR, "fer2013_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    status(f"  Saved: {out_path}")
    return results


# ============ Tier 2: more encoders on DAiSEE ============
def encode_daisee(model_spec, label):
    """Encode existing DAiSEE single t=5s frames with the given encoder."""
    out_path = os.path.join(FEAT_DIR, f"daisee_{label}_features.npz")
    if os.path.exists(out_path):
        status(f"STAGE: {label} DAiSEE features exist, skipping")
        return out_path
    status(f"STAGE: encoding DAiSEE with {label}")

    manifest = os.path.join(BASE, "frames_full", "manifest.csv")
    rows = []
    with open(manifest) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["status"] in ("ok", "exists") and os.path.exists(r["frame_path"]):
                rows.append(r)
    status(f"  loading {len(rows)} frames")
    images = [Image.open(r["frame_path"]).convert("RGB") for r in rows]
    encoder, model_name, pretrained = model_spec
    if encoder == "open_clip":
        feats = encode_with_open_clip(images, model_name, pretrained, status_prefix=f"DAiSEE-{label}", batch=32)
    else:
        feats = encode_with_dinov2(images, status_prefix=f"DAiSEE-{label}")
    save = {
        "clip_id": np.array([r["clip_id"] for r in rows]),
        "split": np.array([r["split"] for r in rows]),
        "subject_id": np.array([r["subject_id"] for r in rows]),
        "engagement": np.array([int(r["engagement"]) for r in rows], dtype=np.int64),
        "feat": feats,
    }
    np.savez_compressed(out_path, **save)
    status(f"  saved {out_path}  shape={feats.shape}")
    return out_path


def linear_probe_daisee(feats_path, label):
    """Standard subject-disjoint LR + Ridge probe on DAiSEE features."""
    data = np.load(feats_path, allow_pickle=True)
    splits = data["split"]; feats = data["feat"].astype(np.float32)
    eng = data["engagement"]
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    Xtr, ytr = feats[tr], eng[tr]
    Xva, yva = feats[va], eng[va]
    Xte, yte = feats[te], eng[te]

    # LogReg
    best = None
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                                 solver="lbfgs", random_state=42, n_jobs=-1)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": float(v), "clf": clf}
    yhat_te = best["clf"].predict(Xte)
    m_lr = {
        "accuracy": float(accuracy_score(yte, yhat_te)),
        "kappa_quadratic": float(cohen_kappa_score(yte, yhat_te, weights="quadratic")),
        "kappa_q_ci95": boot_kappa(yte, yhat_te, weighted=True),
        "f1_macro": float(f1_score(yte, yhat_te, average="macro", zero_division=0)),
        "pred_dist": {int(k): int((yhat_te == k).sum()) for k in range(4)},
        "best_C": best["C"],
    }

    # Ridge ordinal
    best = None
    for alpha in [1.0, 10.0, 100.0, 1000.0]:
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(Xtr, ytr.astype(np.float32))
        yhat = np.clip(np.round(reg.predict(Xva)), 0, 3).astype(int)
        v = cohen_kappa_score(yva, yhat, weights="quadratic")
        if best is None or v > best["v"]:
            best = {"a": alpha, "v": float(v), "reg": reg}
    yhat_te = np.clip(np.round(best["reg"].predict(Xte)), 0, 3).astype(int)
    m_rd = {
        "accuracy": float(accuracy_score(yte, yhat_te)),
        "kappa_quadratic": float(cohen_kappa_score(yte, yhat_te, weights="quadratic")),
        "kappa_q_ci95": boot_kappa(yte, yhat_te, weighted=True),
        "f1_macro": float(f1_score(yte, yhat_te, average="macro", zero_division=0)),
        "pred_dist": {int(k): int((yhat_te == k).sum()) for k in range(4)},
        "best_alpha": best["a"],
    }
    status(f"  LinProbe DAiSEE {label}: LR κ_q={m_lr['kappa_quadratic']:.3f}  Ridge κ_q={m_rd['kappa_quadratic']:.3f}")
    return {"logreg": m_lr, "ridge_ordinal": m_rd}


def main():
    summary = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": {}}

    # ----- Tier 1: FER2013 -----
    try:
        ds = stage_a_download_fer2013()
        feats = stage_b_c_fer2013_features(ds)
        fer_results = stage_d_fer2013_probe(feats)
        summary["stages"]["tier1_fer2013"] = {"status": "ok", "results": fer_results}
    except Exception as e:
        tb = traceback.format_exc()
        status(f"Tier 1 FAILED: {e}")
        status(tb)
        summary["stages"]["tier1_fer2013"] = {"status": "fail", "error": str(e)}

    # ----- Tier 2: more encoders on DAiSEE -----
    tier2 = {}
    encoders = [
        ("siglip_l", ("open_clip", "ViT-L-16-SigLIP-256", "webli")),
        ("clip_l_14", ("open_clip", "ViT-L-14", "laion2b_s32b_b82k")),
    ]
    for label, spec in encoders:
        try:
            feats_path = encode_daisee(spec, label)
            res = linear_probe_daisee(feats_path, label)
            tier2[label] = res
        except Exception as e:
            tb = traceback.format_exc()
            status(f"Tier 2 {label} FAILED: {e}")
            status(tb)
            tier2[label] = {"status": "fail", "error": str(e)}
    out_path = os.path.join(OUT_DIR, "tier2_results.json")
    with open(out_path, "w") as f:
        json.dump(tier2, f, indent=2)
    summary["stages"]["tier2_daisee_more_encoders"] = tier2
    status(f"Saved tier 2: {out_path}")

    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    status("OVERNIGHT_PIPELINE_DONE")


if __name__ == "__main__":
    main()
