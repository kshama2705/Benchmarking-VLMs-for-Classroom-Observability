"""
Test-Time Augmentation (TTA) on SigLIP-L for engagement.

Approach:
  1. Train SigLIP-L LR probe on Train (cached features) — already gives κ=0.199.
  2. For each test frame, generate K augmented versions on-the-fly.
  3. Encode each version with SigLIP-L, run probe, get probabilities.
  4. Average probabilities across augmentations → argmax → final prediction.

Augmentations: original, horizontal flip, +/- center crop, slight color jitter.

Output:
  features/daisee_siglip_l_tta.npz (averaged augmented features for test)
  results/sota/tta_siglip_l.json
"""
import os, csv, json, time
import numpy as np
import torch
from PIL import Image
import open_clip
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(BASE, "frames_full", "manifest.csv")
CACHED = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
FEAT_OUT = os.path.join(BASE, "features", "daisee_siglip_l_tta_test.npz")
RES_OUT = os.path.join(BASE, "results", "sota", "tta_siglip_l.json")
LOG = os.path.join(BASE, "results", "sota", "tta_status.txt")
os.makedirs(os.path.dirname(RES_OUT), exist_ok=True)
RNG = np.random.default_rng(42)
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def boot_kq(yt, yp, n=1000):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def metrics(yt, yp):
    return {
        "kappa_q": float(cohen_kappa_score(yt, yp, weights="quadratic")),
        "kappa_q_ci": boot_kq(yt, yp),
        "accuracy": float(accuracy_score(yt, yp)),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "pred_dist": {int(k): int((yp == k).sum()) for k in range(4)},
    }


def fit_probe(Xtr, ytr, Xva, yva):
    best = None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": float(v), "clf": clf}
    return best["clf"], best["v"], best["C"]


def main():
    open(LOG, "w").close()

    # Load cached features for train/val/test
    log("Loading cached SigLIP-L features...")
    d = np.load(CACHED, allow_pickle=True)
    splits = d["split"]; eng = d["engagement"].astype(np.int64)
    feats = d["feat"].astype(np.float32)
    clip_ids = d["clip_id"]
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    log(f"  Train={tr.sum()} Val={va.sum()} Test={te.sum()}")

    # Train probe on cached train features
    clf, val_v, C = fit_probe(feats[tr], eng[tr], feats[va], eng[va])
    log(f"  baseline probe (no TTA): val={val_v:.3f} C={C}")
    yhat_baseline = clf.predict(feats[te])
    m_baseline = metrics(eng[te], yhat_baseline)
    log(f"  baseline test κ_q={m_baseline['kappa_q']:.3f} {m_baseline['kappa_q_ci']}")

    # Get probe probabilities for ORIGINAL test features (no aug)
    p_orig = clf.predict_proba(feats[te])

    # Now generate augmented features for test only
    log("\nEncoding test set with TTA augmentations...")
    # Load test image paths
    rows = []
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "exists") and os.path.exists(r["frame_path"]):
                rows.append(r)
    test_rows = [r for r in rows if r["split"] == "Test"]
    # Ensure ordering matches cached test features
    cache_te_ids = clip_ids[te]
    id_to_path = {r["clip_id"]: r["frame_path"] for r in rows}
    test_paths = [id_to_path[c] for c in cache_te_ids]
    log(f"  {len(test_paths)} test images for TTA")

    # Load model
    log("Loading SigLIP-L...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-16-SigLIP-256", pretrained="webli")
    model = model.to(DEVICE).eval()

    # TTA augmentations (apply after PIL load, before preprocess)
    augmentations = [
        ("hflip", lambda im: im.transpose(Image.FLIP_LEFT_RIGHT)),
        ("crop_center_110", lambda im: im.resize((int(im.width * 1.1), int(im.height * 1.1)), Image.BICUBIC).crop((int(im.width * 0.05), int(im.height * 0.05), int(im.width * 0.05) + im.width, int(im.height * 0.05) + im.height))),
        ("crop_center_90", lambda im: im.crop((int(im.width * 0.05), int(im.height * 0.05), int(im.width * 0.95), int(im.height * 0.95))).resize((im.width, im.height), Image.BICUBIC)),
    ]

    all_aug_feats = []
    aug_names = ["original"] + [name for name, _ in augmentations]

    # First, also re-encode originals so they're aligned (we trust the cache though)
    # Just generate augmented versions
    BATCH = 16
    with torch.no_grad():
        for aug_idx, (name, aug_fn) in enumerate(augmentations):
            log(f"  Aug {aug_idx + 1}/{len(augmentations)}: {name}")
            aug_feats = np.zeros((len(test_paths), feats.shape[1]), dtype=np.float32)
            t0 = time.time()
            for i in range(0, len(test_paths), BATCH):
                batch_paths = test_paths[i:i + BATCH]
                imgs = []
                for p in batch_paths:
                    img = Image.open(p).convert("RGB")
                    img = aug_fn(img)
                    imgs.append(preprocess(img))
                x = torch.stack(imgs).to(DEVICE)
                f = model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
                aug_feats[i:i + len(batch_paths)] = f.cpu().numpy()
                if (i // BATCH) % 30 == 0:
                    dt = time.time() - t0
                    log(f"    {i + len(batch_paths)}/{len(test_paths)} elapsed={dt:.0f}s")
            all_aug_feats.append(aug_feats)

    # Save augmented features
    np.savez_compressed(FEAT_OUT,
                        original=feats[te],
                        **{n: af for n, af in zip([a[0] for a in augmentations], all_aug_feats)})
    log(f"Saved augmented features: {FEAT_OUT}")

    # Get probe probabilities for each augmented version
    log("\nProbing TTA fusion variants...")
    p_augs = [p_orig]
    for name, af in zip([a[0] for a in augmentations], all_aug_feats):
        p = clf.predict_proba(af)
        p_augs.append(p)
        # Solo metric
        yhat_aug = p.argmax(axis=1)
        m_aug = metrics(eng[te], yhat_aug)
        log(f"  {name:25} κ_q={m_aug['kappa_q']:.3f} {m_aug['kappa_q_ci']}")

    # Average all
    p_mean = np.mean(np.stack(p_augs), axis=0)
    yhat_tta = p_mean.argmax(axis=1)
    m_tta = metrics(eng[te], yhat_tta)
    log(f"  TTA mean (4 versions)    κ_q={m_tta['kappa_q']:.3f} {m_tta['kappa_q_ci']}")

    # Best individual variant + original
    out = {
        "baseline_no_TTA": m_baseline,
        "TTA_mean_4": m_tta,
        "augmentations": [a[0] for a in augmentations],
    }
    for aug_idx, name in enumerate([a[0] for a in augmentations]):
        out[f"solo_{name}"] = metrics(eng[te], p_augs[aug_idx + 1].argmax(axis=1))

    # Try also: orig + each aug pair
    for aug_idx, name in enumerate([a[0] for a in augmentations]):
        p_pair = (p_orig + p_augs[aug_idx + 1]) / 2
        yhat = p_pair.argmax(axis=1)
        m = metrics(eng[te], yhat)
        out[f"TTA_pair_orig+{name}"] = m
        log(f"  TTA pair orig+{name:15} κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}")

    with open(RES_OUT, "w") as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {RES_OUT}")
    log("TTA_DONE")


if __name__ == "__main__":
    main()
