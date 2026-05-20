"""
Use a face-emotion-pretrained ViT (dima806/facial_emotions_image_detection)
as the encoder for DAiSEE engagement.

Hypothesis: an encoder fine-tuned on facial emotions encodes affect more directly
than generic CLIP/SigLIP. Might break the 0.21 ceiling.

Output:
  features/daisee_emotion_vit_features.npz
  results/sota/emotion_encoder.json
"""
import os, csv, json, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(BASE, "frames_full", "manifest.csv")
FEAT_OUT = os.path.join(BASE, "features", "daisee_emotion_vit_features.npz")
RES = os.path.join(BASE, "results", "sota", "emotion_encoder.json")
LOG = os.path.join(BASE, "results", "sota", "emotion_encoder_status.txt")
os.makedirs(os.path.dirname(RES), exist_ok=True)
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


def fit_lr(Xtr, ytr, Xva, yva):
    best = None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": float(v), "clf": clf}
    return best["clf"], best["v"], best["C"]


def encode():
    if os.path.exists(FEAT_OUT):
        log(f"features exist at {FEAT_OUT}, skipping encoding")
        return
    rows = []
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "exists") and os.path.exists(r["frame_path"]):
                rows.append(r)
    log(f"Loading emotion ViT and encoding {len(rows)} frames...")

    proc = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
    model = AutoModel.from_pretrained("dima806/facial_emotions_image_detection").to(DEVICE).eval()

    # Probe one image to get feature dim
    img = Image.open(rows[0]["frame_path"]).convert("RGB")
    inp = proc(img, return_tensors="pt")
    with torch.no_grad():
        out = model(**{k: v.to(DEVICE) for k, v in inp.items()})
    feat_dim = out.last_hidden_state.shape[-1]
    log(f"  feat_dim={feat_dim}")

    BATCH = 32
    feats = np.zeros((len(rows), feat_dim), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            imgs = [Image.open(r["frame_path"]).convert("RGB") for r in chunk]
            inputs = proc(imgs, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            out = model(**inputs)
            # Use the [CLS] token (first hidden state position)
            cls_feat = out.last_hidden_state[:, 0]  # (B, D)
            # Or pooler_output if available
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                # pooler may be untrained (we saw MISSING above); use CLS instead
                pass
            feats[i:i + len(chunk)] = cls_feat.cpu().numpy()
            if (i // BATCH) % 20 == 0:
                dt = time.time() - t0
                rate = (i + len(chunk)) / dt
                log(f"  {i + len(chunk)}/{len(rows)}  rate={rate:.1f} fps  ETA={(len(rows)-i-len(chunk))/rate:.0f}s")

    np.savez_compressed(
        FEAT_OUT,
        clip_id=np.array([r["clip_id"] for r in rows]),
        split=np.array([r["split"] for r in rows]),
        subject_id=np.array([r["subject_id"] for r in rows]),
        engagement=np.array([int(r["engagement"]) for r in rows], dtype=np.int64),
        feat=feats,
    )
    log(f"Saved features: {FEAT_OUT}  shape={feats.shape}")


def probe():
    log("Loading features for probe...")
    d = np.load(FEAT_OUT, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    splits = d["split"]
    eng = d["engagement"].astype(np.int64)
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    Xtr, ytr = X[tr], eng[tr]
    Xva, yva = X[va], eng[va]
    Xte, yte = X[te], eng[te]

    out = {}

    # 4-class LR baseline
    log("\n=== 4-class LR (engagement, ordinal) ===")
    clf, val_v, C = fit_lr(Xtr, ytr, Xva, yva)
    yhat = clf.predict(Xte)
    m = metrics(yte, yhat)
    out["solo_4class_LR"] = {**m, "C": C, "val_kq": val_v}
    log(f"  κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  C={C} val={val_v:.3f}")

    # Ridge ordinal
    log("\n=== 4-class Ridge ordinal ===")
    best = None
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(Xtr, ytr.astype(np.float32))
        yhat_va = np.clip(np.round(reg.predict(Xva)), 0, 3).astype(int)
        v = cohen_kappa_score(yva, yhat_va, weights="quadratic")
        if best is None or v > best["v"]:
            best = {"a": alpha, "v": v, "reg": reg}
    yhat = np.clip(np.round(best["reg"].predict(Xte)), 0, 3).astype(int)
    m = metrics(yte, yhat)
    out["solo_Ridge"] = {**m, "alpha": best["a"]}
    log(f"  κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}")

    # Bagged
    log("\n=== Bagged emotion encoder (5 seeds × K=30) ===")
    all_p_te = []
    OUTER_SEEDS = [0, 7, 42, 2025, 1024]
    K_BAGS = 30
    for outer_seed in OUTER_SEEDS:
        rng = np.random.default_rng(outer_seed)
        seed_probs = []
        for k in range(K_BAGS):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            clf_b, _, _ = fit_lr(Xtr[idx], ytr[idx], Xva, yva)
            seed_probs.append(clf_b.predict_proba(Xte))
        seed_mean = np.stack(seed_probs).mean(axis=0)
        all_p_te.append(seed_mean)
        kq_seed = float(cohen_kappa_score(yte, seed_mean.argmax(axis=1), weights="quadratic"))
        log(f"  seed {outer_seed}: κ_q={kq_seed:.3f}")
    mega = np.stack(all_p_te).mean(axis=0)
    yhat = mega.argmax(axis=1)
    m = metrics(yte, yhat)
    out["bagged_5seed"] = m
    log(f"  Mega: κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}")

    with open(RES, "w") as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {RES}")
    log("EMOTION_ENC_DONE")


def main():
    open(LOG, "w").close()
    encode()
    probe()


if __name__ == "__main__":
    main()
