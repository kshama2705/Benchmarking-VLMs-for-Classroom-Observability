"""
PPEP — Patch-Pooled Engagement Probe (DINOv2 ViT-B/14, vectorized).
Writes progress to a log file directly to avoid pipe-buffering issues.

Output:
  features/dinov2_patch_face_features.npz
  results/ppep/ppep_results.json
"""
import os, csv, json, time
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_squared_error

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(BASE, "frames_full", "manifest.csv")
BBOX_JSON = os.path.join(BASE, "features", "face_bboxes.json")
FEAT_OUT = os.path.join(BASE, "features", "dinov2_patch_face_features.npz")
RES_OUT = os.path.join(BASE, "results", "ppep", "ppep_results.json")
LOG = os.path.join(BASE, "results", "ppep", "ppep_status.txt")
os.makedirs(os.path.dirname(RES_OUT), exist_ok=True)
RNG = np.random.default_rng(42)
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))

INPUT_SIZE = 224
PATCH_SIZE = 14
GRID = INPUT_SIZE // PATCH_SIZE  # 16


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def vectorized_face_mask(bbox, grid, face_thresh=0.3):
    if bbox is None:
        return np.zeros(grid * grid, dtype=np.int64)
    x, y, w, h = bbox
    j_idx, i_idx = np.meshgrid(np.arange(grid), np.arange(grid))
    px0, py0 = x * grid, y * grid
    px1, py1 = (x + w) * grid, (y + h) * grid
    ix0 = np.maximum(j_idx, px0); iy0 = np.maximum(i_idx, py0)
    ix1 = np.minimum(j_idx + 1, px1); iy1 = np.minimum(i_idx + 1, py1)
    overlap = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    return (overlap >= face_thresh).astype(np.int64).reshape(-1)


def load_dinov2():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                           source="github", trust_repo=True).to(DEVICE).eval()
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return model, preprocess


def extract_features():
    if os.path.exists(FEAT_OUT):
        log(f"features exist at {FEAT_OUT}, skipping extraction")
        return

    bboxes = json.load(open(BBOX_JSON))
    rows = []
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "exists") and os.path.exists(r["frame_path"]):
                rows.append(r)
    log(f"Encoding {len(rows)} frames with DINOv2 (vectorized face/bg pooling)")

    model, preprocess = load_dinov2()
    BATCH = 32

    face_feats = np.zeros((len(rows), 768), dtype=np.float32)
    bg_feats = np.zeros((len(rows), 768), dtype=np.float32)
    cls_feats = np.zeros((len(rows), 768), dtype=np.float32)
    mean_feats = np.zeros((len(rows), 768), dtype=np.float32)
    face_counts = np.zeros(len(rows), dtype=np.int32)

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            imgs = torch.stack([preprocess(Image.open(r["frame_path"]).convert("RGB"))
                                for r in chunk]).to(DEVICE)
            outs = model.forward_features(imgs)
            cls = outs["x_norm_clstoken"]
            patches = outs["x_norm_patchtokens"]  # (B, 256, 768)
            # Vectorized face masks for whole batch
            masks_np = np.stack([vectorized_face_mask(bboxes.get(r["clip_id"], {}).get("bbox"),
                                                      grid=GRID) for r in chunk])  # (B, 256)
            masks_t = torch.from_numpy(masks_np).to(DEVICE).float()
            face_sum = masks_t.sum(dim=1, keepdim=True).clamp(min=1)
            bg_t = 1.0 - masks_t
            bg_sum = bg_t.sum(dim=1, keepdim=True).clamp(min=1)
            face_emb = (patches * masks_t.unsqueeze(-1)).sum(dim=1) / face_sum
            bg_emb = (patches * bg_t.unsqueeze(-1)).sum(dim=1) / bg_sum
            mean_emb = patches.mean(dim=1)
            face_feats[i:i + len(chunk)] = face_emb.cpu().numpy()
            bg_feats[i:i + len(chunk)] = bg_emb.cpu().numpy()
            cls_feats[i:i + len(chunk)] = cls.cpu().numpy()
            mean_feats[i:i + len(chunk)] = mean_emb.cpu().numpy()
            face_counts[i:i + len(chunk)] = masks_np.sum(axis=1)
            if (i // BATCH) % 10 == 0:
                dt = time.time() - t0
                log(f"  {i + len(chunk)}/{len(rows)}  elapsed={dt:.0f}s")

    np.savez_compressed(
        FEAT_OUT,
        clip_id=np.array([r["clip_id"] for r in rows]),
        split=np.array([r["split"] for r in rows]),
        subject_id=np.array([r["subject_id"] for r in rows]),
        engagement=np.array([int(r["engagement"]) for r in rows], dtype=np.int64),
        face_feat=face_feats, bg_feat=bg_feats, cls_feat=cls_feats, mean_feat=mean_feats,
        face_patch_counts=face_counts,
    )
    log(f"Saved features: {FEAT_OUT}")
    log(f"  face_patch_counts: median={int(np.median(face_counts))}  "
        f"zero_count_frames={(face_counts == 0).sum()}")
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def boot_kq(yt, yp, n=1000):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def probe(Xtr, ytr, Xva, yva, Xte, yte, label):
    best = None
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                                 solver="lbfgs", random_state=42, n_jobs=-1)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": v, "clf": clf}
    yhat_lr = best["clf"].predict(Xte)
    m_lr = {"best_C": best["C"],
            "kappa_q": float(cohen_kappa_score(yte, yhat_lr, weights="quadratic")),
            "kappa_q_ci": boot_kq(yte, yhat_lr),
            "accuracy": float(accuracy_score(yte, yhat_lr)),
            "f1_macro": float(f1_score(yte, yhat_lr, average="macro", zero_division=0)),
            "pred_dist": {int(k): int((yhat_lr == k).sum()) for k in range(4)}}
    best = None
    for alpha in [1.0, 10.0, 100.0, 1000.0]:
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(Xtr, ytr.astype(np.float32))
        v = cohen_kappa_score(yva, np.clip(np.round(reg.predict(Xva)), 0, 3).astype(int), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"a": alpha, "v": v, "reg": reg}
    yhat_rd = np.clip(np.round(best["reg"].predict(Xte)), 0, 3).astype(int)
    m_rd = {"best_alpha": best["a"],
            "kappa_q": float(cohen_kappa_score(yte, yhat_rd, weights="quadratic")),
            "kappa_q_ci": boot_kq(yte, yhat_rd),
            "accuracy": float(accuracy_score(yte, yhat_rd)),
            "f1_macro": float(f1_score(yte, yhat_rd, average="macro", zero_division=0)),
            "pred_dist": {int(k): int((yhat_rd == k).sum()) for k in range(4)}}
    log(f"  {label:10} LR    κ_q={m_lr['kappa_q']:.3f} {m_lr['kappa_q_ci']}  acc={m_lr['accuracy']:.3f}")
    log(f"  {label:10} Ridge κ_q={m_rd['kappa_q']:.3f} {m_rd['kappa_q_ci']}  acc={m_rd['accuracy']:.3f}")
    return {"logreg": m_lr, "ridge_ordinal": m_rd}


def probes():
    d = np.load(FEAT_OUT, allow_pickle=True)
    splits = d["split"]; eng = d["engagement"]
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    log(f"Probe splits: train={tr.sum()} val={va.sum()} test={te.sum()}")

    out = {}
    for kind in ["face_feat", "bg_feat", "cls_feat", "mean_feat"]:
        Xt = d[kind].astype(np.float32)
        log(f"--- {kind} ---")
        out[kind] = probe(Xt[tr], eng[tr], Xt[va], eng[va], Xt[te], eng[te], kind)

    with open(RES_OUT, "w") as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {RES_OUT}")


def main():
    open(LOG, "w").close()  # truncate
    if not os.path.exists(BBOX_JSON):
        log(f"ERROR: face bboxes missing — run 33 first")
        return
    extract_features()
    probes()
    log("PPEP_DONE")


if __name__ == "__main__":
    main()
