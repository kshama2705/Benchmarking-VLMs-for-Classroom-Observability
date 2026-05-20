"""PPEP on TIPSv2-B14 at 448×448 input (32×32 patch grid — the fine-grain version
TIPS is uniquely positioned to deliver).

Forces the 448 preprocess path. Output: features/tipsv2_b14_448_patch_face_features.npz
"""
import os, csv, json, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel
from torchvision import transforms
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(BASE, "frames_full", "manifest.csv")
BBOX_JSON = os.path.join(BASE, "features", "face_bboxes.json")
FEAT_OUT = os.path.join(BASE, "features", "tipsv2_b14_448_patch_face_features.npz")
RES_OUT = os.path.join(BASE, "results", "ppep", "ppep_results_tipsv2_448.json")
LOG = os.path.join(BASE, "results", "ppep", "tips_448_status.txt")
os.makedirs(os.path.dirname(RES_OUT), exist_ok=True)
RNG = np.random.default_rng(42)
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))

GRID = 32  # TIPSv2-B14 at 448 input -> 32x32 grid = 1024 patches


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


def extract():
    if os.path.exists(FEAT_OUT):
        log("exists, skip")
        return

    bboxes = json.load(open(BBOX_JSON))
    rows = []
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "exists") and os.path.exists(r["frame_path"]):
                rows.append(r)
    log(f"Encoding {len(rows)} frames with TIPSv2-B14 at 448→32×32 grid...")

    model = AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True)
    model = model.to(DEVICE).eval()

    preprocess = transforms.Compose([
        transforms.Resize(448),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # Sanity probe
    sample = preprocess(Image.open(rows[0]["frame_path"]).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model.encode_image(sample)
    log(f"  cls={tuple(out.cls_token.shape)} patches={tuple(out.patch_tokens.shape)}  grid={int(np.sqrt(out.patch_tokens.shape[1]))}")
    if int(np.sqrt(out.patch_tokens.shape[1])) != GRID:
        log(f"  WARN: expected {GRID}x{GRID} but got {int(np.sqrt(out.patch_tokens.shape[1]))} grid")

    BATCH = 4  # 448 input is memory-heavy
    feat_dim = out.patch_tokens.shape[-1]
    face_feats = np.zeros((len(rows), feat_dim), dtype=np.float32)
    bg_feats = np.zeros((len(rows), feat_dim), dtype=np.float32)
    cls_feats = np.zeros((len(rows), feat_dim), dtype=np.float32)
    mean_feats = np.zeros((len(rows), feat_dim), dtype=np.float32)
    face_counts = np.zeros(len(rows), dtype=np.int32)

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            imgs = torch.stack([preprocess(Image.open(r["frame_path"]).convert("RGB"))
                                for r in chunk]).to(DEVICE)
            out = model.encode_image(imgs)
            patches = out.patch_tokens
            cls = out.cls_token.squeeze(1) if out.cls_token.dim() == 3 else out.cls_token
            masks = np.stack([vectorized_face_mask(bboxes.get(r["clip_id"], {}).get("bbox"),
                                                    grid=GRID) for r in chunk])
            masks_t = torch.from_numpy(masks).to(DEVICE).float()
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
            face_counts[i:i + len(chunk)] = masks.sum(axis=1)
            if (i // BATCH) % 50 == 0:
                dt = time.time() - t0
                log(f"  TIPS-448 {i + len(chunk)}/{len(rows)}  elapsed={dt:.0f}s")

    np.savez_compressed(
        FEAT_OUT,
        clip_id=np.array([r["clip_id"] for r in rows]),
        split=np.array([r["split"] for r in rows]),
        subject_id=np.array([r["subject_id"] for r in rows]),
        engagement=np.array([int(r["engagement"]) for r in rows], dtype=np.int64),
        face_feat=face_feats, bg_feat=bg_feats, cls_feat=cls_feats, mean_feat=mean_feats,
        face_patch_counts=face_counts,
    )
    log(f"Saved: {FEAT_OUT}")


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
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": v, "clf": clf}
    yhat = best["clf"].predict(Xte)
    m_lr = {"kappa_q": float(cohen_kappa_score(yte, yhat, weights="quadratic")),
            "kappa_q_ci": boot_kq(yte, yhat),
            "accuracy": float(accuracy_score(yte, yhat)),
            "best_C": best["C"],
            "pred_dist": {int(k): int((yhat == k).sum()) for k in range(4)}}
    best = None
    for alpha in [1.0, 10.0, 100.0, 1000.0]:
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(Xtr, ytr.astype(np.float32))
        v = cohen_kappa_score(yva, np.clip(np.round(reg.predict(Xva)), 0, 3).astype(int), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"a": alpha, "v": v, "reg": reg}
    yhat = np.clip(np.round(best["reg"].predict(Xte)), 0, 3).astype(int)
    m_rd = {"kappa_q": float(cohen_kappa_score(yte, yhat, weights="quadratic")),
            "kappa_q_ci": boot_kq(yte, yhat),
            "accuracy": float(accuracy_score(yte, yhat)),
            "best_alpha": best["a"],
            "pred_dist": {int(k): int((yhat == k).sum()) for k in range(4)}}
    log(f"  {label:10} LR    κ_q={m_lr['kappa_q']:.3f} {m_lr['kappa_q_ci']}")
    log(f"  {label:10} Ridge κ_q={m_rd['kappa_q']:.3f} {m_rd['kappa_q_ci']}")
    return {"logreg": m_lr, "ridge_ordinal": m_rd}


def main():
    open(LOG, "w").close()
    extract()
    d = np.load(FEAT_OUT, allow_pickle=True)
    splits = d["split"]; eng = d["engagement"]
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    out = {}
    for kind in ["face_feat", "bg_feat", "cls_feat", "mean_feat"]:
        Xt = d[kind].astype(np.float32)
        log(f"--- TIPS-448 | {kind} ---")
        out[kind] = probe(Xt[tr], eng[tr], Xt[va], eng[va], Xt[te], eng[te], kind)
    with open(RES_OUT, "w") as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {RES_OUT}")
    log("TIPS_448_DONE")


if __name__ == "__main__":
    main()
