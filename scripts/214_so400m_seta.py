"""
Option C: SigLIP-SO400M encoding + SETA recipe.

Encodes all 8,571 clips with the larger SigLIP-SO400M-patch14-384 model
(400M params, 384px input vs SigLIP-L's 256px), then runs the SETA recipe
(bagged LR + threshold tuning). Larger model = better generalization.

Caches features to features/daisee_siglip_so400m_features.npz.
"""
import os, json, time, csv
import numpy as np

BASE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MANIFEST = os.path.join(BASE, "frames_full/manifest.csv")
FEAT_OUT = os.path.join(BASE, "features/daisee_siglip_so400m_features.npz")
OUT      = os.path.join(BASE, "results/sota/so400m_seta.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

import torch
device = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))
log(f"Device: {device}")

# ---- Step 1: Encode if cache missing ----
if not os.path.exists(FEAT_OUT):
    from transformers import AutoModel, SiglipImageProcessor
    from PIL import Image

    MODEL_ID = "google/siglip-so400m-patch14-384"
    log(f"Loading {MODEL_ID} ...")
    processor = SiglipImageProcessor.from_pretrained(MODEL_ID)
    model     = AutoModel.from_pretrained(MODEL_ID).vision_model.to(device).eval()
    log(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))
    log(f"  {len(rows)} clips to encode")

    FRAMES_ROOT = "/Users/amangoyal/Downloads/frames_full"
    BATCH = 16
    feats = []; clip_ids = []; splits = []; subjects = []; engagements = []
    buf_imgs = []; buf_meta = []

    def flush(buf_imgs, buf_meta):
        if not buf_imgs: return [], []
        inputs = processor(images=buf_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        f = out.pooler_output.cpu().float().numpy()  # (B, 1152)
        return f, buf_meta

    t0 = time.time()
    for i, r in enumerate(rows):
        cid   = r["clip_id"].replace(".avi","").replace(".mp4","")
        fpath = os.path.join(FRAMES_ROOT, r["split"], cid + ".jpg")
        if not os.path.exists(fpath):
            fpath_fallback = r.get("frame_path","")
            if os.path.exists(fpath_fallback):
                fpath = fpath_fallback
            else:
                continue
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception:
            continue
        buf_imgs.append(img)
        buf_meta.append((cid, r["split"], r["subject_id"], int(r["engagement"])))
        if len(buf_imgs) == BATCH:
            f, m = flush(buf_imgs, buf_meta)
            feats.extend(f); clip_ids.extend([x[0] for x in m])
            splits.extend([x[1] for x in m]); subjects.extend([x[2] for x in m])
            engagements.extend([x[3] for x in m])
            buf_imgs = []; buf_meta = []
        if (i+1) % 500 == 0:
            rate = (i+1)/(time.time()-t0)
            eta  = (len(rows)-i-1)/rate/60
            log(f"  {i+1}/{len(rows)}  {rate:.1f} clips/s  ETA {eta:.1f} min")
        if device.type == "mps" and (i+1) % 200 == 0:
            torch.mps.empty_cache()

    if buf_imgs:
        f, m = flush(buf_imgs, buf_meta)
        feats.extend(f); clip_ids.extend([x[0] for x in m])
        splits.extend([x[1] for x in m]); subjects.extend([x[2] for x in m])
        engagements.extend([x[3] for x in m])

    feat_arr = np.array(feats, dtype=np.float32)
    np.savez_compressed(FEAT_OUT,
        feat=feat_arr, clip_id=np.array(clip_ids),
        split=np.array(splits), subject_id=np.array(subjects),
        engagement=np.array(engagements, dtype=np.int64))
    log(f"Saved features: {feat_arr.shape}  →  {FEAT_OUT}")
else:
    log(f"Cache hit: {FEAT_OUT}")

# ---- Step 2: SETA recipe ----
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, accuracy_score

log("Running SETA recipe on SO400M features...")
F = np.load(FEAT_OUT, allow_pickle=True)
feat    = F["feat"].astype(np.float32)
split   = F["split"]
subject = F["subject_id"]
y       = F["engagement"].astype(np.int64)

tr_mask = (split == "Train"); va_mask = (split == "Validation"); te_mask = (split == "Test")
log(f"  splits: tr={tr_mask.sum()} va={va_mask.sum()} te={te_mask.sum()}")

mu = feat[tr_mask].mean(axis=0); sd = feat[tr_mask].std(axis=0) + 1e-6
feat_z = (feat - mu) / sd
X_tr = feat_z[tr_mask]; y_tr = y[tr_mask]
X_va = feat_z[va_mask]; y_va = y[va_mask]
X_te = feat_z[te_mask]; y_te = y[te_mask]

N_BAGS = 20; SEEDS = list(range(5))

def tune_thresholds(e, y_true, step=0.05):
    grid = np.arange(0.0, 3.01, step)
    best = None
    for t1 in grid:
        for t2 in grid[grid > t1]:
            for t3 in grid[grid > t2]:
                yp = np.zeros_like(e, dtype=int)
                yp[e > t1] = 1; yp[e > t2] = 2; yp[e > t3] = 3
                v = cohen_kappa_score(y_true, yp, weights="quadratic")
                if best is None or v > best["v"]:
                    best = {"t": (float(t1),float(t2),float(t3)), "v": float(v)}
    return best

def apply_thr(e, t):
    yp = np.zeros_like(e, dtype=int)
    yp[e > t[0]] = 1; yp[e > t[1]] = 2; yp[e > t[2]] = 3
    return yp

def boot_ci(yt, yp, n=1000, seed=42):
    rng = np.random.default_rng(seed); nt = len(yt); out = []
    for _ in range(n):
        idx = rng.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]

log(f"Bagging {N_BAGS} bags × {len(SEEDS)} seeds...")
all_p_va = []; all_p_te = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    for bag in range(N_BAGS):
        idx = rng.integers(0, len(X_tr), size=len(X_tr))
        clf = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial",
                                  solver="lbfgs", random_state=seed*100+bag)
        clf.fit(X_tr[idx], y_tr[idx])
        all_p_va.append(clf.predict_proba(X_va))
        all_p_te.append(clf.predict_proba(X_te))
    log(f"  seed {seed} done")

e_va = np.mean(all_p_va, axis=0) @ np.array([0,1,2,3])
e_te = np.mean(all_p_te, axis=0) @ np.array([0,1,2,3])
bt   = tune_thresholds(e_va, y_va)
yp   = apply_thr(e_te, bt["t"])
kq   = float(cohen_kappa_score(y_te, yp, weights="quadratic"))
acc  = float(accuracy_score(y_te, yp))
ci   = boot_ci(y_te, yp)

log(f"\nRESULT: κ_q={kq:.4f}  CI={ci}  acc={acc:.3f}  t={bt['t']}")
result = {"method": "so400m_seta", "test_kq": kq, "test_acc": acc, "ci": ci,
          "thresholds": bt["t"], "val_kq": bt["v"]}
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
log(f"Saved: {OUT}")
