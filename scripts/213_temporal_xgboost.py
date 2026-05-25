"""
Option B: Temporal dynamics + SigLIP-L + XGBoost bagged ensemble.

Feature vector (1137-d):
  SigLIP-L frozen (1024) +
  blendshapes_std  (52)  — variability of facial expressions across clip frames
  blendshapes_delta(52)  — mean frame-to-frame change (motion dynamics)
  head_pose_std    (3)   — head movement variability
  eye_gaze_std     (6)   — gaze variability

XGBoost gradient boosting (non-linear, handles noisy features well) with
the same bagged + threshold-tuning SETA recipe. The std/delta features
capture engagement-specific motion signals that pure appearance cannot.
"""
import os, json, time
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(BASE, "results/sota/temporal_xgboost.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

from sklearn.ensemble import HistGradientBoostingClassifier
log("Using sklearn HistGradientBoostingClassifier (XGBoost-equivalent, no libomp needed)")

log("Loading SigLIP-L features...")
F1 = np.load(os.path.join(BASE, "features/daisee_siglip_l_features.npz"), allow_pickle=True)
feat_vis = F1["feat"].astype(np.float32)
split    = F1["split"]
subject  = F1["subject_id"]
y        = F1["engagement"].astype(np.int64)
clip_ids = F1["clip_id"]

log("Loading temporal face dynamics...")
F2 = np.load(os.path.join(BASE, "features/daisee_face_signals_temporal.npz"), allow_pickle=True)
# Align by clip_id
cid2idx = {cid: i for i, cid in enumerate(F2["clip_id"])}
order   = np.array([cid2idx.get(c.replace(".avi","").replace(".mp4",""), -1) for c in clip_ids])
valid   = order >= 0
log(f"  aligned {valid.sum()}/{len(clip_ids)} clips")

N = len(clip_ids)
bs_std   = np.zeros((N, 52), dtype=np.float32)
bs_delta = np.zeros((N, 52), dtype=np.float32)
hp_std   = np.zeros((N, 3),  dtype=np.float32)
gz_std   = np.zeros((N, 6),  dtype=np.float32)

bs_std[valid]   = F2["blendshapes_std"][order[valid]]
bs_delta[valid] = F2["blendshapes_delta"][order[valid]]
hp_std[valid]   = F2["head_pose_std"][order[valid]]
gz_std[valid]   = F2["eye_gaze_std"][order[valid]]

# Concatenate: SigLIP-L (1024) + temporal dynamics (113)
X = np.concatenate([feat_vis, bs_std, bs_delta, hp_std, gz_std], axis=1).astype(np.float32)
log(f"  combined feature dim: {X.shape[1]}")

# Standardize on train
tr_mask = (split == "Train")
va_mask = (split == "Validation")
te_mask = (split == "Test")
mu = X[tr_mask].mean(axis=0); sd = X[tr_mask].std(axis=0) + 1e-6
X_z = (X - mu) / sd

X_tr = X_z[tr_mask]; y_tr = y[tr_mask]
X_va = X_z[va_mask]; y_va = y[va_mask]
X_te = X_z[te_mask]; y_te = y[te_mask]
log(f"  splits: tr={tr_mask.sum()} va={va_mask.sum()} te={te_mask.sum()}")

# Class weights
cnt = np.bincount(y_tr, minlength=4).astype(float)
w   = (1.0 / np.maximum(cnt, 1))[y_tr]
w   = w / w.mean()

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
                    best = {"t": (float(t1), float(t2), float(t3)), "v": float(v)}
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

N_BAGS = 15
SEEDS  = [0, 42, 2025]
log(f"Bagging {N_BAGS} XGBoost bags × {len(SEEDS)} seeds...")

all_e_va = []; all_e_te = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    for bag in range(N_BAGS):
        idx = rng.integers(0, len(X_tr), size=len(X_tr))
        Xb, yb, wb = X_tr[idx], y_tr[idx], w[idx]
        clf = HistGradientBoostingClassifier(
            max_iter=300, max_depth=5, learning_rate=0.05,
            min_samples_leaf=20, l2_regularization=0.1,
            random_state=seed * 100 + bag, verbose=0
        )
        clf.fit(Xb, yb, sample_weight=wb)
        p_va = clf.predict_proba(X_va)
        p_te = clf.predict_proba(X_te)
        # ensure 4 classes
        if p_va.shape[1] < 4:
            pad = np.zeros((p_va.shape[0], 4 - p_va.shape[1]))
            p_va = np.hstack([p_va, pad])
            p_te = np.hstack([p_te, pad])
        all_e_va.append(p_va @ np.array([0, 1, 2, 3]))
        all_e_te.append(p_te @ np.array([0, 1, 2, 3]))
    log(f"  seed {seed} done ({(SEEDS.index(seed)+1)*N_BAGS}/{len(SEEDS)*N_BAGS} bags)")

e_va = np.mean(all_e_va, axis=0)
e_te = np.mean(all_e_te, axis=0)

bt = tune_thresholds(e_va, y_va)
yp_te = apply_thr(e_te, bt["t"])
kq  = float(cohen_kappa_score(y_te, yp_te, weights="quadratic"))
acc = float(accuracy_score(y_te, yp_te))
ci  = boot_ci(y_te, yp_te)

log(f"\nRESULT: κ_q={kq:.4f}  CI={ci}  acc={acc:.3f}  t={bt['t']}")

result = {"method": "temporal_xgboost", "test_kq": kq, "test_acc": acc, "ci": ci,
          "thresholds": bt["t"], "val_kq": bt["v"], "feature_dim": int(X.shape[1])}
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
log(f"Saved: {OUT}")
