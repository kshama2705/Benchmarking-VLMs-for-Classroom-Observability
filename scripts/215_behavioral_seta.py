"""
Option E: Pure behavioral signals + per-subject z-norm + SETA recipe.

No appearance features at all. Only MediaPipe behavioral signals:
  blendshapes_mean/std/delta (156), head_pose_mean/std (6),
  eye_gaze_mean/std (12), landmark_summary_mean (12), detected_count (1)
  Total: 187-d

Per-subject z-norm within each split directly attacks IDEP by removing
between-subject identity variation. SETA recipe for ordinal prediction.
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, accuracy_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(BASE, "results/sota/behavioral_seta.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Loading behavioral features...")
F = np.load(os.path.join(BASE, "features/daisee_face_signals_temporal.npz"), allow_pickle=True)

feat = np.concatenate([
    F["blendshapes_mean"],   # (N, 52)
    F["blendshapes_std"],    # (N, 52)
    F["blendshapes_delta"],  # (N, 52)
    F["head_pose_mean"],     # (N, 3)
    F["head_pose_std"],      # (N, 3)
    F["eye_gaze_mean"],      # (N, 6)
    F["eye_gaze_std"],       # (N, 6)
    F["landmark_summary_mean"],  # (N, 12)
    F["detected_count"].reshape(-1, 1),  # (N, 1)
], axis=1).astype(np.float32)

clip_ids = F["clip_id"]
split    = F["split"]
subject  = F["subject_id"]
y        = F["engagement"].astype(np.int64)
N = len(clip_ids)
log(f"  features: {feat.shape}  N={N}")

# Detection quality check
det = F["detected_count"]
log(f"  detection: mean={det.mean():.1f} min={det.min()} max={det.max()} "
    f"  zero_det={( det==0).sum()} clips")

# --- Per-subject mean subtraction (no division — avoids NaN for subjects with few clips) ---
log("Applying per-subject mean subtraction...")
feat_norm = feat.copy()
for subj in np.unique(subject):
    # Use train clips to compute subject baseline
    tr_subj = (subject == subj) & (split == "Train")
    if tr_subj.sum() < 1:
        continue
    mu = feat[tr_subj].mean(axis=0)
    # Apply to all splits for this subject
    for sp in ["Train", "Validation", "Test"]:
        mask = (subject == subj) & (split == sp)
        if mask.sum() == 0:
            continue
        feat_norm[mask] = feat[mask] - mu

# Global standardization after subject centering
tr_all = (split == "Train")
mu_g = feat_norm[tr_all].mean(axis=0)
sd_g = feat_norm[tr_all].std(axis=0) + 1e-6
feat_norm = (feat_norm - mu_g) / sd_g

# Clip and fix NaN/inf
feat_norm = np.nan_to_num(feat_norm, nan=0.0, posinf=5.0, neginf=-5.0)
feat_norm = np.clip(feat_norm, -5, 5)

# Drop near-zero variance features (cause ill-conditioning)
tr_all = (split == "Train")
var_mask = feat_norm[tr_all].std(axis=0) > 0.01
feat_norm = feat_norm[:, var_mask]
log(f"  kept {var_mask.sum()}/{len(var_mask)} features after variance filter")

tr_mask = (split == "Train")
va_mask = (split == "Validation")
te_mask = (split == "Test")
log(f"  splits: tr={tr_mask.sum()} va={va_mask.sum()} te={te_mask.sum()}")

X_tr = feat_norm[tr_mask]; y_tr = y[tr_mask]
X_va = feat_norm[va_mask]; y_va = y[va_mask]
X_te = feat_norm[te_mask]; y_te = y[te_mask]

# --- SETA recipe ---
N_BAGS = 20
SEEDS  = list(range(5))

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

log(f"Bagging {N_BAGS} bags x {len(SEEDS)} seeds...")
all_proba_va = []; all_proba_te = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    for bag in range(N_BAGS):
        idx = rng.integers(0, len(X_tr), size=len(X_tr))
        clf = LogisticRegression(max_iter=2000, C=0.1, random_state=seed*100+bag,
                                 multi_class="multinomial", solver="saga")
        clf.fit(X_tr[idx], y_tr[idx])
        all_proba_va.append(clf.predict_proba(X_va))
        all_proba_te.append(clf.predict_proba(X_te))
    log(f"  seed {seed} done")

proba_va = np.mean(all_proba_va, axis=0)
proba_te = np.mean(all_proba_te, axis=0)
e_va = proba_va @ np.array([0, 1, 2, 3])
e_te = proba_te @ np.array([0, 1, 2, 3])

bt  = tune_thresholds(e_va, y_va)
yp_te = apply_thr(e_te, bt["t"])
kq  = float(cohen_kappa_score(y_te, yp_te, weights="quadratic"))
acc = float(accuracy_score(y_te, yp_te))
ci  = boot_ci(y_te, yp_te)

log(f"\nRESULT: kappa_q={kq:.4f}  CI={ci}  acc={acc:.3f}  t={bt['t']}")

result = {"method": "behavioral_seta", "test_kq": kq, "test_acc": acc,
          "ci": ci, "thresholds": bt["t"], "val_kq": bt["v"],
          "feature_dim": feat.shape[1]}
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
log(f"Saved: {OUT}")
