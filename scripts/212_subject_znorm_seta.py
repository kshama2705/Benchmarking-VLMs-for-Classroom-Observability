"""
Option A: Subject z-score normalization + SETA recipe.

For each subject, subtract their own mean feature vector and divide by std,
computed from that subject's clips within their split (no label leakage).
This directly attacks the within-subject ranking failure (rho~0.05) by making
each clip's features relative to that subject's baseline appearance.

Then runs the same bagged-LR + threshold-tuning SETA recipe on normalized features.
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, accuracy_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(BASE, "results/sota/subject_znorm_seta.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Loading SigLIP-L features...")
F = np.load(os.path.join(BASE, "features/daisee_siglip_l_features.npz"), allow_pickle=True)
feat    = F["feat"].astype(np.float32)       # (N, 1024)
clip_ids = F["clip_id"]
split    = F["split"]
subject  = F["subject_id"]
y        = F["engagement"].astype(np.int64)
N = len(clip_ids)
log(f"  features: {feat.shape}  N={N}")

# --- Per-subject z-score normalization (within each split) ---
log("Applying per-subject z-score normalization...")
feat_norm = feat.copy()
for subj in np.unique(subject):
    for sp in ["Train", "Validation", "Test"]:
        mask = (subject == subj) & (split == sp)
        if mask.sum() < 2:
            continue
        mu = feat[mask].mean(axis=0)
        sd = feat[mask].std(axis=0) + 1e-6
        feat_norm[mask] = (feat[mask] - mu) / sd

log("  normalization done")

tr_mask = (split == "Train")
va_mask = (split == "Validation")
te_mask = (split == "Test")
log(f"  splits: tr={tr_mask.sum()} va={va_mask.sum()} te={te_mask.sum()}")

X_tr = feat_norm[tr_mask]; y_tr = y[tr_mask]
X_va = feat_norm[va_mask]; y_va = y[va_mask]
X_te = feat_norm[te_mask]; y_te = y[te_mask]

# --- SETA recipe: bagged LR + ordinal threshold tuning ---
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

log(f"Bagging {N_BAGS} bags × {len(SEEDS)} seeds...")
all_proba_va = []; all_proba_te = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    for bag in range(N_BAGS):
        idx = rng.integers(0, len(X_tr), size=len(X_tr))
        Xb, yb = X_tr[idx], y_tr[idx]
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed*100+bag,
                                  multi_class="multinomial", solver="lbfgs")
        clf.fit(Xb, yb)
        all_proba_va.append(clf.predict_proba(X_va))
        all_proba_te.append(clf.predict_proba(X_te))
    log(f"  seed {seed} done ({(seed+1)*N_BAGS}/{len(SEEDS)*N_BAGS} bags)")

proba_va = np.mean(all_proba_va, axis=0)
proba_te = np.mean(all_proba_te, axis=0)
e_va = proba_va @ np.array([0, 1, 2, 3])
e_te = proba_te @ np.array([0, 1, 2, 3])

bt = tune_thresholds(e_va, y_va)
yp_te = apply_thr(e_te, bt["t"])
kq  = float(cohen_kappa_score(y_te, yp_te, weights="quadratic"))
acc = float(accuracy_score(y_te, yp_te))
ci  = boot_ci(y_te, yp_te)

log(f"\nRESULT: κ_q={kq:.4f}  CI={ci}  acc={acc:.3f}  t={bt['t']}")

result = {"method": "subject_znorm_seta", "test_kq": kq, "test_acc": acc, "ci": ci,
          "thresholds": bt["t"], "val_kq": bt["v"]}
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
log(f"Saved: {OUT}")
