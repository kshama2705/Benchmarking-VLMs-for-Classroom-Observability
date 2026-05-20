"""
Train+Val combined SOTA recipe.

Use BOTH train and val labels as training data (no val for selection).
Tune thresholds on a held-out CV split of the combined set, then evaluate
on test.

This adds 1429 val samples (27% more data) to the recipe. If the structural
ceiling is data-limited rather than feature-limited, this should lift κ.
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
MULTI = os.path.join(BASE, "features", "daisee_siglip_l_multiframe_features.npz")
OUT = os.path.join(BASE, "results", "sota", "trainval_combined.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)


def boot_ci(yt, yp, n=1000):
    nt = len(yt); out = []
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def fit_lr(Xtr, ytr, Xva, yva, Cs=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0)):
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=4000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": float(v), "clf": clf}
    return best["clf"]


def bag_lr_kfold(Xtv, ytv, Xte, K_outer=5, K_bag=20):
    """K-fold CV on train+val, with bagging within each fold's train portion.
    Each fold uses the held-out part as 'val' for C-tuning.
    Final test predictions averaged across all folds × bags."""
    kf = KFold(n_splits=K_outer, shuffle=True, random_state=42)
    all_te = []
    classes = np.arange(4, dtype=np.float32)
    fold_kqs = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(Xtv)):
        Xtr = Xtv[tr_idx]; ytr = ytv[tr_idx]
        Xva = Xtv[va_idx]; yva = ytv[va_idx]
        # Bag within this fold
        fold_te = []
        for s in [0, 7, 42, 2025, 1024]:
            rng = np.random.default_rng(s)
            for _ in range(K_bag):
                idx = rng.integers(0, len(ytr), size=len(ytr))
                clf = fit_lr(Xtr[idx], ytr[idx], Xva, yva)
                fold_te.append(clf.predict_proba(Xte))
        p_te = np.mean(fold_te, axis=0); all_te.append(p_te)
        # diagnostic
        e_te = (p_te * classes[None]).sum(1)
        fold_kqs.append({"fold": fold, "n_train": len(tr_idx), "n_val": len(va_idx)})
    return np.mean(all_te, axis=0), fold_kqs


def tune_thresholds(e, y, step=0.04):
    grid = np.arange(0.0, 3.01, step)
    best = None
    for t1 in grid:
        for t2 in grid[grid > t1]:
            for t3 in grid[grid > t2]:
                yp = np.zeros_like(e, dtype=int)
                yp[e > t1] = 1; yp[e > t2] = 2; yp[e > t3] = 3
                v = cohen_kappa_score(y, yp, weights='quadratic')
                if best is None or v > best['v']:
                    best = {'t': (float(t1), float(t2), float(t3)), 'v': float(v)}
    return best


def apply_thr(e, t):
    yp = np.zeros_like(e, dtype=int)
    yp[e > t[0]] = 1; yp[e > t[1]] = 2; yp[e > t[2]] = 3
    return yp


def main():
    print("Loading features...", flush=True)
    d = np.load(SIGLIP, allow_pickle=True)
    feat = d["feat"].astype(np.float32)
    split = d["split"]; eng = d["engagement"].astype(np.int64)

    tr_mask = (split == "Train"); va_mask = (split == "Validation"); te_mask = (split == "Test")
    Xtr = feat[tr_mask]; ytr = eng[tr_mask]
    Xva = feat[va_mask]; yva = eng[va_mask]
    Xte = feat[te_mask]; yte = eng[te_mask]

    # Strategy 1: Train only with K-fold CV for threshold (same as baseline)
    classes = np.arange(4, dtype=np.float32)

    # Strategy 2: Train+Val combined for the LR bag, K-fold for threshold tuning
    Xtv = np.concatenate([Xtr, Xva], axis=0); ytv = np.concatenate([ytr, yva], axis=0)
    print(f"Train+Val combined: {len(ytv)} samples", flush=True)

    print("\n=== Strategy: Train+Val combined LR bag (K-fold CV for thresholds) ===", flush=True)
    t0 = time.time()
    p_te, fold_recs = bag_lr_kfold(Xtv, ytv, Xte, K_outer=5, K_bag=20)
    print(f"  bagging took {(time.time()-t0)/60:.1f} min", flush=True)
    e_te = (p_te * classes[None]).sum(1)
    yp_te_arg = p_te.argmax(1)
    kq_arg = float(cohen_kappa_score(yte, yp_te_arg, weights="quadratic"))
    print(f"  argmax κ={kq_arg:.4f}", flush=True)

    # Tune thresholds on K-fold CV
    print("\n=== K-fold threshold tuning on train+val ===", flush=True)
    # Get OOF expected-class scores for train+val: for thresholds, we need scores on TV
    # Run a separate per-fold bag, save tv predictions
    p_tv_oof = np.zeros((len(ytv), 4), dtype=np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(Xtv)):
        Xtr_f = Xtv[tr_idx]; ytr_f = ytv[tr_idx]
        Xva_f = Xtv[va_idx]; yva_f = ytv[va_idx]
        bag = []
        for s in [0, 7, 42, 2025, 1024]:
            rng = np.random.default_rng(s)
            for _ in range(20):
                idx = rng.integers(0, len(ytr_f), size=len(ytr_f))
                clf = fit_lr(Xtr_f[idx], ytr_f[idx], Xva_f, yva_f)
                bag.append(clf.predict_proba(Xva_f))
        p_tv_oof[va_idx] = np.mean(bag, axis=0)
    e_tv = (p_tv_oof * classes[None]).sum(1)
    bt = tune_thresholds(e_tv, ytv)
    print(f"  thresholds (OOF tune on T+V): {bt['t']}  OOF-val κ={bt['v']:.4f}", flush=True)

    yp_te_thr = apply_thr(e_te, bt['t'])
    kq_thr = float(cohen_kappa_score(yte, yp_te_thr, weights="quadratic"))
    ci_thr = boot_ci(yte, yp_te_thr)
    print(f"\n*** Train+Val combined recipe: test κ_q = {kq_thr:.4f}  CI={ci_thr}  acc={accuracy_score(yte, yp_te_thr):.3f} ***", flush=True)

    # Also: tune thresholds on TRAIN-only K-fold OOF, to isolate the effect of train+val
    print("\n=== Reference: thresholds tuned on TRAIN-only OOF (data parity) ===", flush=True)
    p_tr_oof = np.zeros((len(ytr), 4), dtype=np.float32)
    kf2 = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf2.split(Xtr)):
        bag = []
        for s in [0, 7, 42, 2025, 1024]:
            rng = np.random.default_rng(s)
            for _ in range(20):
                idx = rng.integers(0, len(tr_idx), size=len(tr_idx))
                clf = fit_lr(Xtr[tr_idx][idx], ytr[tr_idx][idx], Xtr[va_idx], ytr[va_idx])
                bag.append(clf.predict_proba(Xtr[va_idx]))
        p_tr_oof[va_idx] = np.mean(bag, axis=0)
    e_tr = (p_tr_oof * classes[None]).sum(1)
    bt2 = tune_thresholds(e_tr, ytr)
    print(f"  thresholds (OOF tune on Train only): {bt2['t']}  OOF-train κ={bt2['v']:.4f}", flush=True)

    out = {
        "trainval_combined": {
            "argmax_kq": kq_arg,
            "threshold_kq": kq_thr,
            "threshold_ci": ci_thr,
            "thresholds": bt['t'],
            "accuracy": accuracy_score(yte, yp_te_thr),
        },
        "train_only_oof_thresholds": {
            "thresholds": bt2['t'], "oof_kq": bt2['v'],
        },
        "fold_info": fold_recs,
    }
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
