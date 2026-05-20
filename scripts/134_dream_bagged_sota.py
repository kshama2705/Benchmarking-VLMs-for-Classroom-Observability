"""
DREAM bagged SOTA — apply the prior SOTA pipeline (LR-uniform K=20 x 5 seeds
bag + ordinal threshold tuning) to the best DREAM modulation (CAT, K=5,
P-train).

Compare apples-to-apples to the baseline raw SigLIP-L bagged SOTA at κ=0.238.

This is the headline DREAM number for the paper. If it doesn't beat 0.238,
DREAM is honestly a negative result.
"""

import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
ANCHORS = os.path.join(BASE, "features", "dream_anchors.npz")
OUT = os.path.join(BASE, "results", "dream", "dream_bagged_sota.json")
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


def bag_lr(Xtr, ytr, Xva, yva, Xte, K=20, seeds=(0, 7, 42, 2025, 1024)):
    all_te = []; all_va = []
    for s in seeds:
        rng = np.random.default_rng(s)
        for _ in range(K):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            clf = fit_lr(Xtr[idx], ytr[idx], Xva, yva)
            all_te.append(clf.predict_proba(Xte))
            all_va.append(clf.predict_proba(Xva))
    return np.mean(all_te, axis=0), np.mean(all_va, axis=0)


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
    d = np.load(SIGLIP, allow_pickle=True)
    feat = d['feat'].astype(np.float32)
    split = d['split']; subject = d['subject_id']; eng = d['engagement'].astype(np.int64)
    a_npz = np.load(ANCHORS, allow_pickle=True)
    subjects = a_npz['subject_ids']

    tr_mask = (split == 'Train'); va_mask = (split == 'Validation'); te_mask = (split == 'Test')
    Xtr_raw = feat[tr_mask]; ytr = eng[tr_mask]
    Xva_raw = feat[va_mask]; yva = eng[va_mask]
    Xte_raw = feat[te_mask]; yte = eng[te_mask]

    classes = np.arange(4, dtype=np.float32)
    results = {}

    def evaluate(Xtr, Xva, Xte, name):
        t0 = time.time()
        p_te, p_va = bag_lr(Xtr, ytr, Xva, yva, Xte, K=20)
        e_te = (p_te * classes[None]).sum(1); e_va = (p_va * classes[None]).sum(1)
        yp_arg = p_te.argmax(1)
        bt = tune_thresholds(e_va, yva)
        yp_thr = apply_thr(e_te, bt['t'])
        kq_arg = float(cohen_kappa_score(yte, yp_arg, weights="quadratic"))
        kq_thr = float(cohen_kappa_score(yte, yp_thr, weights="quadratic"))
        ci_arg = boot_ci(yte, yp_arg)
        ci_thr = boot_ci(yte, yp_thr)
        dt = time.time() - t0
        print(f"[{name}] arg κ={kq_arg:.4f} {ci_arg}  thr κ={kq_thr:.4f} {ci_thr}  t={bt['t']}  [{dt:.0f}s]", flush=True)
        return {"name": name,
                "argmax": {"kq": kq_arg, "ci": ci_arg, "acc": float(accuracy_score(yte, yp_arg))},
                "threshold": {"kq": kq_thr, "ci": ci_thr, "t": bt['t'], "val_kq": bt['v'],
                              "acc": float(accuracy_score(yte, yp_thr))}}

    # Baseline (raw SigLIP-L bagged)
    print("\n=== BAGGED BASELINE (raw SigLIP-L) ===", flush=True)
    results["baseline_bagged"] = evaluate(Xtr_raw, Xva_raw, Xte_raw, "baseline_bagged")

    # DREAM CAT K=5, p_train
    print("\n=== BAGGED DREAM-CAT K=5 P-train ===", flush=True)
    anchor_idx = a_npz["p_train_anchor_idx_k5"]
    anchor_means = {}
    for si, s in enumerate(subjects):
        idx = anchor_idx[si]; idx = idx[idx >= 0]
        anchor_means[s] = feat[idx].mean(0) if len(idx) > 0 else np.zeros(feat.shape[1], np.float32)
    a_full = np.stack([anchor_means[s] for s in subject], axis=0)

    Xtr_cat = np.concatenate([feat[tr_mask], feat[tr_mask] - a_full[tr_mask]], axis=1)
    Xva_cat = np.concatenate([feat[va_mask], feat[va_mask] - a_full[va_mask]], axis=1)
    Xte_cat = np.concatenate([feat[te_mask], feat[te_mask] - a_full[te_mask]], axis=1)

    results["dream_cat_K5_bagged"] = evaluate(Xtr_cat, Xva_cat, Xte_cat, "dream_cat_K5_bagged")

    # Also test DREAM-SUB K=5 bagged (expected to fail)
    print("\n=== BAGGED DREAM-SUB K=5 P-train ===", flush=True)
    Xtr_sub = feat[tr_mask] - a_full[tr_mask]
    Xva_sub = feat[va_mask] - a_full[va_mask]
    Xte_sub = feat[te_mask] - a_full[te_mask]
    results["dream_sub_K5_bagged"] = evaluate(Xtr_sub, Xva_sub, Xte_sub, "dream_sub_K5_bagged")

    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
