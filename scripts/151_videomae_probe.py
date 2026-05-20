"""
VideoMAE-base probe on DAiSEE — apply the SOTA recipe (bagged LR + ordinal
threshold tuning) to VideoMAE features and compare to the SigLIP-L baseline.

If VideoMAE breaks the κ_q ≈ 0.24 ceiling, this becomes a major paper
contribution: "Video temporal encoders escape the frozen-feature ceiling."

Expects features/daisee_videomae_features.npz to exist. Will also try
partial features if the final is missing.
"""

import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(BASE, "features", "daisee_videomae_features.npz")
PARTIAL = os.path.join(BASE, "features", "daisee_videomae_partial.npz")
OUT = os.path.join(BASE, "results", "sota", "videomae_probe.json")
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
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
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
    feat_path = FEAT if os.path.exists(FEAT) else PARTIAL
    if not os.path.exists(feat_path):
        print(f"No features yet at {FEAT} or {PARTIAL}; exiting", flush=True)
        return
    print(f"Loading: {feat_path}", flush=True)
    d = np.load(feat_path, allow_pickle=True)
    feat = d["feat"].astype(np.float32)
    split = d["split"]; subj = d["subject_id"]
    eng = d["engagement"].astype(np.int64)
    success = d.get("success", np.ones(len(feat), dtype=np.int8))

    # Only keep successfully encoded clips
    mask_ok = success.astype(bool)
    print(f"Encoded: {mask_ok.sum()}/{len(mask_ok)} clips ({100*mask_ok.mean():.1f}%)", flush=True)
    tr_mask = mask_ok & (split == "Train")
    va_mask = mask_ok & (split == "Validation")
    te_mask = mask_ok & (split == "Test")
    print(f"  tr={tr_mask.sum()}  va={va_mask.sum()}  te={te_mask.sum()}", flush=True)

    if min(tr_mask.sum(), va_mask.sum(), te_mask.sum()) < 100:
        print("Insufficient encoded clips per split; cannot run probe yet", flush=True)
        return

    Xtr = feat[tr_mask]; ytr = eng[tr_mask]
    Xva = feat[va_mask]; yva = eng[va_mask]
    Xte = feat[te_mask]; yte = eng[te_mask]

    results = {}
    classes = np.arange(4, dtype=np.float32)

    # Single LR (fast sanity)
    t0 = time.time()
    clf = fit_lr(Xtr, ytr, Xva, yva)
    p_te = clf.predict_proba(Xte); p_va = clf.predict_proba(Xva)
    e_te = (p_te * classes[None]).sum(1); e_va = (p_va * classes[None]).sum(1)
    yp_arg = p_te.argmax(1)
    bt = tune_thresholds(e_va, yva)
    yp_thr = apply_thr(e_te, bt['t'])
    kq_arg = float(cohen_kappa_score(yte, yp_arg, weights='quadratic'))
    kq_thr = float(cohen_kappa_score(yte, yp_thr, weights='quadratic'))
    ci_arg = boot_ci(yte, yp_arg); ci_thr = boot_ci(yte, yp_thr)
    dt = time.time() - t0
    print(f"[VideoMAE single LR] arg κ={kq_arg:.4f} {ci_arg}  thr κ={kq_thr:.4f} {ci_thr}  [{dt:.0f}s]", flush=True)
    results["single_lr"] = {"argmax": {"kq": kq_arg, "ci": ci_arg},
                             "threshold": {"kq": kq_thr, "ci": ci_thr, "t": bt['t']}}

    # Bagged LR + threshold (the SOTA recipe)
    print("Running bagged LR + threshold (K=20 × 5 seeds = 100 LRs)...", flush=True)
    t0 = time.time()
    p_te, p_va = bag_lr(Xtr, ytr, Xva, yva, Xte, K=20)
    e_te = (p_te * classes[None]).sum(1); e_va = (p_va * classes[None]).sum(1)
    yp_arg = p_te.argmax(1)
    bt = tune_thresholds(e_va, yva)
    yp_thr = apply_thr(e_te, bt['t'])
    kq_arg = float(cohen_kappa_score(yte, yp_arg, weights='quadratic'))
    kq_thr = float(cohen_kappa_score(yte, yp_thr, weights='quadratic'))
    ci_arg = boot_ci(yte, yp_arg); ci_thr = boot_ci(yte, yp_thr)
    dt = time.time() - t0
    print(f"[VideoMAE BAGGED] arg κ={kq_arg:.4f} {ci_arg}  thr κ={kq_thr:.4f} {ci_thr}  [{dt:.0f}s]", flush=True)
    results["bagged_lr_threshold"] = {"argmax": {"kq": kq_arg, "ci": ci_arg},
                                       "threshold": {"kq": kq_thr, "ci": ci_thr, "t": bt['t']}}

    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
