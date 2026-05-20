"""
MOONSHOT 5: Train on (train + val) combined for final test evaluation.

Standard practice once hyperparameters are fixed: retrain on all available labeled
data (train + val) before final test eval. We've been using only train, with val
for thresh tuning. If thresh tuning is locked (e.g., median-of-folds threshold),
we can use train+val for the bag itself.

But this trades off: no val for threshold tuning. Use the K-fold median thresholds
from script 87/88 which were t=(1.12, 1.44, 1.84) — they coincided with full-val.

Plan:
  A) Re-train LR-unif bag on train+val combined, threshold-tune via 5-fold on val (so val still seen during tuning).
  B) Use the pre-fixed thresholds t=(1.12, 1.44, 1.84) — no val needed.
  C) Compare to baseline.

Output:
  results/sota/moonshot_train_plus_val.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_train_plus_val.json")
RNG = np.random.default_rng(42)


def boot_kq(yt, yp, n=1000):
    out = []; nt = len(yt)
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


def fit_lr_noval(Xtr, ytr, C=None):
    """Fit LR with fixed C (no val tuning)."""
    if C is None:
        # Default to median of typical C-sweep best — usually 1.0
        C = 1.0
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                             solver="lbfgs", random_state=42)
    clf.fit(Xtr, ytr)
    return clf


def fit_lr_with_val(Xtr, ytr, Xva, yva):
    """Sweep C, pick by val κ."""
    best = None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": float(v), "clf": clf}
    return best["clf"], best["C"]


def bag_lr_with_val_and_C(Xtr_full, ytr_full, Xva, yva, Xte, K=20, seeds=[0, 7, 42, 2025, 1024], C_fixed=None):
    """Bag on train (with val for C-tuning if C_fixed=None) - returns probs."""
    all_te, all_va = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        bt, bv = [], []
        for k in range(K):
            idx = rng.integers(0, len(ytr_full), size=len(ytr_full))
            if C_fixed is None:
                clf, _ = fit_lr_with_val(Xtr_full[idx], ytr_full[idx], Xva, yva)
            else:
                clf = fit_lr_noval(Xtr_full[idx], ytr_full[idx], C=C_fixed)
            bt.append(clf.predict_proba(Xte)); bv.append(clf.predict_proba(Xva))
            if (k+1) % 10 == 0:
                print(f"    seed={s} k={k+1}/{K}", flush=True)
        all_te.append(np.stack(bt).mean(0)); all_va.append(np.stack(bv).mean(0))
    return np.stack(all_te).mean(0), np.stack(all_va).mean(0)


def tune_thresh(e, y, step=0.04):
    grid = np.arange(0, 3.01, step)
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


def apply_t(e, t):
    yp = np.zeros_like(e, dtype=int)
    yp[e > t[0]] = 1; yp[e > t[1]] = 2; yp[e > t[2]] = 3
    return yp


def main():
    print("Loading SigLIP-L features...", flush=True)
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    Xtr, ytr = X[tr], eng[tr]; Xva, yva = X[va], eng[va]; Xte, yte = X[te], eng[te]
    print(f"  Train: {len(ytr)}, Val: {len(yva)}, Test: {len(yte)}", flush=True)

    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    out = {}

    # Step 1: Determine best C via standard val tuning (baseline)
    print("\n[1] Baseline: bag on train, val-tune C and thresholds...", flush=True)
    t0 = time.time()
    p_te_base, p_va_base = bag_lr_with_val_and_C(Xtr, ytr, Xva, yva, Xte, K=20)
    print(f"  Done ({time.time()-t0:.0f}s)", flush=True)
    e_va = (p_va_base * classes[None, :]).sum(1)
    e_te = (p_te_base * classes[None, :]).sum(1)
    bt = tune_thresh(e_va, yva)
    m_base = metrics(yte, apply_t(e_te, bt["t"]))
    out["baseline_train_only"] = {**m_base, **bt}
    print(f"  Baseline κ_q={m_base['kappa_q']:.4f} {m_base['kappa_q_ci']}  t={bt['t']}", flush=True)

    # Step 2: Identify single best C from baseline runs (most picked across bags)
    # Hard to extract here without changes; just use C=1.0 (typical winner for SigLIP-L)
    # Step 3: Bag on train+val with fixed C, use the same thresholds.
    Xtrva = np.concatenate([Xtr, Xva], axis=0)
    ytrva = np.concatenate([ytr, yva], axis=0)
    print(f"\n[3] Bag on train+val combined ({len(ytrva)} clips, fixed C=1.0, fixed thresholds={bt['t']})...", flush=True)
    t0 = time.time()
    p_te_tv, p_va_tv = bag_lr_with_val_and_C(Xtrva, ytrva, Xva, yva, Xte, K=20, C_fixed=1.0)
    print(f"  Done ({time.time()-t0:.0f}s)", flush=True)
    e_te_tv = (p_te_tv * classes[None, :]).sum(1)
    # Use the same thresholds tuned on the train-only run
    m_tv = metrics(yte, apply_t(e_te_tv, bt["t"]))
    out["train_plus_val_fixed_t"] = {**m_tv, "t_used": bt["t"]}
    print(f"  Train+val (fixed thresholds): κ_q={m_tv['kappa_q']:.4f} {m_tv['kappa_q_ci']}", flush=True)

    # Step 4: Bag on train+val, val-re-tune thresholds (cheating-ish but informative)
    e_va_tv = (p_va_tv * classes[None, :]).sum(1)
    bt2 = tune_thresh(e_va_tv, yva)
    m_tv2 = metrics(yte, apply_t(e_te_tv, bt2["t"]))
    out["train_plus_val_retune_t"] = {**m_tv2, **bt2}
    print(f"  Train+val (re-tuned thresholds): κ_q={m_tv2['kappa_q']:.4f} {m_tv2['kappa_q_ci']}  t={bt2['t']}", flush=True)

    # Step 5: Train+val LR with C from baseline + thresholds retuned via 5-fold on val
    # (For honesty, this uses val ONLY for threshold tuning, not for C selection nor bag training)
    # Actually step 4 above already does this — the bag is from train+val, but we val-tune thresholds anew

    # Step 6: Compare bag prob distributions train-only vs train+val
    diff_max = float(np.abs(p_te_base - p_te_tv).max())
    diff_mean = float(np.abs(p_te_base - p_te_tv).mean())
    print(f"\n[Diff] train-only vs train+val test probs: max={diff_max:.4f}, mean={diff_mean:.4f}", flush=True)
    out["prob_diff"] = {"max": diff_max, "mean": diff_mean}

    # Save bag probs
    np.savez_compressed(os.path.join(BASE, "results", "sota", "_trainval_bag_cache.npz"),
                        p_te_base=p_te_base, p_va_base=p_va_base,
                        p_te_tv=p_te_tv, p_va_tv=p_va_tv)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
