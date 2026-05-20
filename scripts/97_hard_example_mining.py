"""
MOONSHOT 8: Hard example mining / curriculum reweighting.

Approach: train initial LR bag, compute per-sample loss (high loss = hard example).
Upweight hard examples in retraining. Iterate.

Variants:
  A) Standard sample-weight by inverse-confidence
  B) Focal-style: weight by (1 - max_prob)^gamma
  C) Drop noisy: remove training samples with prob disagreement with true label > τ

Output:
  results/sota/moonshot_hardmining.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_hardmining.json")
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


def fit_lr(Xtr, ytr, Xva, yva, sample_weight=None):
    best = None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr, sample_weight=sample_weight)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": float(v), "clf": clf}
    return best["clf"]


def bag_lr(Xtr, ytr, Xva, yva, Xte, K=20, seeds=[0, 7, 42, 2025, 1024], sample_weight=None):
    all_te, all_va = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        bt, bv = [], []
        for k in range(K):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            sw = sample_weight[idx] if sample_weight is not None else None
            clf = fit_lr(Xtr[idx], ytr[idx], Xva, yva, sample_weight=sw)
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


def get_oof_predictions(Xtr, ytr, Xva, yva, n_folds=5):
    """Out-of-fold predictions on train via K-fold."""
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    p_tr_oof = np.zeros((len(ytr), 4))
    for cal, hold in kf.split(Xtr):
        clf = fit_lr(Xtr[cal], ytr[cal], Xva, yva)
        p_tr_oof[hold] = clf.predict_proba(Xtr[hold])
    return p_tr_oof


def main():
    print("Loading...", flush=True)
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    Xtr, ytr = X[tr], eng[tr]; Xva, yva = X[va], eng[va]; Xte, yte = X[te], eng[te]

    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    out = {}

    # Step 1: Get OOF train predictions
    print("\n[1] Computing out-of-fold predictions on train (5 folds)...", flush=True)
    t0 = time.time()
    p_tr_oof = get_oof_predictions(Xtr, ytr, Xva, yva, n_folds=5)
    print(f"  Done ({time.time()-t0:.0f}s)", flush=True)

    # Compute per-sample loss / margin
    p_true = p_tr_oof[np.arange(len(ytr)), ytr]  # probability of true class
    p_max = p_tr_oof.max(axis=1)
    correct = p_tr_oof.argmax(axis=1) == ytr
    print(f"\n  OOF train acc: {correct.mean():.3f}", flush=True)
    print(f"  p_true mean: {p_true.mean():.3f}, p_max mean: {p_max.mean():.3f}", flush=True)

    # ============ Variant A: focal-style sample weights (1 - p_true)^gamma ============
    print("\n[A] Focal sample weights...", flush=True)
    for gamma in [1.0, 2.0, 4.0]:
        sw = (1 - p_true) ** gamma + 0.1  # add baseline so not all weights = 0
        sw = sw / sw.mean()  # normalize
        print(f"  gamma={gamma}: weight range [{sw.min():.3f}, {sw.max():.3f}]", flush=True)
        t0 = time.time()
        p_te, p_va = bag_lr(Xtr, ytr, Xva, yva, Xte, K=20, sample_weight=sw)
        print(f"  Bag done ({time.time()-t0:.0f}s)", flush=True)
        m_solo = metrics(yte, p_te.argmax(1))
        e_va = (p_va * classes[None, :]).sum(1); e_te = (p_te * classes[None, :]).sum(1)
        bt = tune_thresh(e_va, yva)
        m_thr = metrics(yte, apply_t(e_te, bt["t"]))
        out[f"focal_gamma{gamma}"] = {"solo": m_solo, "threshold": {**m_thr, **bt}}
        print(f"  Solo: κ_q={m_solo['kappa_q']:.4f}  +thresh: κ_q={m_thr['kappa_q']:.4f} {m_thr['kappa_q_ci']}", flush=True)

    # ============ Variant B: drop very hard (likely mislabeled) examples ============
    print("\n[B] Drop hardest examples (high disagreement)...", flush=True)
    for drop_frac in [0.05, 0.10, 0.15]:
        # Sort by p_true ascending; drop bottom fraction
        thresh = np.quantile(p_true, drop_frac)
        keep = p_true > thresh
        n_dropped = (~keep).sum()
        print(f"  drop_frac={drop_frac}: thresh={thresh:.3f}, dropped {n_dropped}/{len(ytr)}", flush=True)
        t0 = time.time()
        p_te, p_va = bag_lr(Xtr[keep], ytr[keep], Xva, yva, Xte, K=20)
        print(f"  Bag done ({time.time()-t0:.0f}s)", flush=True)
        m_solo = metrics(yte, p_te.argmax(1))
        e_va = (p_va * classes[None, :]).sum(1); e_te = (p_te * classes[None, :]).sum(1)
        bt = tune_thresh(e_va, yva)
        m_thr = metrics(yte, apply_t(e_te, bt["t"]))
        out[f"drop_hard_{drop_frac}"] = {"solo": m_solo, "threshold": {**m_thr, **bt}, "n_dropped": int(n_dropped)}
        print(f"  Solo: κ_q={m_solo['kappa_q']:.4f}  +thresh: κ_q={m_thr['kappa_q']:.4f} {m_thr['kappa_q_ci']}", flush=True)

    # ============ Variant C: upweight high-confidence correct examples ============
    print("\n[C] Upweight easy correct examples (curriculum reverse)...", flush=True)
    sw = (p_true + 0.1) ** 2  # high weight when p_true is high
    sw = sw / sw.mean()
    print(f"  weight range [{sw.min():.3f}, {sw.max():.3f}]", flush=True)
    t0 = time.time()
    p_te, p_va = bag_lr(Xtr, ytr, Xva, yva, Xte, K=20, sample_weight=sw)
    print(f"  Bag done ({time.time()-t0:.0f}s)", flush=True)
    m_solo = metrics(yte, p_te.argmax(1))
    e_va = (p_va * classes[None, :]).sum(1); e_te = (p_te * classes[None, :]).sum(1)
    bt = tune_thresh(e_va, yva)
    m_thr = metrics(yte, apply_t(e_te, bt["t"]))
    out["easy_curriculum"] = {"solo": m_solo, "threshold": {**m_thr, **bt}}
    print(f"  Solo: κ_q={m_solo['kappa_q']:.4f}  +thresh: κ_q={m_thr['kappa_q']:.4f} {m_thr['kappa_q_ci']}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
