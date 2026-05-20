"""
Two orthogonal angles to push past κ=0.228:

[A] Pseudo-labeling iterative refinement:
    - Take current SOTA test predictions (cached fusion probs)
    - Select top-X% high-confidence test samples as pseudo-labels
    - Add to training set with hard labels
    - Retrain LR + RSB bags, re-fuse, re-threshold
    - Iterate up to 3 rounds

[B] Cross-validated ordinal threshold tuning:
    - Instead of single val-tuned thresholds, K-fold CV on train+val (or just val)
    - Mean and median threshold estimates may generalize better
    - Test if "robust" thresholds beat point-estimate thresholds on test

Uses cached bag probs from script 80 to skip the ~20min bag rebuild.

Output:
  results/sota/pseudolabel_threshold.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import KFold

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "pseudolabel_threshold.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)


def boot_kq(yt, yp, n=1000):
    out = []
    nt = len(yt)
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


def fit_lr(Xtr, ytr, Xva, yva):
    best = None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": float(v), "clf": clf}
    return best["clf"]


def bag_lr_uniform(Xtr, ytr, Xva, yva, Xte, K=20, seeds=[0, 7, 42, 2025, 1024]):
    all_te, all_va = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        bt, bv = [], []
        for _ in range(K):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            clf = fit_lr(Xtr[idx], ytr[idx], Xva, yva)
            bt.append(clf.predict_proba(Xte)); bv.append(clf.predict_proba(Xva))
        all_te.append(np.stack(bt).mean(0)); all_va.append(np.stack(bv).mean(0))
    return np.stack(all_te).mean(0), np.stack(all_va).mean(0)


def bag_rsb(Xtr, ytr, Xva, yva, Xte, K=20, frac=0.45, seeds=[0, 7, 42, 2025, 1024]):
    D = Xtr.shape[1]
    n_feat = int(D * frac)
    all_te, all_va = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        bt, bv = [], []
        for _ in range(K):
            r_idx = rng.integers(0, len(ytr), size=len(ytr))
            f_idx = rng.choice(D, size=n_feat, replace=False)
            clf = fit_lr(Xtr[r_idx][:, f_idx], ytr[r_idx], Xva[:, f_idx], yva)
            bt.append(clf.predict_proba(Xte[:, f_idx]))
            bv.append(clf.predict_proba(Xva[:, f_idx]))
        all_te.append(np.stack(bt).mean(0)); all_va.append(np.stack(bv).mean(0))
    return np.stack(all_te).mean(0), np.stack(all_va).mean(0)


def tune_thresholds(e, y, grid_step=0.02):
    """Find best (t1, t2, t3) for quad-κ on expected-class score."""
    grid = np.arange(0.0, 3.01, grid_step)
    best = None
    for t1 in grid:
        for t2 in grid[grid > t1]:
            for t3 in grid[grid > t2]:
                yp = np.zeros_like(e, dtype=int)
                yp[e > t1] = 1
                yp[e > t2] = 2
                yp[e > t3] = 3
                v = cohen_kappa_score(y, yp, weights="quadratic")
                if best is None or v > best["v"]:
                    best = {"t1": float(t1), "t2": float(t2), "t3": float(t3), "v": float(v)}
    return best


def apply_thresholds(e, t1, t2, t3):
    yp = np.zeros_like(e, dtype=int)
    yp[e > t1] = 1
    yp[e > t2] = 2
    yp[e > t3] = 3
    return yp


def main():
    print("Loading SigLIP-L features + cached bag probs...")
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    splits = d["split"]; eng = d["engagement"].astype(np.int64)
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    Xtr, ytr = X[tr], eng[tr]; Xva, yva = X[va], eng[va]; Xte, yte = X[te], eng[te]

    if not os.path.exists(CACHE):
        print("ERROR: bag cache missing. Run script 80 first.")
        return
    c = np.load(CACHE)
    p_te_unif = c["p_te_unif"]; p_va_unif = c["p_va_unif"]
    p_te_rsb = c["p_te_rsb"]; p_va_rsb = c["p_va_rsb"]

    out = {}
    classes = np.array([0, 1, 2, 3], dtype=np.float32)

    print("\n=== Baseline fusion + threshold (from cache) ===")
    # Val-tune fusion weight
    best_w = None
    for w in np.linspace(0, 1, 21):
        pv = w * p_va_unif + (1 - w) * p_va_rsb
        v = cohen_kappa_score(yva, pv.argmax(1), weights="quadratic")
        if best_w is None or v > best_w["v"]:
            best_w = {"w_unif": float(w), "v": float(v)}
    wu = best_w["w_unif"]
    p_va_fuse = wu * p_va_unif + (1 - wu) * p_va_rsb
    p_te_fuse = wu * p_te_unif + (1 - wu) * p_te_rsb
    e_va = (p_va_fuse * classes[None, :]).sum(1)
    e_te = (p_te_fuse * classes[None, :]).sum(1)

    base_t = tune_thresholds(e_va, yva)
    yp_te_base = apply_thresholds(e_te, base_t["t1"], base_t["t2"], base_t["t3"])
    m_base = metrics(yte, yp_te_base)
    out["baseline"] = {**m_base, **base_t, "w_unif": wu, "val_kq": float(best_w["v"])}
    print(f"  w_unif={wu:.2f}  ts=({base_t['t1']:.2f},{base_t['t2']:.2f},{base_t['t3']:.2f})  "
          f"test κ_q={m_base['kappa_q']:.3f} {m_base['kappa_q_ci']}")

    print("\n=== [B] K-fold CV threshold tuning on val ===")
    # K-fold the val set, tune thresholds on each fold's training portion, evaluate stability
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_ts = []
    for fold_idx, (cal_idx, _hold_idx) in enumerate(kf.split(e_va)):
        ft = tune_thresholds(e_va[cal_idx], yva[cal_idx])
        fold_ts.append((ft["t1"], ft["t2"], ft["t3"]))
        print(f"  fold {fold_idx}: t=({ft['t1']:.2f},{ft['t2']:.2f},{ft['t3']:.2f})  v={ft['v']:.3f}")
    fold_ts = np.array(fold_ts)
    mean_t = fold_ts.mean(0); median_t = np.median(fold_ts, axis=0)
    print(f"  mean   t=({mean_t[0]:.3f},{mean_t[1]:.3f},{mean_t[2]:.3f})")
    print(f"  median t=({median_t[0]:.3f},{median_t[1]:.3f},{median_t[2]:.3f})")
    yp_te_mean = apply_thresholds(e_te, *mean_t)
    yp_te_med = apply_thresholds(e_te, *median_t)
    out["kfold_mean_t"] = {**metrics(yte, yp_te_mean), "t": mean_t.tolist()}
    out["kfold_median_t"] = {**metrics(yte, yp_te_med), "t": median_t.tolist()}
    print(f"  Mean-t   test κ_q={out['kfold_mean_t']['kappa_q']:.3f} {out['kfold_mean_t']['kappa_q_ci']}")
    print(f"  Median-t test κ_q={out['kfold_median_t']['kappa_q']:.3f} {out['kfold_median_t']['kappa_q_ci']}")

    print("\n=== [A] Pseudo-labeling iterative refinement ===")
    # Build initial pseudo-labeled set from high-conf test predictions
    initial_yhat_te = yp_te_base  # use baseline threshold-tuned prediction
    initial_conf = p_te_fuse.max(1)

    last_kq = m_base["kappa_q"]
    pl_history = []
    for it in range(1, 4):
        print(f"\n--- Pseudo-label round {it} ---")
        # Use predictions from current best (initially baseline)
        cur_yhat = yp_te_base if it == 1 else last_yhat_te
        cur_conf = initial_conf if it == 1 else last_conf
        # Top 50% high-confidence test samples
        thr_conf = np.quantile(cur_conf, 0.5)
        sel = cur_conf >= thr_conf
        n_sel = int(sel.sum())
        print(f"  selecting {n_sel}/{len(yte)} high-conf test samples (thresh conf={thr_conf:.3f})")

        # Augment train with pseudo labels
        X_aug = np.vstack([Xtr, Xte[sel]])
        y_aug = np.concatenate([ytr, cur_yhat[sel]])

        # Quick uniform bag with K=10 (smaller for speed)
        t0 = time.time()
        p_te_u, p_va_u = bag_lr_uniform(X_aug, y_aug, Xva, yva, Xte, K=10)
        print(f"  uniform bag done ({time.time()-t0:.0f}s)")
        t0 = time.time()
        p_te_r, p_va_r = bag_rsb(X_aug, y_aug, Xva, yva, Xte, K=10, frac=0.45)
        print(f"  rsb-45 bag done ({time.time()-t0:.0f}s)")

        # Re-fuse + re-threshold
        best_w = None
        for w in np.linspace(0, 1, 21):
            pv = w * p_va_u + (1 - w) * p_va_r
            v = cohen_kappa_score(yva, pv.argmax(1), weights="quadratic")
            if best_w is None or v > best_w["v"]:
                best_w = {"w_unif": float(w), "v": float(v)}
        wu_it = best_w["w_unif"]
        p_va_it = wu_it * p_va_u + (1 - wu_it) * p_va_r
        p_te_it = wu_it * p_te_u + (1 - wu_it) * p_te_r
        e_va_it = (p_va_it * classes[None, :]).sum(1)
        e_te_it = (p_te_it * classes[None, :]).sum(1)
        bt = tune_thresholds(e_va_it, yva)
        yp_te_it = apply_thresholds(e_te_it, bt["t1"], bt["t2"], bt["t3"])
        m_it = metrics(yte, yp_te_it)
        pl_history.append({"round": it, "n_pseudo": n_sel, "w_unif": wu_it,
                            "thresholds": [bt["t1"], bt["t2"], bt["t3"]],
                            **m_it})
        print(f"  test κ_q={m_it['kappa_q']:.3f} {m_it['kappa_q_ci']}  (was {last_kq:.3f})")

        last_yhat_te = yp_te_it
        last_conf = p_te_it.max(1)
        last_kq = m_it["kappa_q"]

    out["pseudolabel_rounds"] = pl_history

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
