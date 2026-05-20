"""
Use binary L0-vs-rest detector to override predictions on the current SOTA pred dist.

Current SOTA (solo LR-unif + threshold): 0/111/848/825 predictions vs true ~54/196/945/589.
NO L0 predictions at all — model misses every L0 case. With quadratic kappa,
true=L0 predicted=L3 squared error = 9 (penalty), but predicted=L0 = 0.
Recovering even 30 L0 cases correctly could lift κ meaningfully.

Plan:
1. Train binary L0-vs-rest LR on SigLIP-L features (bagged, K=20 × 5 seeds)
2. Get p(L0) on val + test
3. Tune (val) an L0 override threshold τ: if p(L0) > τ, override current SOTA prediction to 0
4. Evaluate on test

Output:
  results/sota/l0_override.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
L0_CACHE = os.path.join(BASE, "results", "sota", "_l0_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "l0_override.json")
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


def fit_lr_binary(Xtr, ytr, Xva, yva):
    """Binary LR with C-sweep, picks C maximizing val AUC."""
    best = None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xva)[:, 1]
        try:
            v = roc_auc_score(yva, p)
        except Exception:
            v = 0
        if best is None or v > best["v"]:
            best = {"C": C, "v": float(v), "clf": clf}
    return best["clf"], best["v"]


def bag_lr_binary(Xtr, ytr, Xva, yva, Xte, K=20, seeds=[0, 7, 42, 2025, 1024]):
    all_te, all_va = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        bt, bv = [], []
        for k in range(K):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            clf, auc = fit_lr_binary(Xtr[idx], ytr[idx], Xva, yva)
            bt.append(clf.predict_proba(Xte)[:, 1])
            bv.append(clf.predict_proba(Xva)[:, 1])
            if (k + 1) % 10 == 0:
                print(f"    seed={s} k={k+1}/{K} auc={auc:.3f}", flush=True)
        all_te.append(np.stack(bt).mean(0)); all_va.append(np.stack(bv).mean(0))
    return np.stack(all_te).mean(0), np.stack(all_va).mean(0)


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


def apply_thresh(e, t):
    yp = np.zeros_like(e, dtype=int)
    yp[e > t[0]] = 1; yp[e > t[1]] = 2; yp[e > t[2]] = 3
    return yp


def main():
    print("Loading SigLIP-L features + cached LR-unif bag probs...", flush=True)
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    Xtr, ytr = X[tr], eng[tr]; Xva, yva = X[va], eng[va]; Xte, yte = X[te], eng[te]

    c = np.load(CACHE)
    p_va_unif = c["p_va_unif"]; p_te_unif = c["p_te_unif"]
    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    e_va_unif = (p_va_unif * classes[None, :]).sum(1)
    e_te_unif = (p_te_unif * classes[None, :]).sum(1)

    # Re-derive SOTA baseline thresholds
    b_sota = tune_thresholds(e_va_unif, yva, step=0.02)
    yhat_te_sota = apply_thresh(e_te_unif, b_sota["t"])
    m_sota = metrics(yte, yhat_te_sota)
    print(f"\nSOTA baseline: κ_q={m_sota['kappa_q']:.4f} {m_sota['kappa_q_ci']}", flush=True)
    print(f"  pred_dist: {m_sota['pred_dist']}", flush=True)

    out = {"sota_baseline": m_sota}

    # Train binary L0-vs-rest bag
    if os.path.exists(L0_CACHE):
        l0 = np.load(L0_CACHE)
        p_va_l0 = l0["p_va_l0"]; p_te_l0 = l0["p_te_l0"]
        print(f"\nL0 detector loaded from cache: te shape={p_te_l0.shape}", flush=True)
    else:
        print("\nTraining binary L0-vs-rest bag (K=20 × 5 seeds)...", flush=True)
        ytr_b = (ytr == 0).astype(int)
        yva_b = (yva == 0).astype(int)
        t0 = time.time()
        p_te_l0, p_va_l0 = bag_lr_binary(Xtr, ytr_b, Xva, yva_b, Xte, K=20)
        print(f"  L0 bag done ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(L0_CACHE, p_va_l0=p_va_l0, p_te_l0=p_te_l0)

    # Compute AUC for sanity
    yte_l0 = (yte == 0).astype(int)
    yva_l0 = (yva == 0).astype(int)
    try:
        auc_te = roc_auc_score(yte_l0, p_te_l0)
        auc_va = roc_auc_score(yva_l0, p_va_l0)
        print(f"  AUC val={auc_va:.3f}  AUC test={auc_te:.3f}", flush=True)
    except Exception:
        print("  AUC failed", flush=True)

    # Tune override threshold τ on val: maximize κ when overriding SOTA pred → 0 if p(L0) > τ
    yhat_va_sota = apply_thresh(e_va_unif, b_sota["t"])
    print("\nTuning L0 override threshold τ on val...", flush=True)
    best_tau = None
    for tau in np.arange(0.05, 0.99, 0.01):
        yhat_va_override = yhat_va_sota.copy()
        yhat_va_override[p_va_l0 > tau] = 0
        v = cohen_kappa_score(yva, yhat_va_override, weights="quadratic")
        if best_tau is None or v > best_tau["v"]:
            best_tau = {"tau": float(tau), "v": float(v), "n_overrides_va": int((p_va_l0 > tau).sum())}
    tau = best_tau["tau"]
    print(f"  Best τ={tau:.3f}  val κ={best_tau['v']:.4f}  overrides={best_tau['n_overrides_va']}/1429", flush=True)

    # Apply to test
    yhat_te_override = yhat_te_sota.copy()
    n_overrides_te = int((p_te_l0 > tau).sum())
    yhat_te_override[p_te_l0 > tau] = 0
    m_override = metrics(yte, yhat_te_override)
    out["l0_override"] = {**m_override, "tau": tau, "val_kq": best_tau["v"], "n_overrides_te": n_overrides_te}
    print(f"\nL0 override on test: κ_q={m_override['kappa_q']:.4f} {m_override['kappa_q_ci']}", flush=True)
    print(f"  test overrides: {n_overrides_te}/1784 → pred_dist={m_override['pred_dist']}", flush=True)

    # Also try: tune (t1, t2, t3, tau) jointly via random search
    print("\nJoint random search (t1, t2, t3, tau)...", flush=True)
    rng = np.random.default_rng(42)
    best_j = None
    for _ in range(30000):
        ts = sorted(rng.uniform(0, 3, size=3))
        tau_j = rng.uniform(0.05, 0.99)
        yp_va = apply_thresh(e_va_unif, ts)
        yp_va[p_va_l0 > tau_j] = 0
        v = cohen_kappa_score(yva, yp_va, weights="quadratic")
        if best_j is None or v > best_j["v"]:
            best_j = {"t": ts, "tau": float(tau_j), "v": float(v)}
    yp_te_j = apply_thresh(e_te_unif, best_j["t"])
    n_ov = int((p_te_l0 > best_j["tau"]).sum())
    yp_te_j[p_te_l0 > best_j["tau"]] = 0
    m_j = metrics(yte, yp_te_j)
    out["joint_override"] = {**m_j, "t": best_j["t"], "tau": best_j["tau"], "val_kq": best_j["v"], "n_overrides_te": n_ov}
    print(f"  joint: t=({best_j['t'][0]:.2f},{best_j['t'][1]:.2f},{best_j['t'][2]:.2f}) τ={best_j['tau']:.3f}", flush=True)
    print(f"  κ_q={m_j['kappa_q']:.4f} {m_j['kappa_q_ci']}  val={best_j['v']:.4f}  overrides={n_ov}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
