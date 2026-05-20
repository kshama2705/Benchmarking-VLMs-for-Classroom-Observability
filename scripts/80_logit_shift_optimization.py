"""
Post-hoc per-class logit shift optimization.

Take the current SOTA bagging output (LR + LR-RSB-45 weighted fusion) and add
per-class biases tuned on val to maximize κ_q. This is essentially logit
adjustment / Menon-style class-prior fix but optimized directly for the
ordinal metric rather than for log-prior matching.

Output:
  results/sota/logit_shift.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
OUT = os.path.join(BASE, "results", "sota", "logit_shift.json")
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


def bag_lr_uniform(Xtr, ytr, Xva, yva, Xte, K=30, seeds=[0, 7, 42, 2025, 1024]):
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


def bag_rsb(Xtr, ytr, Xva, yva, Xte, K=30, frac=0.45, seeds=[0, 7, 42, 2025, 1024]):
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


def random_search_bias(p_va, y_va, n_iter=20000, bias_max=1.0, seed=42):
    """Random search over (b0, b1, b2, b3) shifts in log-space. b3 fixed to 0 (rel)."""
    rng = np.random.default_rng(seed)
    log_p = np.log(p_va + 1e-12)
    best = {"v": cohen_kappa_score(y_va, p_va.argmax(1), weights="quadratic"), "b": np.zeros(4)}
    for _ in range(n_iter):
        b = rng.uniform(-bias_max, bias_max, size=4)
        b[3] = 0  # anchor
        adj = log_p + b[None, :]
        v = cohen_kappa_score(y_va, adj.argmax(1), weights="quadratic")
        if v > best["v"]:
            best = {"v": float(v), "b": b.copy()}
    return best


def coordinate_descent_bias(p_va, y_va, init=None, steps=80, eps=0.01):
    """Sweep each bias 1D after warm start."""
    log_p = np.log(p_va + 1e-12)
    b = init.copy() if init is not None else np.zeros(4)
    best_v = cohen_kappa_score(y_va, (log_p + b[None, :]).argmax(1), weights="quadratic")
    grid = np.arange(-2.0, 2.0 + eps, eps)
    for it in range(steps):
        improved = False
        for c in [0, 1, 2]:  # fix b[3]=anchor
            best_local = b[c]
            for cand in grid:
                bt = b.copy(); bt[c] = cand
                v = cohen_kappa_score(y_va, (log_p + bt[None, :]).argmax(1), weights="quadratic")
                if v > best_v:
                    best_v = float(v); best_local = float(cand); improved = True
            b[c] = best_local
        if not improved:
            break
    return b, best_v


def main():
    print("Loading SigLIP-L features...")
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    splits = d["split"]; eng = d["engagement"].astype(np.int64)
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    Xtr, ytr = X[tr], eng[tr]; Xva, yva = X[va], eng[va]; Xte, yte = X[te], eng[te]

    print("\n[1/2] Rebuild uniform + RSB-45 bags...")
    cache = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
    if os.path.exists(cache):
        c = np.load(cache)
        p_te_unif = c["p_te_unif"]; p_va_unif = c["p_va_unif"]
        p_te_rsb = c["p_te_rsb"]; p_va_rsb = c["p_va_rsb"]
        print(f"  loaded from cache: {cache}")
    else:
        t0 = time.time()
        p_te_unif, p_va_unif = bag_lr_uniform(Xtr, ytr, Xva, yva, Xte)
        print(f"  uniform done ({time.time()-t0:.0f}s)")
        t0 = time.time()
        p_te_rsb, p_va_rsb = bag_rsb(Xtr, ytr, Xva, yva, Xte, frac=0.45)
        print(f"  rsb-45 done ({time.time()-t0:.0f}s)")
        np.savez_compressed(cache, p_te_unif=p_te_unif, p_va_unif=p_va_unif,
                            p_te_rsb=p_te_rsb, p_va_rsb=p_va_rsb)
        print(f"  cached bags to {cache}")

    # Tune the fusion weight on val for w_unif vs w_rsb
    best_w = None
    for w in np.linspace(0, 1, 21):
        pv = w * p_va_unif + (1 - w) * p_va_rsb
        v = cohen_kappa_score(yva, pv.argmax(1), weights="quadratic")
        if best_w is None or v > best_w["v"]:
            best_w = {"w_unif": float(w), "v": float(v)}
    wu = best_w["w_unif"]
    p_va_fuse = wu * p_va_unif + (1 - wu) * p_va_rsb
    p_te_fuse = wu * p_te_unif + (1 - wu) * p_te_rsb
    print(f"  w_unif={wu:.2f}  val κ={best_w['v']:.3f}")
    m_base = metrics(yte, p_te_fuse.argmax(1))
    print(f"  baseline test κ_q={m_base['kappa_q']:.3f} {m_base['kappa_q_ci']}")

    out = {"baseline_fusion": {**m_base, "w_unif": wu, "val_kq": best_w["v"]}}

    print("\n[2/2] Logit shift optimization...")
    # Random search
    rs = random_search_bias(p_va_fuse, yva, n_iter=30000)
    print(f"  Random search: val κ={rs['v']:.3f} bias={rs['b']}")
    # Refine with coord descent
    b_opt, v_opt = coordinate_descent_bias(p_va_fuse, yva, init=rs["b"], steps=50, eps=0.005)
    print(f"  Coord descent: val κ={v_opt:.3f}  bias={b_opt}")

    # Apply to test
    log_te = np.log(p_te_fuse + 1e-12)
    yhat_te = (log_te + b_opt[None, :]).argmax(1)
    m_shift = metrics(yte, yhat_te)
    out["logit_shift"] = {**m_shift, "bias": b_opt.tolist(), "val_kq": float(v_opt)}
    print(f"  shifted test κ_q={m_shift['kappa_q']:.3f} {m_shift['kappa_q_ci']}")
    print(f"  pred_dist baseline={m_base['pred_dist']}  shifted={m_shift['pred_dist']}")

    # Also: ordinal-style threshold tuning on expected-class score
    print("\n[3/3] Ordinal threshold tuning on expected-class score...")
    classes = np.array([0, 1, 2, 3])
    e_va = (p_va_fuse * classes[None, :]).sum(1)
    e_te = (p_te_fuse * classes[None, :]).sum(1)

    # Grid search for thresholds t1 < t2 < t3 over [0, 3]
    grid = np.arange(0.0, 3.01, 0.02)
    best_t = None
    for t1 in grid:
        for t2 in grid[grid > t1]:
            for t3 in grid[grid > t2]:
                yp = np.zeros_like(e_va, dtype=int)
                yp[e_va > t1] = 1
                yp[e_va > t2] = 2
                yp[e_va > t3] = 3
                v = cohen_kappa_score(yva, yp, weights="quadratic")
                if best_t is None or v > best_t["v"]:
                    best_t = {"t1": float(t1), "t2": float(t2), "t3": float(t3), "v": float(v)}
    t1, t2, t3 = best_t["t1"], best_t["t2"], best_t["t3"]
    yp_te = np.zeros_like(e_te, dtype=int)
    yp_te[e_te > t1] = 1
    yp_te[e_te > t2] = 2
    yp_te[e_te > t3] = 3
    m_thr = metrics(yte, yp_te)
    out["ordinal_threshold"] = {**m_thr, "t1": t1, "t2": t2, "t3": t3, "val_kq": float(best_t["v"])}
    print(f"  thresholds: t1={t1:.2f} t2={t2:.2f} t3={t3:.2f}  val κ={best_t['v']:.3f}")
    print(f"  shifted test κ_q={m_thr['kappa_q']:.3f} {m_thr['kappa_q_ci']}")
    print(f"  pred_dist={m_thr['pred_dist']}")

    # Save probability arrays for downstream use
    np.savez_compressed(
        os.path.join(BASE, "results", "sota", "fusion_probs.npz"),
        p_va_fuse=p_va_fuse, p_te_fuse=p_te_fuse,
        p_va_unif=p_va_unif, p_te_unif=p_te_unif,
        p_va_rsb=p_va_rsb, p_te_rsb=p_te_rsb,
        yva=yva, yte=yte,
    )

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
