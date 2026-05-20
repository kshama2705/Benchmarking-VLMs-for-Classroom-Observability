"""
MOONSHOT 17: Mahalanobis distance classifier.

Generative approach: model each class as Gaussian N(μ_k, Σ_k) in feature space.
Classify by min Mahalanobis distance to class mean.

  d_k(x) = (x - μ_k)^T Σ_k^{-1} (x - μ_k)
  p(y=k|x) ∝ p(x|y=k) p(y=k) ≈ exp(-0.5 d_k(x)) (assuming common covariance)

With ZCA whitening + shared covariance, this is equivalent to LDA.
Different from LR: LR is discriminative, Mahalanobis is generative. May
find different decision boundaries.

For L0 (rare), the small sample makes per-class covariance unstable. Use
shrinkage to global covariance.

Output:
  results/sota/moonshot_mahalanobis.json
"""
import os, json, time
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.covariance import ShrunkCovariance, LedoitWolf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_mahalanobis.json")
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

    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    out = {}

    # LDA with shrinkage
    print("\n[A] LDA with shrinkage (auto)...", flush=True)
    for shrink in ["auto", 0.1, 0.5, 0.9]:
        try:
            if shrink == "auto":
                lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            else:
                lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrink)
            lda.fit(Xtr, ytr)
            p_te = lda.predict_proba(Xte)
            p_va = lda.predict_proba(Xva)
            m_solo = metrics(yte, p_te.argmax(1))
            e_va = (p_va * classes[None, :]).sum(1); e_te = (p_te * classes[None, :]).sum(1)
            bt = tune_thresh(e_va, yva)
            m_thr = metrics(yte, apply_t(e_te, bt["t"]))
            out[f"lda_shrink_{shrink}"] = {"solo": m_solo, "threshold": {**m_thr, **bt}}
            print(f"  shrink={shrink}: solo κ={m_solo['kappa_q']:.4f}  +thresh κ={m_thr['kappa_q']:.4f} {m_thr['kappa_q_ci']}", flush=True)
        except Exception as e:
            print(f"  shrink={shrink}: failed {e}", flush=True)

    # Bagged LDA (with shrinkage auto)
    print("\n[B] Bagged LDA (K=20 × 5 seeds, shrink=auto)...", flush=True)
    seeds = [0, 7, 42, 2025, 1024]; K = 20
    all_te, all_va = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        bt_, bv_ = [], []
        for k in range(K):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            lda.fit(Xtr[idx], ytr[idx])
            bt_.append(lda.predict_proba(Xte))
            bv_.append(lda.predict_proba(Xva))
        all_te.append(np.stack(bt_).mean(0)); all_va.append(np.stack(bv_).mean(0))
        print(f"  seed={s} done", flush=True)
    p_te = np.stack(all_te).mean(0); p_va = np.stack(all_va).mean(0)
    m_solo = metrics(yte, p_te.argmax(1))
    e_va = (p_va * classes[None, :]).sum(1); e_te = (p_te * classes[None, :]).sum(1)
    bt = tune_thresh(e_va, yva)
    m_thr = metrics(yte, apply_t(e_te, bt["t"]))
    out["bagged_lda_threshold"] = {**m_thr, **bt}
    print(f"  Bagged LDA: solo κ={m_solo['kappa_q']:.4f}  +thresh κ={m_thr['kappa_q']:.4f} {m_thr['kappa_q_ci']}", flush=True)

    # Cache for fusion
    np.savez_compressed(os.path.join(BASE, "results", "sota", "_lda_bag_cache.npz"),
                        p_te=p_te, p_va=p_va)

    # Fusion with LR-unif
    print("\n[C] Fusion: bagged LDA + cached LR-unif...", flush=True)
    c = np.load(CACHE)
    p_te_lr = c["p_te_unif"]; p_va_lr = c["p_va_unif"]
    best_w = None
    for w in np.linspace(0, 1, 21):
        pv = w * p_va_lr + (1 - w) * p_va
        e = (pv * classes[None, :]).sum(1)
        bt2 = tune_thresh(e, yva)
        if best_w is None or bt2["v"] > best_w["v"]:
            best_w = {"w_lr": float(w), **bt2}
    wl = best_w["w_lr"]
    p_te_f = wl * p_te_lr + (1 - wl) * p_te
    e_te_f = (p_te_f * classes[None, :]).sum(1)
    m_f = metrics(yte, apply_t(e_te_f, best_w["t"]))
    out["fusion_lr_lda"] = {**m_f, "w_lr": wl, **best_w}
    print(f"  Fusion: w_lr={wl:.2f}  κ_q={m_f['kappa_q']:.4f} {m_f['kappa_q_ci']}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
