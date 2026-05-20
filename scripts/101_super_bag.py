"""
MOONSHOT 12: Super-bag — K=50 × 10 seeds = 500 LR fits on SigLIP-L.

Just scale up the winner. If asymptotic, K=20×5=100 fits is enough. If not,
K=500 might find more.

Output:
  results/sota/moonshot_super_bag.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_super_bag.json")
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


def bag_lr_super(Xtr, ytr, Xva, yva, Xte, K=50, seeds=None):
    if seeds is None:
        seeds = [0, 7, 42, 2025, 1024, 100, 200, 300, 400, 500]
    n_seeds = len(seeds)
    all_te, all_va = [], []
    for s_i, s in enumerate(seeds):
        rng = np.random.default_rng(s)
        bt, bv = [], []
        for k in range(K):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            clf = fit_lr(Xtr[idx], ytr[idx], Xva, yva)
            bt.append(clf.predict_proba(Xte)); bv.append(clf.predict_proba(Xva))
            if (k+1) % 10 == 0:
                print(f"    seed {s_i+1}/{n_seeds} ({s}) k={k+1}/{K}", flush=True)
        all_te.append(np.stack(bt).mean(0)); all_va.append(np.stack(bv).mean(0))
        # Save partial each outer seed
        np.savez_compressed(os.path.join(BASE, "results", "sota", "_super_bag_partial.npz"),
                            p_te=np.stack(all_te).mean(0), p_va=np.stack(all_va).mean(0),
                            n_outer=s_i + 1)
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

    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    out = {}

    print("\nSuper-bag: K=50 × 10 seeds = 500 LR fits...", flush=True)
    t0 = time.time()
    p_te, p_va = bag_lr_super(Xtr, ytr, Xva, yva, Xte)
    print(f"  Done ({time.time()-t0:.0f}s)", flush=True)

    m_solo = metrics(yte, p_te.argmax(1))
    e_va = (p_va * classes[None, :]).sum(1); e_te = (p_te * classes[None, :]).sum(1)
    bt = tune_thresh(e_va, yva)
    m_thr = metrics(yte, apply_t(e_te, bt["t"]))
    out["solo"] = m_solo
    out["threshold"] = {**m_thr, **bt}
    print(f"\nSolo super-bag: κ_q={m_solo['kappa_q']:.4f}", flush=True)
    print(f"+threshold: κ_q={m_thr['kappa_q']:.4f} {m_thr['kappa_q_ci']}", flush=True)

    # Cache
    np.savez_compressed(os.path.join(BASE, "results", "sota", "_super_bag_cache.npz"),
                        p_te=p_te, p_va=p_va)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
