"""
MOONSHOT 16: HistGradientBoostingClassifier with threshold tuning.

Prior GBC (script 71) hit 0.188 raw. With threshold tuning, may approach LR.
Also test bagged GBC × seeds for variance reduction.

Output:
  results/sota/moonshot_gbc_thresh.json
"""
import os, json, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_gbc_thresh.json")
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


def fit_gbc(Xtr, ytr, Xva, yva):
    """Sweep a couple GBC configs, pick best val."""
    best = None
    for lr, max_iter, max_depth in [(0.05, 200, 6), (0.1, 100, 5), (0.05, 300, 4)]:
        clf = HistGradientBoostingClassifier(
            learning_rate=lr, max_iter=max_iter, max_depth=max_depth,
            class_weight="balanced", random_state=42, early_stopping=False,
        )
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"clf": clf, "v": float(v), "config": (lr, max_iter, max_depth)}
    return best["clf"], best["config"]


def bag_gbc(Xtr, ytr, Xva, yva, Xte, K=10, seeds=[0, 7, 42, 2025, 1024]):
    all_te, all_va = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        bt, bv = [], []
        for k in range(K):
            t0 = time.time()
            idx = rng.integers(0, len(ytr), size=len(ytr))
            clf, cfg = fit_gbc(Xtr[idx], ytr[idx], Xva, yva)
            bt.append(clf.predict_proba(Xte)); bv.append(clf.predict_proba(Xva))
            if (k+1) % 5 == 0:
                print(f"    seed={s} k={k+1}/{K} cfg={cfg} {time.time()-t0:.0f}s", flush=True)
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

    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    out = {}

    print("\nBagged GBC (K=10 × 5 seeds, 3 config sweep)...", flush=True)
    t0 = time.time()
    p_te, p_va = bag_gbc(Xtr, ytr, Xva, yva, Xte, K=10)
    print(f"  Done ({time.time()-t0:.0f}s)", flush=True)

    m_solo = metrics(yte, p_te.argmax(1))
    e_va = (p_va * classes[None, :]).sum(1); e_te = (p_te * classes[None, :]).sum(1)
    bt = tune_thresh(e_va, yva)
    m_thr = metrics(yte, apply_t(e_te, bt["t"]))
    out["solo"] = m_solo
    out["threshold"] = {**m_thr, **bt}
    print(f"\nSolo GBC: κ_q={m_solo['kappa_q']:.4f}", flush=True)
    print(f"+threshold: κ_q={m_thr['kappa_q']:.4f} {m_thr['kappa_q_ci']}", flush=True)

    # Cache
    np.savez_compressed(os.path.join(BASE, "results", "sota", "_gbc_bag_cache.npz"),
                        p_te=p_te, p_va=p_va)

    # Fusion with cached LR-unif
    print("\nFusion with LR-unif...", flush=True)
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
    out["fusion_lr_gbc"] = {**m_f, "w_lr": wl, **best_w}
    print(f"Fusion: w_lr={wl:.2f}  t={best_w['t']}  κ_q={m_f['kappa_q']:.4f} {m_f['kappa_q_ci']}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
