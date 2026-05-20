"""
MOONSHOT 3: Concatenate features from multiple encoders, single bag + threshold.

SigLIP-L (1024) + CLIP-L/14 (768) + DINOv2-B/14 (768) = 2560-d combined feature.
The hope: features from different encoders capture complementary axes.
Single LR-bag on concat may extract more than any individual fusion.

Output:
  results/sota/moonshot_concat.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CLIPL = os.path.join(BASE, "features", "daisee_clip_l_14_features.npz")
DINO = os.path.join(BASE, "features", "dinov2_vitb14_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_concat.json")
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


def bag_lr(Xtr, ytr, Xva, yva, Xte, K=20, seeds=[0, 7, 42, 2025, 1024]):
    all_te, all_va = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        bt, bv = [], []
        for k in range(K):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            clf = fit_lr(Xtr[idx], ytr[idx], Xva, yva)
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
    print("Loading encoders...", flush=True)
    d1 = np.load(SIGLIP, allow_pickle=True); Xs = d1["feat"].astype(np.float32)
    d2 = np.load(CLIPL, allow_pickle=True); Xc = d2["feat"].astype(np.float32)
    d3 = np.load(DINO, allow_pickle=True); Xd = d3["feat"].astype(np.float32)
    sp = d1["split"]; eng = d1["engagement"].astype(np.int64)
    print(f"  SigLIP-L: {Xs.shape}  CLIP-L: {Xc.shape}  DINOv2: {Xd.shape}", flush=True)

    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    ytr = eng[tr]; yva = eng[va]; yte = eng[te]

    # Standardize each encoder separately, then concat
    s1 = StandardScaler().fit(Xs[tr]); Xs_n = s1.transform(Xs).astype(np.float32)
    s2 = StandardScaler().fit(Xc[tr]); Xc_n = s2.transform(Xc).astype(np.float32)
    s3 = StandardScaler().fit(Xd[tr]); Xd_n = s3.transform(Xd).astype(np.float32)

    Xcat = np.concatenate([Xs_n, Xc_n, Xd_n], axis=1)
    print(f"  Concat dim: {Xcat.shape[1]}", flush=True)
    Xtr_cat = Xcat[tr]; Xva_cat = Xcat[va]; Xte_cat = Xcat[te]

    out = {}
    classes = np.array([0, 1, 2, 3], dtype=np.float32)

    print("\nBagging LR on 3-encoder concat (K=20 × 5 seeds)...", flush=True)
    t0 = time.time()
    p_te, p_va = bag_lr(Xtr_cat, ytr, Xva_cat, yva, Xte_cat, K=20)
    print(f"  Done ({time.time()-t0:.0f}s)", flush=True)

    m_solo = metrics(yte, p_te.argmax(1))
    e_va = (p_va * classes[None, :]).sum(1)
    e_te = (p_te * classes[None, :]).sum(1)
    bt = tune_thresh(e_va, yva)
    m_thr = metrics(yte, apply_t(e_te, bt["t"]))
    out["solo"] = m_solo
    out["threshold"] = {**m_thr, **bt}
    print(f"\nSolo concat: κ_q={m_solo['kappa_q']:.4f}", flush=True)
    print(f"+threshold: κ_q={m_thr['kappa_q']:.4f} {m_thr['kappa_q_ci']}", flush=True)

    # Cache
    np.savez_compressed(os.path.join(BASE, "results", "sota", "_concat_bag_cache.npz"),
                        p_te=p_te, p_va=p_va)

    # Fusion with cached LR-unif on SigLIP-L
    print("\nFusion with cached LR-unif (SigLIP-L only)...", flush=True)
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
    out["fusion_lr_concat"] = {**m_f, "w_lr": wl, **best_w}
    print(f"Fusion: w_lr={wl:.2f}  t={best_w['t']}  κ_q={m_f['kappa_q']:.4f} {m_f['kappa_q_ci']}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
