"""
MOONSHOT 18: PCA + Polynomial features + LR bag.

Linear LR misses feature interactions. Polynomial features capture them.
On 1024-d this is intractable (524k features). Apply PCA to 50-100 dims first,
then polynomial degree 2 = 5050 features. Bagged LR + threshold.

Output:
  results/sota/moonshot_pca_poly.json
"""
import os, json, time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_pca_poly.json")
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
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
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
    print("Loading SigLIP-L features...", flush=True)
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    Xtr, ytr = X[tr], eng[tr]; Xva, yva = X[va], eng[va]; Xte, yte = X[te], eng[te]

    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    out = {}

    for n_pca in [50, 100]:
        for poly_deg in [1, 2]:
            name = f"pca{n_pca}_poly{poly_deg}"
            print(f"\n=== {name} ===", flush=True)
            t0 = time.time()
            sc = StandardScaler().fit(Xtr)
            Xtr_s = sc.transform(Xtr); Xva_s = sc.transform(Xva); Xte_s = sc.transform(Xte)
            pca = PCA(n_components=n_pca, random_state=42).fit(Xtr_s)
            Xtr_p = pca.transform(Xtr_s); Xva_p = pca.transform(Xva_s); Xte_p = pca.transform(Xte_s)
            print(f"  PCA explained var: {pca.explained_variance_ratio_.sum():.3f}", flush=True)

            if poly_deg > 1:
                pf = PolynomialFeatures(degree=poly_deg, include_bias=False, interaction_only=False)
                Xtr_f = pf.fit_transform(Xtr_p)
                Xva_f = pf.transform(Xva_p)
                Xte_f = pf.transform(Xte_p)
                print(f"  Poly{poly_deg} dim: {Xtr_f.shape[1]}", flush=True)
                # Re-standardize after poly
                sc2 = StandardScaler().fit(Xtr_f)
                Xtr_f = sc2.transform(Xtr_f); Xva_f = sc2.transform(Xva_f); Xte_f = sc2.transform(Xte_f)
            else:
                Xtr_f = Xtr_p; Xva_f = Xva_p; Xte_f = Xte_p

            p_te, p_va = bag_lr(Xtr_f.astype(np.float32), ytr,
                                Xva_f.astype(np.float32), yva,
                                Xte_f.astype(np.float32), K=20)
            print(f"  Bag done ({time.time()-t0:.0f}s)", flush=True)
            m_solo = metrics(yte, p_te.argmax(1))
            e_va = (p_va * classes[None, :]).sum(1); e_te = (p_te * classes[None, :]).sum(1)
            bt = tune_thresh(e_va, yva)
            m_thr = metrics(yte, apply_t(e_te, bt["t"]))
            out[name] = {"solo": m_solo, "threshold": {**m_thr, **bt}}
            print(f"  Solo κ={m_solo['kappa_q']:.4f}  +thresh κ={m_thr['kappa_q']:.4f} {m_thr['kappa_q_ci']}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
