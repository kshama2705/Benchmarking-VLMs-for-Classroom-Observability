"""
PCA + bagging: project SigLIP-L features to lower dim via PCA fit on Train,
then bag LR probes on the PCA features. Tests whether PCA-based regularization
helps beyond random feature subspacing.

PCA dimensions tested: 64, 128, 256, 512.

Output:
  results/sota/pca_bagging.json
"""
import os, json, time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
OUT = os.path.join(BASE, "results", "sota", "pca_bagging.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)

OUTER_SEEDS = [0, 7, 42, 2025, 1024]
K_BAGS = 30
PCA_DIMS = [64, 128, 256, 512]


def boot_kq(yt, yp, n=1000):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def fit_lr(Xtr, ytr, Xva, yva):
    best = None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": float(v), "clf": clf}
    return best["clf"], best["v"], best["C"]


def metrics(yt, yp):
    return {
        "kappa_q": float(cohen_kappa_score(yt, yp, weights="quadratic")),
        "kappa_q_ci": boot_kq(yt, yp),
        "accuracy": float(accuracy_score(yt, yp)),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "pred_dist": {int(k): int((yp == k).sum()) for k in range(4)},
    }


def main():
    print("Loading...")
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    splits = d["split"]; eng = d["engagement"].astype(np.int64)
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    Xtr, ytr = X[tr], eng[tr]; Xva, yva = X[va], eng[va]; Xte, yte = X[te], eng[te]
    n_tr = len(ytr)

    out = {}
    for pca_dim in PCA_DIMS:
        print(f"\n=== PCA dim={pca_dim} ===")
        pca = PCA(n_components=pca_dim, random_state=42)
        Xtr_p = pca.fit_transform(Xtr).astype(np.float32)
        Xva_p = pca.transform(Xva).astype(np.float32)
        Xte_p = pca.transform(Xte).astype(np.float32)
        print(f"  PCA explained var: {pca.explained_variance_ratio_.sum():.3f}")

        all_seed_p_te = []
        per_seed_kappas = []
        for outer_seed in OUTER_SEEDS:
            rng = np.random.default_rng(outer_seed)
            seed_probs = []
            for k in range(K_BAGS):
                idx = rng.integers(0, n_tr, size=n_tr)
                clf, _, _ = fit_lr(Xtr_p[idx], ytr[idx], Xva_p, yva)
                seed_probs.append(clf.predict_proba(Xte_p))
            seed_mean = np.stack(seed_probs).mean(axis=0)
            all_seed_p_te.append(seed_mean)
            kq_seed = float(cohen_kappa_score(yte, seed_mean.argmax(axis=1), weights="quadratic"))
            per_seed_kappas.append(kq_seed)
            print(f"  seed {outer_seed}: κ_q={kq_seed:.3f}")

        mega = np.stack(all_seed_p_te).mean(axis=0)
        yhat = mega.argmax(axis=1)
        m = metrics(yte, yhat)
        per_seed_arr = np.array(per_seed_kappas)
        out[f"pca{pca_dim}"] = {
            **m,
            "per_seed_kappas": per_seed_kappas,
            "per_seed_mean": float(per_seed_arr.mean()),
            "per_seed_std": float(per_seed_arr.std(ddof=1)),
            "explained_var": float(pca.explained_variance_ratio_.sum()),
        }
        print(f"  Mega: κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  "
              f"per-seed mean={per_seed_arr.mean():.3f} ± {per_seed_arr.std(ddof=1):.3f}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
