"""
Random feature subspace bagging — random forests on top of SigLIP-L LR.

Each "tree" (LR probe) sees:
  - Bootstrap-resampled training samples
  - A random subset of feature dimensions (e.g., 50%, 70%)

Tests whether feature-randomness adds diversity beyond sample-randomness alone.

Output:
  results/sota/random_subspace_bagging.json
"""
import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
OUT = os.path.join(BASE, "results", "sota", "random_subspace_bagging.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)

OUTER_SEEDS = [0, 7, 42, 2025, 1024]
K_BAGS = 30
SUBSPACE_FRACS = [1.0, 0.8, 0.5, 0.3]  # 1.0 = no feature subsampling (= regular bagging)


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
    Xtr = X[tr]; ytr = eng[tr]
    Xva = X[va]; yva = eng[va]
    Xte = X[te]; yte = eng[te]
    D = X.shape[1]
    n_tr = len(ytr)

    out = {}
    for frac in SUBSPACE_FRACS:
        n_features = max(1, int(D * frac))
        print(f"\n=== Subspace frac={frac:.1f} (using {n_features}/{D} features) ===")
        all_p_te = []
        for outer_seed in OUTER_SEEDS:
            rng = np.random.default_rng(outer_seed)
            seed_probs = []
            for k in range(K_BAGS):
                # Bootstrap rows
                row_idx = rng.integers(0, n_tr, size=n_tr)
                # Random feature subset (only if frac < 1)
                if frac < 1.0:
                    col_idx = rng.choice(D, size=n_features, replace=False)
                    Xtr_sub = Xtr[row_idx][:, col_idx]
                    Xva_sub = Xva[:, col_idx]
                    Xte_sub = Xte[:, col_idx]
                else:
                    Xtr_sub = Xtr[row_idx]
                    Xva_sub = Xva
                    Xte_sub = Xte
                clf, _, _ = fit_lr(Xtr_sub, ytr[row_idx], Xva_sub, yva)
                seed_probs.append(clf.predict_proba(Xte_sub))
            seed_mean = np.stack(seed_probs).mean(axis=0)
            all_p_te.append(seed_mean)
            kq_seed = float(cohen_kappa_score(yte, seed_mean.argmax(axis=1), weights="quadratic"))
            print(f"  seed {outer_seed}: κ_q = {kq_seed:.3f}")

        # Mega
        mega = np.stack(all_p_te).mean(axis=0)
        yhat = mega.argmax(axis=1)
        m = metrics(yte, yhat)
        out[f"frac{frac:.1f}_mega"] = m
        print(f"  Mega (5 seeds × K={K_BAGS}) κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
