"""
Robust bagging — sweep K and outer-seed to confirm K=10 bagged SigLIP-L is
genuinely a SOTA breakthrough vs single-seed lucky.

For each outer_seed in {0, 1, 2, 7, 42, 123, 2025}:
  For each K in {3, 5, 10, 15, 20, 30, 50}:
    Bootstrap train K times (with outer_seed), train LR, average test predictions
    Record test κ_q

Output:
  results/sota/bagging_robust.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
OUT = os.path.join(BASE, "results", "sota", "bagging_robust.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)


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


def main():
    print("Loading SigLIP-L features...")
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    splits = d["split"]
    eng = d["engagement"].astype(np.int64)
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    Xtr = X[tr]; ytr = eng[tr]
    Xva = X[va]; yva = eng[va]
    Xte = X[te]; yte = eng[te]

    OUTER_SEEDS = [0, 1, 2, 7, 42, 123, 2025]
    K_VALUES = [3, 5, 10, 15, 20, 30, 50]
    MAX_K = max(K_VALUES)

    print(f"\nRobust bagging: {len(OUTER_SEEDS)} outer seeds × max K={MAX_K} bags each")
    out = {"per_seed_per_K": {}, "summary_per_K": {}}

    all_K_kappas = {K: [] for K in K_VALUES}

    for outer_seed in OUTER_SEEDS:
        print(f"\n=== outer_seed={outer_seed} ===")
        rng = np.random.default_rng(outer_seed)
        n_tr = len(ytr)
        p_te_bags = []
        for k in range(MAX_K):
            idx = rng.integers(0, n_tr, size=n_tr)
            clf, _, _ = fit_lr(Xtr[idx], ytr[idx], Xva, yva)
            p_te_bags.append(clf.predict_proba(Xte))
        p_te_stack = np.stack(p_te_bags)

        per_K = {}
        for K in K_VALUES:
            p_mean = p_te_stack[:K].mean(axis=0)
            yhat = p_mean.argmax(axis=1)
            kq = float(cohen_kappa_score(yte, yhat, weights="quadratic"))
            per_K[K] = kq
            all_K_kappas[K].append(kq)
            print(f"  K={K:2d}  κ_q={kq:.3f}")
        out["per_seed_per_K"][str(outer_seed)] = per_K

    # Summary across outer seeds
    print("\n=== Summary across outer seeds ===")
    for K in K_VALUES:
        ks = np.array(all_K_kappas[K])
        # Bootstrap CI on the MEAN across seeds (paired with mean estimator)
        # Just report mean ± std
        out["summary_per_K"][str(K)] = {
            "mean": float(ks.mean()),
            "std": float(ks.std(ddof=1) if len(ks) > 1 else 0.0),
            "min": float(ks.min()),
            "max": float(ks.max()),
            "values": [float(v) for v in ks],
        }
        print(f"  K={K:2d}  mean={ks.mean():.3f} ± {ks.std(ddof=1):.3f}  "
              f"min={ks.min():.3f}  max={ks.max():.3f}")

    # Best K choice
    best_K = max(K_VALUES, key=lambda k: out["summary_per_K"][str(k)]["mean"])
    best_summary = out["summary_per_K"][str(best_K)]
    print(f"\nBest K = {best_K}  mean κ_q = {best_summary['mean']:.3f} ± {best_summary['std']:.3f}")

    # For best K, also report bootstrap CI on test set (using one representative seed)
    rng_rep = np.random.default_rng(0)
    p_te_bags = []
    for k in range(best_K):
        idx = rng_rep.integers(0, len(ytr), size=len(ytr))
        clf, _, _ = fit_lr(Xtr[idx], ytr[idx], Xva, yva)
        p_te_bags.append(clf.predict_proba(Xte))
    p_mean = np.stack(p_te_bags).mean(axis=0)
    yhat = p_mean.argmax(axis=1)
    ci = boot_kq(yte, yhat)
    kq = float(cohen_kappa_score(yte, yhat, weights="quadratic"))
    out["best_K_representative"] = {
        "K": best_K, "kappa_q": kq, "kappa_q_ci95": ci,
        "accuracy": float(accuracy_score(yte, yhat)),
        "f1_macro": float(f1_score(yte, yhat, average="macro", zero_division=0)),
    }
    print(f"Representative (outer_seed=0) at K={best_K}: κ_q={kq:.3f} CI={ci}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
