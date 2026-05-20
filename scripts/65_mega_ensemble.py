"""
Mega-ensemble: average predictions from S outer seeds × K bags = S*K LR probes.

If K=30 across 7 seeds gives mean 0.210 ± 0.013, then averaging 7×30=210 probes
might converge to the true mean with much smaller variance, potentially exceeding
the single-seed K=10 lucky result.

Output:
  results/sota/mega_ensemble.json
"""
import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
OUT = os.path.join(BASE, "results", "sota", "mega_ensemble.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)

OUTER_SEEDS = [0, 1, 2, 7, 42, 123, 2025, 999, 31337, 1024]
K_PER_SEED = 30


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
    print("Loading SigLIP-L features...")
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    splits = d["split"]; eng = d["engagement"].astype(np.int64)
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    Xtr, ytr = X[tr], eng[tr]
    Xva, yva = X[va], eng[va]
    Xte, yte = X[te], eng[te]

    n_total_probes = len(OUTER_SEEDS) * K_PER_SEED
    print(f"\nTraining {n_total_probes} LR probes ({len(OUTER_SEEDS)} seeds × {K_PER_SEED} bags)...")

    all_p_te = []
    n_tr = len(ytr)
    for s_idx, outer_seed in enumerate(OUTER_SEEDS):
        rng = np.random.default_rng(outer_seed)
        seed_probs = []
        for k in range(K_PER_SEED):
            idx = rng.integers(0, n_tr, size=n_tr)
            clf, _, _ = fit_lr(Xtr[idx], ytr[idx], Xva, yva)
            seed_probs.append(clf.predict_proba(Xte))
        seed_mean = np.stack(seed_probs).mean(axis=0)
        all_p_te.append(seed_mean)
        kq_seed = float(cohen_kappa_score(yte, seed_mean.argmax(axis=1), weights="quadratic"))
        print(f"  seed {outer_seed}: K={K_PER_SEED} mean κ_q = {kq_seed:.3f}")

    out = {"per_seed": {}, "mega": None}
    for s_idx, seed in enumerate(OUTER_SEEDS):
        kq = float(cohen_kappa_score(yte, all_p_te[s_idx].argmax(axis=1), weights="quadratic"))
        out["per_seed"][str(seed)] = {"K": K_PER_SEED, "kappa_q": kq}

    # Mega: average across all seeds (= average of seed-means)
    mega = np.stack(all_p_te).mean(axis=0)
    yhat = mega.argmax(axis=1)
    m = metrics(yte, yhat)
    out["mega"] = {**m, "n_probes": n_total_probes, "n_seeds": len(OUTER_SEEDS), "K_per_seed": K_PER_SEED}
    print(f"\n=== MEGA ENSEMBLE ({n_total_probes} probes total) ===")
    print(f"  κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  acc={m['accuracy']:.3f}")

    # Also try sub-ensembles: first N seeds at K=30
    for n_seeds in [2, 3, 5, 7, 10]:
        if n_seeds > len(all_p_te):
            continue
        sub_mean = np.stack(all_p_te[:n_seeds]).mean(axis=0)
        yhat_sub = sub_mean.argmax(axis=1)
        kq = float(cohen_kappa_score(yte, yhat_sub, weights="quadratic"))
        ci = boot_kq(yte, yhat_sub)
        out[f"sub_{n_seeds}seeds"] = {"kappa_q": kq, "kappa_q_ci": ci, "n_seeds": n_seeds}
        print(f"  Sub-ensemble {n_seeds} seeds (K={K_PER_SEED} each): κ_q={kq:.3f} {ci}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
