"""
Validate random-subspace-bagging SOTA (frac=0.5, κ=0.228) is robust:

1. Finer frac sweep: 0.35, 0.40, 0.45, 0.50, 0.55, 0.60
2. More seeds: 10 outer seeds (vs 5 before)
3. K=40 bags per seed (vs 30)
4. Report per-seed, per-frac, and mega-ensemble κ

Output:
  results/sota/sota_validation.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
OUT = os.path.join(BASE, "results", "sota", "sota_validation.json")
LOG = os.path.join(BASE, "results", "sota", "sota_validation_status.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)

OUTER_SEEDS = [0, 1, 2, 7, 42, 123, 1024, 2025, 31337, 999]
K_BAGS = 40
FRACS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


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
    open(LOG, "w").close()
    log("Loading SigLIP-L features...")
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    splits = d["split"]; eng = d["engagement"].astype(np.int64)
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    Xtr = X[tr]; ytr = eng[tr]
    Xva = X[va]; yva = eng[va]
    Xte = X[te]; yte = eng[te]
    D = X.shape[1]; n_tr = len(ytr)

    out = {"K_bags": K_BAGS, "outer_seeds": OUTER_SEEDS, "per_frac": {}}
    log(f"K={K_BAGS}, seeds={len(OUTER_SEEDS)}, fracs={FRACS}")

    for frac in FRACS:
        n_features = max(1, int(D * frac))
        log(f"\n=== frac={frac:.2f} (n_features={n_features}) ===")
        all_seed_p_te = []
        per_seed_kappas = []
        for outer_seed in OUTER_SEEDS:
            rng = np.random.default_rng(outer_seed)
            seed_probs = []
            for k in range(K_BAGS):
                row_idx = rng.integers(0, n_tr, size=n_tr)
                col_idx = rng.choice(D, size=n_features, replace=False)
                Xtr_sub = Xtr[row_idx][:, col_idx]
                Xva_sub = Xva[:, col_idx]
                Xte_sub = Xte[:, col_idx]
                clf, _, _ = fit_lr(Xtr_sub, ytr[row_idx], Xva_sub, yva)
                seed_probs.append(clf.predict_proba(Xte_sub))
            seed_mean = np.stack(seed_probs).mean(axis=0)
            all_seed_p_te.append(seed_mean)
            kq_seed = float(cohen_kappa_score(yte, seed_mean.argmax(axis=1), weights="quadratic"))
            per_seed_kappas.append(kq_seed)
            log(f"  seed {outer_seed}: κ_q = {kq_seed:.3f}")

        # Sub-ensembles: first N seeds
        sub_results = {}
        for n_use in [3, 5, 7, 10]:
            if n_use > len(all_seed_p_te):
                continue
            mega = np.stack(all_seed_p_te[:n_use]).mean(axis=0)
            yhat = mega.argmax(axis=1)
            m = metrics(yte, yhat)
            sub_results[f"sub_{n_use}seeds"] = m
            log(f"  sub-ensemble {n_use} seeds: κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}")

        per_seed_arr = np.array(per_seed_kappas)
        out["per_frac"][str(frac)] = {
            "per_seed_kappas": per_seed_kappas,
            "per_seed_mean": float(per_seed_arr.mean()),
            "per_seed_std": float(per_seed_arr.std(ddof=1)),
            "per_seed_min": float(per_seed_arr.min()),
            "per_seed_max": float(per_seed_arr.max()),
            "sub_ensembles": sub_results,
        }

    # Summary
    log("\n=== Summary ===")
    for frac in FRACS:
        f_data = out["per_frac"][str(frac)]
        sub10 = f_data["sub_ensembles"].get("sub_10seeds", {})
        log(f"  frac={frac:.2f}  per-seed mean={f_data['per_seed_mean']:.3f} ± {f_data['per_seed_std']:.3f}  "
            f"10-seed mega κ_q={sub10.get('kappa_q', 0):.3f}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {OUT}")
    log("SOTA_VALIDATION_DONE")


if __name__ == "__main__":
    main()
