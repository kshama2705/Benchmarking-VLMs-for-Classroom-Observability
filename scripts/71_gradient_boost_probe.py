"""
Gradient boosting probe on SigLIP-L features (different model class than LR).

sklearn's HistGradientBoostingClassifier — handles large feature dims, no libomp.

Bagged + multi-seed.

Output:
  results/sota/gradient_boost.json
"""
import os, json, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
OUT = os.path.join(BASE, "results", "sota", "gradient_boost.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)


def boot_kq(yt, yp, n=1000):
    out = []
    nt = len(yt)
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


def main():
    print("Loading SigLIP-L features...")
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    splits = d["split"]; eng = d["engagement"].astype(np.int64)
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    Xtr, ytr = X[tr], eng[tr]; Xva, yva = X[va], eng[va]; Xte, yte = X[te], eng[te]

    out = {}

    # Single shot HistGBC, tune via val
    print("\n=== Single-shot HistGradientBoosting ===")
    best = None
    for max_iter in [100, 200, 300]:
        for lr in [0.05, 0.1]:
            for max_depth in [3, 5, 7]:
                clf = HistGradientBoostingClassifier(
                    max_iter=max_iter, learning_rate=lr, max_depth=max_depth,
                    class_weight="balanced", random_state=42)
                clf.fit(Xtr, ytr)
                v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
                if best is None or v > best["v"]:
                    best = {"max_iter": max_iter, "lr": lr, "max_depth": max_depth,
                            "v": float(v), "clf": clf}
    print(f"  best: max_iter={best['max_iter']} lr={best['lr']} depth={best['max_depth']}  val={best['v']:.3f}")
    yhat = best["clf"].predict(Xte)
    m = metrics(yte, yhat)
    out["single_shot"] = {**m, "hparams": {k: best[k] for k in ["max_iter", "lr", "max_depth"]}, "val_kq": best["v"]}
    print(f"  Test κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}")

    # Bagged HistGBC (multi-seed × K bags)
    print("\n=== Bagged HistGBC (5 seeds × K=20 bags) ===")
    OUTER_SEEDS = [0, 7, 42, 2025, 1024]
    K = 20
    all_seed_p_te = []
    for outer_seed in OUTER_SEEDS:
        rng = np.random.default_rng(outer_seed)
        bag_probs = []
        for k in range(K):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            clf = HistGradientBoostingClassifier(
                max_iter=best["max_iter"], learning_rate=best["lr"],
                max_depth=best["max_depth"], class_weight="balanced",
                random_state=int(outer_seed * 1000 + k))
            clf.fit(Xtr[idx], ytr[idx])
            bag_probs.append(clf.predict_proba(Xte))
        seed_mean = np.stack(bag_probs).mean(axis=0)
        all_seed_p_te.append(seed_mean)
        kq_seed = float(cohen_kappa_score(yte, seed_mean.argmax(axis=1), weights="quadratic"))
        print(f"  seed {outer_seed}: κ_q={kq_seed:.3f}")
    mega = np.stack(all_seed_p_te).mean(axis=0)
    yhat = mega.argmax(axis=1)
    m = metrics(yte, yhat)
    out["bagged_mega"] = m
    print(f"  Mega: κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}")

    # GBC + LR fusion
    print("\n=== HistGBC + LR fusion ===")
    # Bagged LR probabilities (re-derive)
    lr_probs_te = []
    lr_probs_va = []
    for outer_seed in OUTER_SEEDS:
        rng = np.random.default_rng(outer_seed)
        bag_te = []; bag_va = []
        for k in range(K):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            best_lr = None
            for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
                clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                         solver="lbfgs", random_state=42)
                clf.fit(Xtr[idx], ytr[idx])
                v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
                if best_lr is None or v > best_lr["v"]:
                    best_lr = {"v": v, "clf": clf}
            bag_te.append(best_lr["clf"].predict_proba(Xte))
            bag_va.append(best_lr["clf"].predict_proba(Xva))
        lr_probs_te.append(np.stack(bag_te).mean(axis=0))
        lr_probs_va.append(np.stack(bag_va).mean(axis=0))
    lr_mega_te = np.stack(lr_probs_te).mean(axis=0)
    lr_mega_va = np.stack(lr_probs_va).mean(axis=0)

    # GBC val predictions (for weight search)
    gbc_probs_va = []
    for outer_seed in OUTER_SEEDS:
        rng = np.random.default_rng(outer_seed)
        bag_va = []
        for k in range(K):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            clf = HistGradientBoostingClassifier(
                max_iter=best["max_iter"], learning_rate=best["lr"],
                max_depth=best["max_depth"], class_weight="balanced",
                random_state=int(outer_seed * 1000 + k))
            clf.fit(Xtr[idx], ytr[idx])
            bag_va.append(clf.predict_proba(Xva))
        gbc_probs_va.append(np.stack(bag_va).mean(axis=0))
    gbc_mega_va = np.stack(gbc_probs_va).mean(axis=0)
    gbc_mega_te = mega  # from earlier

    best_w = None
    for w in np.linspace(0.0, 1.0, 41):
        pv = w * gbc_mega_va + (1 - w) * lr_mega_va
        v = cohen_kappa_score(yva, pv.argmax(axis=1), weights="quadratic")
        if best_w is None or v > best_w["v"]:
            best_w = {"w": float(w), "v": float(v)}
    wb = best_w["w"]
    p_fuse = wb * gbc_mega_te + (1 - wb) * lr_mega_te
    yhat = p_fuse.argmax(axis=1)
    m = metrics(yte, yhat)
    out["GBC+LR_fusion"] = {**m, "w": wb, "val_kq": best_w["v"]}
    print(f"  GBC+LR fusion (w={wb:.2f}): κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  val={best_w['v']:.3f}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
