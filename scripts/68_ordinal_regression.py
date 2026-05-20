"""
Ordinal regression for engagement — exploits the ordered nature of {0,1,2,3}.

Approaches:
  1. Ridge regression on integer labels (already tried) — basic
  2. Cumulative logit model (a.k.a. proportional odds) — proper ordinal
  3. CORN-style binary decomposition (3 binary classifiers: >0, >1, >2)
  4. Bagged version of (3)

Output:
  results/sota/ordinal_regression.json
"""
import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
OUT = os.path.join(BASE, "results", "sota", "ordinal_regression.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)


def boot_kq(yt, yp, n=1000):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def fit_lr_binary(Xtr, ytr_bin, Xva, yva_bin):
    best = None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr_bin)
        # Predict positive-class probability
        p_va = clf.predict_proba(Xva)[:, 1]
        # Use accuracy on val as criterion for binary decision
        v = ((p_va > 0.5).astype(int) == yva_bin).mean()
        if best is None or v > best["v"]:
            best = {"C": C, "v": float(v), "clf": clf}
    return best["clf"], best["v"]


def corn_predict(thresh_clfs, X):
    """Given list of binary classifiers (>0, >1, >2), produce integer prediction."""
    n = len(thresh_clfs)
    cum_probs = np.zeros((X.shape[0], n))  # P(y > k) for k=0..n-1
    for k, clf in enumerate(thresh_clfs):
        cum_probs[:, k] = clf.predict_proba(X)[:, 1]
    # Convert cumulative to per-class probabilities (assumes proper ordering)
    # P(y=0) = 1 - P(y>0)
    # P(y=1) = P(y>0) - P(y>1)
    # P(y=2) = P(y>1) - P(y>2)
    # P(y=3) = P(y>2)
    p = np.zeros((X.shape[0], n + 1))
    p[:, 0] = 1 - cum_probs[:, 0]
    for k in range(1, n):
        p[:, k] = cum_probs[:, k - 1] - cum_probs[:, k]
    p[:, n] = cum_probs[:, n - 1]
    # Clip negatives (proportional-odds is not enforced)
    p = np.clip(p, 0, None)
    p = p / p.sum(axis=1, keepdims=True).clip(min=1e-6)
    return p


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
    Xtr, ytr = X[tr], eng[tr]
    Xva, yva = X[va], eng[va]
    Xte, yte = X[te], eng[te]

    out = {}

    # Standard LR baseline (re-derive)
    print("\n=== Standard LR baseline ===")
    best = None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": v, "clf": clf}
    yhat = best["clf"].predict(Xte)
    out["baseline_LR"] = {**metrics(yte, yhat), "C": best["C"]}
    print(f"  baseline LR  κ_q={out['baseline_LR']['kappa_q']:.3f} {out['baseline_LR']['kappa_q_ci']}")
    p_lr_te = best["clf"].predict_proba(Xte)

    # CORN-style binary decomposition (single shot)
    print("\n=== CORN binary decomposition ===")
    thresh_clfs = []
    for thr in [0, 1, 2]:
        ytr_bin = (ytr > thr).astype(int)
        yva_bin = (yva > thr).astype(int)
        clf, vacc = fit_lr_binary(Xtr, ytr_bin, Xva, yva_bin)
        thresh_clfs.append(clf)
        print(f"  threshold y>{thr}: val_acc={vacc:.3f}")
    p_corn_te = corn_predict(thresh_clfs, Xte)
    yhat_corn = p_corn_te.argmax(axis=1)
    out["CORN_single"] = metrics(yte, yhat_corn)
    print(f"  CORN single  κ_q={out['CORN_single']['kappa_q']:.3f} {out['CORN_single']['kappa_q_ci']}")

    # CORN bagged
    print("\n=== CORN bagged ===")
    OUTER_SEEDS = [0, 7, 42, 2025, 1024]
    K_BAGS = 30
    all_p_te = []
    for outer_seed in OUTER_SEEDS:
        rng = np.random.default_rng(outer_seed)
        bag_probs = []
        for k in range(K_BAGS):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            t_clfs = []
            for thr in [0, 1, 2]:
                ytr_bin = (ytr[idx] > thr).astype(int)
                yva_bin = (yva > thr).astype(int)
                clf, _ = fit_lr_binary(Xtr[idx], ytr_bin, Xva, yva_bin)
                t_clfs.append(clf)
            bag_probs.append(corn_predict(t_clfs, Xte))
        seed_mean = np.stack(bag_probs).mean(axis=0)
        all_p_te.append(seed_mean)
        kq_seed = float(cohen_kappa_score(yte, seed_mean.argmax(axis=1), weights="quadratic"))
        print(f"  seed {outer_seed}: κ_q={kq_seed:.3f}")
    mega = np.stack(all_p_te).mean(axis=0)
    yhat = mega.argmax(axis=1)
    out["CORN_bagged_mega"] = metrics(yte, yhat)
    print(f"  CORN bagged mega κ_q={out['CORN_bagged_mega']['kappa_q']:.3f} {out['CORN_bagged_mega']['kappa_q_ci']}")

    # CORN + LR fusion (late)
    print("\n=== CORN + LR baseline fusion ===")
    # Get bagged LR predictions
    all_p_lr_te = []
    all_p_lr_va = []
    p_lr_va = best["clf"].predict_proba(Xva)  # we already have single-shot LR predictions
    for outer_seed in OUTER_SEEDS:
        rng = np.random.default_rng(outer_seed + 1000)
        bag_te = []; bag_va = []
        for k in range(K_BAGS):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            best_c = None
            for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
                clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                         solver="lbfgs", random_state=42)
                clf.fit(Xtr[idx], ytr[idx])
                v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
                if best_c is None or v > best_c["v"]:
                    best_c = {"C": C, "v": v, "clf": clf}
            bag_te.append(best_c["clf"].predict_proba(Xte))
            bag_va.append(best_c["clf"].predict_proba(Xva))
        all_p_lr_te.append(np.stack(bag_te).mean(axis=0))
        all_p_lr_va.append(np.stack(bag_va).mean(axis=0))
    p_lr_bagged_te = np.stack(all_p_lr_te).mean(axis=0)
    p_lr_bagged_va = np.stack(all_p_lr_va).mean(axis=0)

    p_corn_bagged_va = np.zeros_like(p_lr_bagged_va)
    for outer_seed in OUTER_SEEDS:
        rng = np.random.default_rng(outer_seed)
        bag_va = []
        for k in range(K_BAGS):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            t_clfs = []
            for thr in [0, 1, 2]:
                ytr_bin = (ytr[idx] > thr).astype(int)
                yva_bin = (yva > thr).astype(int)
                clf, _ = fit_lr_binary(Xtr[idx], ytr_bin, Xva, yva_bin)
                t_clfs.append(clf)
            bag_va.append(corn_predict(t_clfs, Xva))
        p_corn_bagged_va += np.stack(bag_va).mean(axis=0)
    p_corn_bagged_va /= len(OUTER_SEEDS)
    p_corn_bagged_te = np.stack(all_p_te).mean(axis=0)

    # Sweep weight
    best_w = None
    for w in np.linspace(0.0, 1.0, 41):
        pv = w * p_corn_bagged_va + (1 - w) * p_lr_bagged_va
        v = cohen_kappa_score(yva, pv.argmax(axis=1), weights="quadratic")
        if best_w is None or v > best_w["v"]:
            best_w = {"w": float(w), "v": float(v)}
    wb = best_w["w"]
    p_fuse = wb * p_corn_bagged_te + (1 - wb) * p_lr_bagged_te
    yhat = p_fuse.argmax(axis=1)
    m = metrics(yte, yhat)
    out["CORN+LR_bagged_fusion"] = {**m, "w": wb, "val_kq": best_w["v"]}
    print(f"  CORN+LR bagged (w={wb:.2f}) κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  val={best_w['v']:.3f}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
