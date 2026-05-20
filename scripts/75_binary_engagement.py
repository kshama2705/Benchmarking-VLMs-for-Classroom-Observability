"""
Binary engagement reformulation.

Collapse 4-class engagement (0/1/2/3) into binary tasks:
  - L23 vs L01 (typical 'engaged' vs 'not engaged' split)
  - L3 vs L012 (only highly-engaged vs rest)
  - L0 vs L123 (only not-engaged vs rest)

Test if frozen SigLIP-L LR probe achieves much higher κ on these coarser tasks.

Also: ordinal-as-regression evaluation (Pearson r on integer labels).

Output:
  results/sota/binary_engagement.json
"""
import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
OUT = os.path.join(BASE, "results", "sota", "binary_engagement.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)


def boot_kq(yt, yp, n=1000, binary=False):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        if binary:
            out.append(cohen_kappa_score(yt[idx], yp[idx]))
        else:
            out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def fit_lr_binary(Xtr, ytr, Xva, yva):
    best = None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva))
        if best is None or v > best["v"]:
            best = {"C": C, "v": float(v), "clf": clf}
    return best["clf"], best["v"], best["C"]


def metrics_binary(yt, yp, p_pos=None):
    res = {
        "n": int(len(yt)),
        "accuracy": float(accuracy_score(yt, yp)),
        "kappa_unweighted": float(cohen_kappa_score(yt, yp)),
        "kappa_ci95": boot_kq(yt, yp, binary=True),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "pred_dist": {0: int((yp == 0).sum()), 1: int((yp == 1).sum())},
        "true_dist": {0: int((yt == 0).sum()), 1: int((yt == 1).sum())},
    }
    if p_pos is not None:
        try:
            res["auc"] = float(roc_auc_score(yt, p_pos))
        except Exception:
            res["auc"] = None
    return res


def bagged_binary(Xtr, ytr, Xva, yva, Xte, yte, K=30, seeds=[0, 7, 42, 2025, 1024]):
    all_p_te = []
    all_p_va = []
    for outer_seed in seeds:
        rng = np.random.default_rng(outer_seed)
        bag_te = []
        bag_va = []
        for _ in range(K):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            clf, _, _ = fit_lr_binary(Xtr[idx], ytr[idx], Xva, yva)
            bag_te.append(clf.predict_proba(Xte))
            bag_va.append(clf.predict_proba(Xva))
        all_p_te.append(np.stack(bag_te).mean(axis=0))
        all_p_va.append(np.stack(bag_va).mean(axis=0))
    return np.stack(all_p_te).mean(axis=0), np.stack(all_p_va).mean(axis=0)


def main():
    print("Loading SigLIP-L features...")
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    splits = d["split"]
    eng = d["engagement"].astype(np.int64)
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    Xtr, ytr = X[tr], eng[tr]
    Xva, yva = X[va], eng[va]
    Xte, yte = X[te], eng[te]

    out = {}

    print("\n=== Class distribution ===")
    print(f"  Train: {np.bincount(ytr, minlength=4).tolist()}")
    print(f"  Val:   {np.bincount(yva, minlength=4).tolist()}")
    print(f"  Test:  {np.bincount(yte, minlength=4).tolist()}")

    binary_splits = [
        ("L23_vs_L01", lambda y: (y >= 2).astype(int)),
        ("L3_vs_L012", lambda y: (y >= 3).astype(int)),
        ("L0_vs_L123", lambda y: (y == 0).astype(int)),
    ]

    for name, fn in binary_splits:
        print(f"\n=== Binary split: {name} ===")
        ytr_b, yva_b, yte_b = fn(ytr), fn(yva), fn(yte)
        print(f"  Train pos rate: {ytr_b.mean():.3f}  Val: {yva_b.mean():.3f}  Test: {yte_b.mean():.3f}")
        if yte_b.sum() == 0 or yte_b.sum() == len(yte_b):
            print(f"  SKIP: test set has only one class")
            continue
        # Solo
        clf, val_v, C = fit_lr_binary(Xtr, ytr_b, Xva, yva_b)
        yhat = clf.predict(Xte)
        p_pos = clf.predict_proba(Xte)[:, 1]
        m = metrics_binary(yte_b, yhat, p_pos)
        out[f"{name}_solo"] = {**m, "C": C, "val_kappa": val_v}
        print(f"  solo  acc={m['accuracy']:.3f}  κ={m['kappa_unweighted']:.3f} {m['kappa_ci95']}  "
              f"F1={m['f1_macro']:.3f}  AUC={m.get('auc', 'n/a'):.3f}")
        # Bagged
        p_te, p_va = bagged_binary(Xtr, ytr_b, Xva, yva_b, Xte, yte_b)
        yhat_bag = p_te.argmax(axis=1)
        m_bag = metrics_binary(yte_b, yhat_bag, p_te[:, 1])
        out[f"{name}_bagged"] = m_bag
        print(f"  bagged acc={m_bag['accuracy']:.3f}  κ={m_bag['kappa_unweighted']:.3f} {m_bag['kappa_ci95']}  "
              f"F1={m_bag['f1_macro']:.3f}  AUC={m_bag.get('auc', 'n/a'):.3f}")

    # Also: 4-class ordinal as regression (Pearson r)
    print("\n=== 4-class ordinal as regression (Pearson r) ===")
    from scipy.stats import pearsonr, spearmanr
    best = None
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(Xtr, ytr.astype(np.float32))
        v = pearsonr(yva, reg.predict(Xva))[0]
        if best is None or v > best["v"]:
            best = {"a": alpha, "v": float(v), "reg": reg}
    pred_te = best["reg"].predict(Xte)
    pr = float(pearsonr(yte, pred_te)[0])
    sr = float(spearmanr(yte, pred_te)[0])
    out["ridge_regression"] = {"alpha": best["a"], "pearson_r": pr, "spearman_r": sr, "val_pearson": best["v"]}
    print(f"  Ridge α={best['a']}  Pearson r={pr:.3f}  Spearman r={sr:.3f}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
