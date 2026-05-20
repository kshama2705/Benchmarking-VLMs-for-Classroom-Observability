"""
MOONSHOT 33: Subject-level classifier trained on OOF train + val per-subject stats.

Use the OOF train predictions (from script 104) to compute per-subject features
on train, combine with val features (using all labeled subjects: 70 train + 22 val = 92).
Train a classifier to predict subject modal class. Apply to test.

Output:
  results/sota/moonshot_oof_subject_clf.json
"""
import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
OOF_CACHE = os.path.join(BASE, "results", "sota", "_oof_stacking_cache.npz")
BAG_CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_oof_subject_clf.json")
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


def subj_features(e_arr, subj_arr):
    """Compute per-subject feature stats."""
    feats = {}
    for s in np.unique(subj_arr):
        mask = subj_arr == s
        e_s = e_arr[mask]
        feats[s] = {
            "mean": float(e_s.mean()),
            "std": float(e_s.std()),
            "median": float(np.median(e_s)),
            "min": float(e_s.min()),
            "max": float(e_s.max()),
            "q25": float(np.quantile(e_s, 0.25)),
            "q75": float(np.quantile(e_s, 0.75)),
            "iqr": float(np.quantile(e_s, 0.75) - np.quantile(e_s, 0.25)),
            "n": int(mask.sum()),
        }
    return feats


def main():
    print("Loading...", flush=True)
    d = np.load(SIGLIP, allow_pickle=True)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    subj = d["subject_id"]
    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    ytr = eng[tr]; yva = eng[va]; yte = eng[te]
    subj_tr = subj[tr]; subj_va = subj[va]; subj_te = subj[te]

    classes = np.array([0, 1, 2, 3], dtype=np.float32)

    # OOF train predictions (from script 104)
    if not os.path.exists(OOF_CACHE):
        print("ERROR: OOF cache missing. Run script 104 first.", flush=True)
        return
    o = np.load(OOF_CACHE)
    oof_tr = o["oof_s"]  # OOF train SigLIP-L LR predictions
    p_va = o["va_s"]
    p_te = o["te_s"]
    e_tr = (oof_tr * classes[None, :]).sum(1)
    e_va = (p_va * classes[None, :]).sum(1)
    e_te = (p_te * classes[None, :]).sum(1)

    # Per-subject feature stats
    tr_feats = subj_features(e_tr, subj_tr)
    va_feats = subj_features(e_va, subj_va)
    te_feats = subj_features(e_te, subj_te)

    # Per-subject true label (modal/mean)
    tr_y = {s: int(np.round(ytr[subj_tr == s].mean())) for s in np.unique(subj_tr)}
    va_y = {s: int(np.round(yva[subj_va == s].mean())) for s in np.unique(subj_va)}
    te_y = {s: int(np.round(yte[subj_te == s].mean())) for s in np.unique(subj_te)}

    print(f"Train subjects: {len(tr_feats)}", flush=True)
    print(f"Val subjects: {len(va_feats)}", flush=True)
    print(f"Test subjects: {len(te_feats)}", flush=True)
    print(f"Train subj class dist: {np.bincount([tr_y[s] for s in tr_y], minlength=4).tolist()}", flush=True)
    print(f"Val subj class dist: {np.bincount([va_y[s] for s in va_y], minlength=4).tolist()}", flush=True)
    print(f"Test subj class dist: {np.bincount([te_y[s] for s in te_y], minlength=4).tolist()}", flush=True)

    feat_keys = ["mean", "std", "median", "min", "max", "q25", "q75", "iqr", "n"]

    # Build feature matrices (train + val combined for fitting)
    train_subjects = list(tr_feats.keys())
    val_subjects = list(va_feats.keys())
    test_subjects = list(te_feats.keys())

    Xtr_s = np.array([[tr_feats[s][k] for k in feat_keys] for s in train_subjects])
    Xva_s = np.array([[va_feats[s][k] for k in feat_keys] for s in val_subjects])
    Xte_s = np.array([[te_feats[s][k] for k in feat_keys] for s in test_subjects])
    ytr_s = np.array([tr_y[s] for s in train_subjects])
    yva_s = np.array([va_y[s] for s in val_subjects])

    # Combined train+val
    X_all = np.concatenate([Xtr_s, Xva_s], axis=0)
    y_all = np.concatenate([ytr_s, yva_s], axis=0)
    print(f"\nCombined train+val: {X_all.shape}", flush=True)

    # Standardize features
    sc = StandardScaler().fit(X_all)
    X_all_s = sc.transform(X_all)
    Xte_s_norm = sc.transform(Xte_s)
    Xva_s_norm = sc.transform(Xva_s)

    out = {}

    # ============ A: Ridge regression on subject features ============
    print("\n[A] Ridge regression on train+val subject features...", flush=True)
    best_ridge = None
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        # Tune on val only (subjects from val are subset of X_all)
        # Use train subjects only for training, val for tuning
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(sc.transform(Xtr_s), ytr_s.astype(np.float32))
        pred_va = reg.predict(sc.transform(Xva_s))
        # MSE on val
        mse_va = np.mean((pred_va - yva_s) ** 2)
        if best_ridge is None or mse_va < best_ridge["mse"]:
            best_ridge = {"alpha": alpha, "mse": float(mse_va), "reg": reg}
    print(f"  Best alpha: {best_ridge['alpha']}  val MSE: {best_ridge['mse']:.3f}", flush=True)
    # Fit on ALL train+val with best alpha
    reg_final = Ridge(alpha=best_ridge["alpha"], random_state=42)
    reg_final.fit(X_all_s, y_all.astype(np.float32))
    pred_te_subj = reg_final.predict(Xte_s_norm)
    pred_va_subj = reg_final.predict(Xva_s_norm)

    # Tune thresholds on val SUBJECT predictions
    grid = np.arange(0, 3.01, 0.04)
    best_t = None
    for t1 in grid:
        for t2 in grid[grid > t1]:
            for t3 in grid[grid > t2]:
                yp = np.zeros_like(yva_s)
                yp[pred_va_subj > t1] = 1; yp[pred_va_subj > t2] = 2; yp[pred_va_subj > t3] = 3
                v = cohen_kappa_score(yva_s, yp, weights="quadratic")
                if best_t is None or v > best_t["v"]:
                    best_t = {"t": (t1, t2, t3), "v": v}
    print(f"  Subject-level thresholds: {best_t['t']}  val κ={best_t['v']:.3f}", flush=True)

    # Apply to test (per-subject prediction → broadcast to clips)
    yhat_te = np.zeros_like(yte)
    for i, s in enumerate(test_subjects):
        mask = subj_te == s
        sm = pred_te_subj[i]
        cp = 0
        if sm > best_t["t"][0]: cp = 1
        if sm > best_t["t"][1]: cp = 2
        if sm > best_t["t"][2]: cp = 3
        yhat_te[mask] = cp
    m = metrics(yte, yhat_te)
    out["ridge_trainval_subj"] = {**m, "alpha": best_ridge["alpha"], **best_t}
    print(f"  Test κ_q={m['kappa_q']:.4f} {m['kappa_q_ci']}", flush=True)

    # ============ B: Use only mean feature with train+val thresholds ============
    print("\n[B] Use only mean E[y] (just like baseline), but threshold-tuned on train+val combined...", flush=True)
    all_means = np.concatenate([Xtr_s[:, 0], Xva_s[:, 0]])  # idx 0 = mean
    all_y = y_all
    best_t = None
    for t1 in grid:
        for t2 in grid[grid > t1]:
            for t3 in grid[grid > t2]:
                yp = np.zeros_like(all_y)
                yp[all_means > t1] = 1; yp[all_means > t2] = 2; yp[all_means > t3] = 3
                v = cohen_kappa_score(all_y, yp, weights="quadratic")
                if best_t is None or v > best_t["v"]:
                    best_t = {"t": (t1, t2, t3), "v": v}
    print(f"  Thresholds (train+val): {best_t['t']}  trainval κ={best_t['v']:.3f}", flush=True)
    # Apply to test
    te_means = Xte_s[:, 0]
    yhat_te = np.zeros_like(yte)
    for i, s in enumerate(test_subjects):
        mask = subj_te == s
        sm = te_means[i]
        cp = 0
        if sm > best_t["t"][0]: cp = 1
        if sm > best_t["t"][1]: cp = 2
        if sm > best_t["t"][2]: cp = 3
        yhat_te[mask] = cp
    m = metrics(yte, yhat_te)
    out["mean_trainval_thresh"] = {**m, **best_t}
    print(f"  Test κ_q={m['kappa_q']:.4f} {m['kappa_q_ci']}", flush=True)

    # Also use cached LR-unif bag (not OOF) for test E[y]
    print("\n[C] Use cached LR-unif bag for test E[y] + train+val thresholds (from B)", flush=True)
    bc = np.load(BAG_CACHE)
    p_te_lr = bc["p_te_unif"]
    e_te_lr = (p_te_lr * classes[None, :]).sum(1)
    te_subj_means_lr = {s: e_te_lr[subj_te == s].mean() for s in test_subjects}
    yhat_te = np.zeros_like(yte)
    for s in test_subjects:
        mask = subj_te == s
        sm = te_subj_means_lr[s]
        cp = 0
        if sm > best_t["t"][0]: cp = 1
        if sm > best_t["t"][1]: cp = 2
        if sm > best_t["t"][2]: cp = 3
        yhat_te[mask] = cp
    m = metrics(yte, yhat_te)
    out["cached_lr_trainval_thresh"] = {**m, **best_t}
    print(f"  Test κ_q={m['kappa_q']:.4f} {m['kappa_q_ci']}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
