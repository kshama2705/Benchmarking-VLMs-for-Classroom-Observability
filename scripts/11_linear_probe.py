"""
Subject-disjoint linear probe on CLIP ViT-B/32 features for DAiSEE engagement.

Uses official DAiSEE splits (subject-disjoint by construction):
  - fit on Train, select hyperparams on Validation
  - report on full 1,784-clip Test split

Models:
  - LogReg (categorical) with class-balanced weights
  - Ridge regression on integer labels with threshold-decoded prediction
    (an ordinal-aware variant: predicts a continuous score, then rounds to {0,1,2,3})

Metrics: accuracy, Cohen's quadratic-weighted κ, macro-F1, MSE, per-class recall,
prediction distribution. Bootstrap 95% CIs (1000 resamples) on Test.

Output:
  results/linear_probe/probe_results.json
  results/linear_probe/probe_test_predictions.csv
"""

import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score, mean_squared_error,
    confusion_matrix, classification_report,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_FILE = os.path.join(BASE_DIR, "features", "clip_vitb32_features.npz")
OUT_DIR = os.path.join(BASE_DIR, "results", "linear_probe")
RNG = np.random.default_rng(42)


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa_quadratic": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "kappa_linear": float(cohen_kappa_score(y_true, y_pred, weights="linear")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "pred_dist": {int(k): int((y_pred == k).sum()) for k in range(4)},
        "true_dist": {int(k): int((y_true == k).sum()) for k in range(4)},
    }


def bootstrap_ci(y_true, y_pred, n=1000, alpha=0.05):
    """1000-iter bootstrap 95% CIs over test indices."""
    accs, kqs, kls, f1s, mses = [], [], [], [], []
    n_test = len(y_true)
    for _ in range(n):
        idx = RNG.integers(0, n_test, size=n_test)
        yt, yp = y_true[idx], y_pred[idx]
        accs.append(accuracy_score(yt, yp))
        kqs.append(cohen_kappa_score(yt, yp, weights="quadratic"))
        kls.append(cohen_kappa_score(yt, yp, weights="linear"))
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
        mses.append(mean_squared_error(yt, yp))

    def ci(arr):
        lo = float(np.quantile(arr, alpha / 2))
        hi = float(np.quantile(arr, 1 - alpha / 2))
        return [lo, hi]

    return {
        "accuracy_ci95": ci(accs),
        "kappa_quadratic_ci95": ci(kqs),
        "kappa_linear_ci95": ci(kls),
        "f1_macro_ci95": ci(f1s),
        "mse_ci95": ci(mses),
    }


def fit_logreg(Xtr, ytr, Xva, yva, Xte, yte):
    """Logistic regression with class-balanced weights; sweep C on validation."""
    best = None
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(
            C=C, class_weight="balanced", max_iter=5000,
            solver="lbfgs", random_state=42,
        )
        clf.fit(Xtr, ytr)
        yhat_va = clf.predict(Xva)
        score = cohen_kappa_score(yva, yhat_va, weights="quadratic")
        if best is None or score > best["val_kappa"]:
            best = {"C": C, "val_kappa": float(score), "clf": clf}
    yhat_te = best["clf"].predict(Xte)
    yhat_va = best["clf"].predict(Xva)
    res = {
        "method": "logreg_balanced",
        "best_C": best["C"],
        "val_metrics": metrics(yva, yhat_va),
        "test_metrics": metrics(yte, yhat_te),
        "test_metrics_ci": bootstrap_ci(yte, yhat_te),
    }
    return res, yhat_te


def fit_ridge_ordinal(Xtr, ytr, Xva, yva, Xte, yte):
    """Ridge regression on integer labels; predict by clamping+rounding to {0,1,2,3}."""
    best = None
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(Xtr, ytr.astype(np.float32))
        raw_va = reg.predict(Xva)
        yhat_va = np.clip(np.round(raw_va), 0, 3).astype(int)
        score = cohen_kappa_score(yva, yhat_va, weights="quadratic")
        if best is None or score > best["val_kappa"]:
            best = {"alpha": alpha, "val_kappa": float(score), "reg": reg}
    raw_te = best["reg"].predict(Xte)
    yhat_te = np.clip(np.round(raw_te), 0, 3).astype(int)
    raw_va = best["reg"].predict(Xva)
    yhat_va = np.clip(np.round(raw_va), 0, 3).astype(int)
    res = {
        "method": "ridge_ordinal",
        "best_alpha": best["alpha"],
        "val_metrics": metrics(yva, yhat_va),
        "test_metrics": metrics(yte, yhat_te),
        "test_metrics_ci": bootstrap_ci(yte, yhat_te),
    }
    return res, yhat_te


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = np.load(FEAT_FILE, allow_pickle=True)
    splits = data["split"]
    y = data["engagement"]
    X = data["feat"]
    clip_ids = data["clip_id"]
    subjects = data["subject_id"]

    tr = splits == "Train"
    va = splits == "Validation"
    te = splits == "Test"
    print(f"Train: {tr.sum()}  Val: {va.sum()}  Test: {te.sum()}")
    print(f"Subject overlap train∩test: {len(set(subjects[tr]) & set(subjects[te]))}")
    print(f"Subject overlap val∩test:   {len(set(subjects[va]) & set(subjects[te]))}")

    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    Xte, yte = X[te], y[te]
    cte = clip_ids[te]

    print("\n=== LogReg (class-balanced) ===")
    res_lr, yhat_lr = fit_logreg(Xtr, ytr, Xva, yva, Xte, yte)
    print(json.dumps(res_lr, indent=2))

    print("\n=== Ridge (ordinal) ===")
    res_rd, yhat_rd = fit_ridge_ordinal(Xtr, ytr, Xva, yva, Xte, yte)
    print(json.dumps(res_rd, indent=2))

    out = {
        "n_train": int(tr.sum()),
        "n_val": int(va.sum()),
        "n_test": int(te.sum()),
        "subject_overlap_train_test": len(set(subjects[tr]) & set(subjects[te])),
        "subject_overlap_val_test": len(set(subjects[va]) & set(subjects[te])),
        "logreg": res_lr,
        "ridge_ordinal": res_rd,
    }
    with open(os.path.join(OUT_DIR, "probe_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    import csv as _csv
    with open(os.path.join(OUT_DIR, "probe_test_predictions.csv"), "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["clip_id", "ground_truth", "logreg_pred", "ridge_pred"])
        for cid, yt, ylr, yrd in zip(cte, yte, yhat_lr, yhat_rd):
            w.writerow([cid, int(yt), int(ylr), int(yrd)])

    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    main()
