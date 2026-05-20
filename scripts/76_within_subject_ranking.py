"""
Within-subject ranking analysis.

For each test subject:
  - Get all their test clips
  - Predict engagement scores (or Ridge regression value)
  - Compute Spearman correlation with true engagement
  - Compute AUC for binary "high engagement" within their clips

If frozen features rank a single subject's clips well, we have a deployable
signal even if cross-subject thresholds don't work.

Output:
  results/sota/within_subject_ranking.json
"""
import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from scipy.stats import spearmanr, pearsonr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
OUT = os.path.join(BASE, "results", "sota", "within_subject_ranking.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)


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


def fit_ridge(Xtr, ytr, Xva, yva):
    best = None
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(Xtr, ytr.astype(np.float32))
        v = pearsonr(yva, reg.predict(Xva))[0]
        if best is None or v > best["v"]:
            best = {"alpha": alpha, "v": float(v), "reg": reg}
    return best["reg"], best["v"], best["alpha"]


def main():
    print("Loading SigLIP-L features...")
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    splits = d["split"]; eng = d["engagement"].astype(np.int64)
    subj = d["subject_id"]
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    Xtr, ytr = X[tr], eng[tr]; Xva, yva = X[va], eng[va]; Xte, yte = X[te], eng[te]
    subj_te = subj[te]

    # Bagged predictions on test (already-validated SOTA method, frac=1.0 vanilla)
    print("\nTraining 5-seed × 30-bag SigLIP-L LR for ranking scores...")
    OUTER_SEEDS = [0, 7, 42, 2025, 1024]; K = 30
    all_p_te = []
    expected_te = np.zeros(len(yte), dtype=np.float32)
    for outer_seed in OUTER_SEEDS:
        rng = np.random.default_rng(outer_seed)
        seed_probs = []
        for k in range(K):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            clf, _, _ = fit_lr(Xtr[idx], ytr[idx], Xva, yva)
            seed_probs.append(clf.predict_proba(Xte))
        all_p_te.append(np.stack(seed_probs).mean(axis=0))
    p_te_mean = np.stack(all_p_te).mean(axis=0)
    # Continuous engagement score: expected class
    classes = np.array([0, 1, 2, 3])
    expected_te = (p_te_mean * classes[None, :]).sum(axis=1)
    print(f"  Mean expected score: {expected_te.mean():.3f}")

    # Also Ridge regression (continuous output)
    print("\nTraining Ridge regression...")
    reg, ridge_val, ridge_alpha = fit_ridge(Xtr, ytr, Xva, yva)
    ridge_pred = reg.predict(Xte)
    print(f"  Ridge α={ridge_alpha} val pearson={ridge_val:.3f}")

    # Global metrics (sanity)
    yhat = p_te_mean.argmax(axis=1)
    print(f"\nGlobal κ_q (4-class bagged): {cohen_kappa_score(yte, yhat, weights='quadratic'):.3f}")
    print(f"Global Pearson r (Ridge): {pearsonr(yte, ridge_pred)[0]:.3f}")
    print(f"Global Spearman r (expected score): {spearmanr(yte, expected_te)[0]:.3f}")

    # Per-subject analysis
    print("\n=== Per-subject ranking analysis ===")
    subjects = np.unique(subj_te)
    out = {"n_subjects": int(len(subjects)), "per_subject": [], "global": {
        "kappa_q_bagged": float(cohen_kappa_score(yte, yhat, weights="quadratic")),
        "pearson_ridge": float(pearsonr(yte, ridge_pred)[0]),
        "spearman_expected": float(spearmanr(yte, expected_te)[0]),
    }}

    spearmans = []
    pearsons = []
    aucs_high = []
    aucs_low = []
    label_var = []
    for s in subjects:
        mask = (subj_te == s)
        n = mask.sum()
        y_s = yte[mask]
        pred_s = expected_te[mask]
        ridge_s = ridge_pred[mask]
        # Need at least 2 distinct labels to compute correlation
        if len(np.unique(y_s)) < 2 or n < 3:
            continue
        sp = spearmanr(y_s, pred_s)[0]
        pr = pearsonr(y_s, ridge_s)[0]
        # Binary AUC: y >= 2 ('engaged')
        if (y_s >= 2).sum() > 0 and (y_s < 2).sum() > 0:
            auc_h = roc_auc_score((y_s >= 2).astype(int), pred_s)
        else:
            auc_h = None
        # Binary AUC: y == 0 (not engaged)
        if (y_s == 0).sum() > 0 and (y_s != 0).sum() > 0:
            auc_l = roc_auc_score((y_s == 0).astype(int), -pred_s)  # negate so high prob = positive for "y==0"
        else:
            auc_l = None
        out["per_subject"].append({
            "subject": str(s),
            "n_clips": int(n),
            "label_std": float(y_s.std()),
            "spearman": float(sp) if not np.isnan(sp) else None,
            "pearson": float(pr) if not np.isnan(pr) else None,
            "auc_high_engagement": float(auc_h) if auc_h is not None else None,
            "auc_low_engagement": float(auc_l) if auc_l is not None else None,
        })
        if not np.isnan(sp): spearmans.append(sp)
        if not np.isnan(pr): pearsons.append(pr)
        if auc_h is not None: aucs_high.append(auc_h)
        if auc_l is not None: aucs_low.append(auc_l)
        label_var.append(y_s.std())

    out["summary"] = {
        "n_subjects_analyzed": len(spearmans),
        "mean_spearman": float(np.mean(spearmans)) if spearmans else None,
        "median_spearman": float(np.median(spearmans)) if spearmans else None,
        "mean_pearson": float(np.mean(pearsons)) if pearsons else None,
        "mean_auc_high": float(np.mean(aucs_high)) if aucs_high else None,
        "mean_auc_low": float(np.mean(aucs_low)) if aucs_low else None,
        "mean_label_std": float(np.mean(label_var)) if label_var else None,
    }
    print(f"\n=== Summary across {len(spearmans)} subjects with variation ===")
    print(f"  Mean Spearman r: {out['summary']['mean_spearman']:.3f}")
    print(f"  Median Spearman r: {out['summary']['median_spearman']:.3f}")
    print(f"  Mean Pearson r:  {out['summary']['mean_pearson']:.3f}")
    print(f"  Mean AUC (high engagement): {out['summary']['mean_auc_high']:.3f}")
    print(f"  Mean AUC (low engagement):  {out['summary']['mean_auc_low']:.3f}")
    print(f"  Mean label std (within subject): {out['summary']['mean_label_std']:.3f}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
