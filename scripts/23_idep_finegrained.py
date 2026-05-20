"""
N4: Fine-grained IDEP characterization.

For each k in {0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 69}:
  - Compute orthogonal projection onto orthogonal complement of top-k identity dirs
  - Measure post-projection subject-ID recoverability (within-train 80/20 split)
  - Measure engagement κ_q (LogReg + Ridge) on official subject-disjoint test
  - Report identity-engagement trade-off curve

Output:
  results/idep/idep_finegrained.json
  results/idep/idep_curve.csv  (k, id_acc, engagement_kq) for plotting
"""
import os, json, csv
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_squared_error

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "idep")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)

FEATS = {
    "CLIP_ViT-B32": os.path.join(BASE, "features", "clip_vitb32_features.npz"),
    "DINOv2_ViT-B14": os.path.join(BASE, "features", "dinov2_vitb14_features.npz"),
}

K_SWEEP = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 69]


def boot_kappa(yt, yp, n=1000):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def fit_id_classifier(X, y):
    return LogisticRegression(C=10.0, max_iter=5000, solver="lbfgs", random_state=42).fit(X, y)


def get_top_id_directions(W, k):
    """Return D x k matrix of top-k right-singular vectors of W (the directions
    along which identity is most linearly predictable)."""
    if k == 0:
        return np.zeros((W.shape[1], 0))
    U, S, Vt = np.linalg.svd(W.T, full_matrices=False)  # W.T is (D, K_subj)
    return U[:, :k]


def project_orth(X, U):
    """X @ (I - U U^T)"""
    if U.shape[1] == 0:
        return X
    return X - (X @ U) @ U.T


def post_id_acc(X_train, y_subj_train):
    """80/20 within-train clip split, train new linear classifier on residual features."""
    rng = np.random.default_rng(42)
    n_subj = int(y_subj_train.max() + 1)
    train_idx, test_idx = [], []
    for sid in range(n_subj):
        ci = np.where(y_subj_train == sid)[0]
        rng.shuffle(ci)
        n_t = int(0.8 * len(ci))
        train_idx.extend(ci[:n_t]); test_idx.extend(ci[n_t:])
    train_idx = np.array(train_idx); test_idx = np.array(test_idx)
    clf = LogisticRegression(C=10.0, max_iter=2000, solver="lbfgs", random_state=42)
    clf.fit(X_train[train_idx], y_subj_train[train_idx])
    return float(clf.score(X_train[test_idx], y_subj_train[test_idx]))


def engagement_probe(Xtr, ytr, Xva, yva, Xte, yte):
    best_lr = None
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best_lr is None or v > best_lr["v"]:
            best_lr = {"C": C, "v": v, "clf": clf}
    yhat_lr = best_lr["clf"].predict(Xte)
    kq_lr = float(cohen_kappa_score(yte, yhat_lr, weights="quadratic"))
    ci_lr = boot_kappa(yte, yhat_lr)

    best_rd = None
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(Xtr, ytr.astype(np.float32))
        yhat = np.clip(np.round(reg.predict(Xva)), 0, 3).astype(int)
        v = cohen_kappa_score(yva, yhat, weights="quadratic")
        if best_rd is None or v > best_rd["v"]:
            best_rd = {"a": alpha, "v": v, "reg": reg}
    yhat_rd = np.clip(np.round(best_rd["reg"].predict(Xte)), 0, 3).astype(int)
    kq_rd = float(cohen_kappa_score(yte, yhat_rd, weights="quadratic"))
    ci_rd = boot_kappa(yte, yhat_rd)

    return {"lr_kq": kq_lr, "lr_ci": ci_lr, "rd_kq": kq_rd, "rd_ci": ci_rd}


def run(label, path):
    print(f"\n========== {label} ==========")
    data = np.load(path, allow_pickle=True)
    splits = data["split"]; feats = data["feat"].astype(np.float32)
    eng = data["engagement"]; subj = data["subject_id"]
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"

    unique_subj = sorted(set(subj[tr].tolist()))
    sid_map = {s: i for i, s in enumerate(unique_subj)}
    y_subj = np.array([sid_map[s] for s in subj[tr]])

    # Fit identity classifier on full train
    id_clf = fit_id_classifier(feats[tr], y_subj)
    W = id_clf.coef_  # (n_subj, D)

    rows = []
    for k in K_SWEEP:
        U = get_top_id_directions(W, k)
        Ftr = project_orth(feats[tr], U)
        Fva = project_orth(feats[va], U)
        Fte = project_orth(feats[te], U)
        id_acc = post_id_acc(Ftr, y_subj)
        eng_res = engagement_probe(Ftr, eng[tr], Fva, eng[va], Fte, eng[te])
        rows.append({
            "k": k,
            "id_acc": id_acc,
            "lr_kq": eng_res["lr_kq"],
            "lr_ci_lo": eng_res["lr_ci"][0], "lr_ci_hi": eng_res["lr_ci"][1],
            "rd_kq": eng_res["rd_kq"],
            "rd_ci_lo": eng_res["rd_ci"][0], "rd_ci_hi": eng_res["rd_ci"][1],
        })
        print(f"  k={k:3d}  id_acc={id_acc:.3f}  "
              f"LR κ_q={eng_res['lr_kq']:.3f}[{eng_res['lr_ci'][0]:.3f},{eng_res['lr_ci'][1]:.3f}]  "
              f"Ridge κ_q={eng_res['rd_kq']:.3f}[{eng_res['rd_ci'][0]:.3f},{eng_res['rd_ci'][1]:.3f}]")
    return rows


def main():
    out = {}
    all_csv = []
    for label, path in FEATS.items():
        if not os.path.exists(path):
            continue
        rows = run(label, path)
        out[label] = rows
        for r in rows:
            all_csv.append({"encoder": label, **r})

    with open(os.path.join(OUT, "idep_finegrained.json"), "w") as f:
        json.dump(out, f, indent=2)

    if all_csv:
        with open(os.path.join(OUT, "idep_curve.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_csv[0].keys()))
            w.writeheader()
            w.writerows(all_csv)
    print(f"\nSaved: {os.path.join(OUT, 'idep_finegrained.json')}")
    print(f"Saved: {os.path.join(OUT, 'idep_curve.csv')}")


if __name__ == "__main__":
    main()
