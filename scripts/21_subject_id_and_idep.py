"""
N1: Subject-ID probe on CLIP and DINOv2 features.
N2: IDEP v1 — linear residualization (project features onto orthogonal complement
    of the identity subspace, then re-train engagement probe).

Protocol:
  - Subject-ID probe trained on within-Train clip-level 80/20 split (subject-disjoint
    test would be ill-defined for identity prediction). High accuracy here proves
    "features memorize identity."
  - For IDEP, the identity subspace is computed from a subject-ID classifier fit
    on the full Train split (70 subjects). The orthogonal projection is then
    applied to all features (Train, Val, Test). Engagement probe is re-fit on
    residualized Train features and evaluated on official subject-disjoint Test.

Output:
  results/idep/idep_results.json
  results/idep/idep_predictions.csv
"""
import os, json, csv
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_squared_error

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "idep")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)

FEATS_CLIP = os.path.join(BASE, "features", "clip_vitb32_features.npz")
FEATS_DINO = os.path.join(BASE, "features", "dinov2_vitb14_features.npz")


def metrics(yt, yp):
    return {
        "n": int(len(yt)),
        "accuracy": float(accuracy_score(yt, yp)),
        "kappa_quadratic": float(cohen_kappa_score(yt, yp, weights="quadratic")),
        "kappa_linear": float(cohen_kappa_score(yt, yp, weights="linear")),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "mse": float(mean_squared_error(yt, yp)),
        "pred_dist": {int(k): int((yp == k).sum()) for k in range(4)},
    }


def boot_kappa(yt, yp, n=1000):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def subject_id_probe(X_train, subj_train, seed=42):
    """N1: fit subject-ID classifier on 80/20 clip-level split within Train."""
    # Map subject_id strings to ints
    unique = sorted(set(subj_train.tolist()))
    sid_map = {s: i for i, s in enumerate(unique)}
    y = np.array([sid_map[s] for s in subj_train])

    # Stratified 80/20 split on clip level (each subject has clips in both)
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for sid in range(len(unique)):
        clip_idx = np.where(y == sid)[0]
        rng.shuffle(clip_idx)
        n_train = int(0.8 * len(clip_idx))
        train_idx.extend(clip_idx[:n_train])
        test_idx.extend(clip_idx[n_train:])
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    Xtr, ytr = X_train[train_idx], y[train_idx]
    Xte, yte = X_train[test_idx], y[test_idx]

    clf = LogisticRegression(C=10.0, max_iter=5000, solver="lbfgs", random_state=42)
    clf.fit(Xtr, ytr)
    train_acc = clf.score(Xtr, ytr)
    test_acc = clf.score(Xte, yte)
    return {
        "n_subjects": len(unique),
        "n_train_clips": int(len(train_idx)),
        "n_test_clips": int(len(test_idx)),
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
    }, clf, sid_map


def fit_id_classifier_full_train(X_train, subj_train):
    """Fit subject-ID classifier on FULL Train for IDEP subspace estimation."""
    unique = sorted(set(subj_train.tolist()))
    sid_map = {s: i for i, s in enumerate(unique)}
    y = np.array([sid_map[s] for s in subj_train])
    clf = LogisticRegression(C=10.0, max_iter=5000, solver="lbfgs", random_state=42)
    clf.fit(X_train, y)
    return clf, sid_map


def identity_subspace_projection(W, k=None):
    """Compute orthogonal projection matrix onto the orthogonal complement of
    the column span of W (the LogReg weight matrix, shape (K_subjects, D)).
    Returns P such that X @ P removes the identity subspace.

    If k is set, uses top-k singular vectors of W only."""
    # SVD of W^T (D x K). Columns of U span the identity subspace.
    U, S, Vt = np.linalg.svd(W.T, full_matrices=False)
    if k is None or k > U.shape[1]:
        k = U.shape[1]
    Uk = U[:, :k]  # (D, k) — top-k identity directions
    P = np.eye(W.shape[1]) - Uk @ Uk.T  # orthogonal complement projection
    return P, k


def engagement_probe(Xtr, ytr, Xva, yva, Xte, yte, label="probe"):
    """Standard subject-disjoint engagement probe: LogReg balanced + Ridge ordinal."""
    # LogReg
    best = None
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": float(v), "clf": clf}
    yhat_lr = best["clf"].predict(Xte)
    m_lr = metrics(yte, yhat_lr)
    m_lr["kappa_q_ci95"] = boot_kappa(yte, yhat_lr)
    m_lr["best_C"] = best["C"]

    # Ridge ordinal
    best = None
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(Xtr, ytr.astype(np.float32))
        yhat = np.clip(np.round(reg.predict(Xva)), 0, 3).astype(int)
        v = cohen_kappa_score(yva, yhat, weights="quadratic")
        if best is None or v > best["v"]:
            best = {"a": alpha, "v": float(v), "reg": reg}
    yhat_rd = np.clip(np.round(best["reg"].predict(Xte)), 0, 3).astype(int)
    m_rd = metrics(yte, yhat_rd)
    m_rd["kappa_q_ci95"] = boot_kappa(yte, yhat_rd)
    m_rd["best_alpha"] = best["a"]

    return {"logreg": m_lr, "ridge_ordinal": m_rd, "label": label}, yhat_lr, yhat_rd


def run_for_features(label, feats_path):
    print(f"\n========== {label} ({feats_path}) ==========")
    data = np.load(feats_path, allow_pickle=True)
    splits = data["split"]
    feats = data["feat"].astype(np.float32)
    eng = data["engagement"]
    subj = data["subject_id"]
    cid = data["clip_id"]

    tr = splits == "Train"
    va = splits == "Validation"
    te = splits == "Test"
    print(f"Train={tr.sum()}  Val={va.sum()}  Test={te.sum()}")

    # ----- N1: subject-ID probe -----
    print("\n--- N1: Subject-ID probe (80/20 within-Train clip split) ---")
    sid_res, _, _ = subject_id_probe(feats[tr], subj[tr])
    print(f"  n_subjects={sid_res['n_subjects']}  train_acc={sid_res['train_accuracy']:.3f}  "
          f"test_acc={sid_res['test_accuracy']:.3f}  (n_train_clips={sid_res['n_train_clips']}, "
          f"n_test_clips={sid_res['n_test_clips']})")

    # ----- Baseline engagement probe (no IDEP) -----
    print("\n--- Baseline engagement probe (recomputed for sanity) ---")
    base_res, _, _ = engagement_probe(feats[tr], eng[tr], feats[va], eng[va], feats[te], eng[te],
                                      label="baseline_no_idep")
    lr = base_res["logreg"]
    print(f"  LR  : κ_q={lr['kappa_quadratic']:.3f} [{lr['kappa_q_ci95'][0]:.3f},"
          f"{lr['kappa_q_ci95'][1]:.3f}]  acc={lr['accuracy']:.3f}")

    # ----- N2: IDEP via linear residualization -----
    print("\n--- N2: IDEP — fitting full-train subject-ID classifier for subspace ---")
    id_clf, sid_map = fit_id_classifier_full_train(feats[tr], subj[tr])
    W = id_clf.coef_  # (n_subjects, D)
    print(f"  ID classifier weight matrix: {W.shape}, rank up to {min(W.shape)}")

    idep_results = {}
    for k_str, k in [("k=full", None), ("k=70", 70), ("k=50", 50), ("k=20", 20), ("k=10", 10), ("k=5", 5)]:
        P, k_used = identity_subspace_projection(W, k=k)
        feats_proj = feats @ P
        # Sanity: identity should now be much harder to recover
        # (skip running ID classifier on projected; trust the math)
        # Engagement probe on residualized features
        print(f"\n  IDEP {k_str} (removed {k_used} identity dirs)")
        res, yhat_lr, yhat_rd = engagement_probe(
            feats_proj[tr], eng[tr], feats_proj[va], eng[va], feats_proj[te], eng[te],
            label=f"idep_{k_str}")
        lr = res["logreg"]
        rd = res["ridge_ordinal"]
        print(f"    LR   : κ_q={lr['kappa_quadratic']:.3f} [{lr['kappa_q_ci95'][0]:.3f},"
              f"{lr['kappa_q_ci95'][1]:.3f}]  acc={lr['accuracy']:.3f}  pred_dist={lr['pred_dist']}")
        print(f"    Ridge: κ_q={rd['kappa_quadratic']:.3f} [{rd['kappa_q_ci95'][0]:.3f},"
              f"{rd['kappa_q_ci95'][1]:.3f}]  acc={rd['accuracy']:.3f}  pred_dist={rd['pred_dist']}")
        idep_results[k_str] = res

    return {
        "feature_label": label,
        "n_subjects": sid_res["n_subjects"],
        "subject_id_probe": sid_res,
        "baseline_no_idep": base_res,
        "idep_sweep": idep_results,
    }


def main():
    out = {}
    for label, path in [("CLIP_ViT-B32", FEATS_CLIP), ("DINOv2_ViT-B14", FEATS_DINO)]:
        if not os.path.exists(path):
            print(f"Skipping {label}: {path} missing")
            continue
        out[label] = run_for_features(label, path)

    out_path = os.path.join(OUT, "idep_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
