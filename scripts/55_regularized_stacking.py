"""
Regularized stacking ensemble — use Ridge regression with strong L2 on the
concatenated probe outputs to learn fusion weights that don't overfit val.

Setup:
  - For each base feature group (face_temporal, body_pose, SigLIP_L, CLIP_L, DINOv2)
    train LR on Train, predict probabilities on Train+Val+Test (with cross-validation
    on Train to get unbiased Train predictions).
  - Concatenate all probe probabilities → (N, 4 × n_probes) features.
  - Train Ridge regression (on these probe outputs) to predict engagement on Train.
  - Sweep alpha on Val.
  - Evaluate on Test.

This is a principled K-fold stacking approach that avoids the val-overfit problem
the random class-aware search hit.

Output:
  results/sota/regularized_stacking.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPORAL = os.path.join(BASE, "features", "daisee_face_signals_temporal.npz")
POSE = os.path.join(BASE, "features", "daisee_pose_signals.npz")
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CLIPL = os.path.join(BASE, "features", "daisee_clip_l_14_features.npz")
DINO = os.path.join(BASE, "features", "dinov2_vitb14_features.npz")
OUT = os.path.join(BASE, "results", "sota", "regularized_stacking.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)


def boot_kq(yt, yp, n=1000):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def fit_lr(Xtr, ytr, Xva=None, yva=None):
    if Xva is None:
        # Use 5-fold CV on train to pick C
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        best_C = None
        best_avg = -1e9
        for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
            vs = []
            for tr_idx, va_idx in kf.split(Xtr):
                clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                         solver="lbfgs", random_state=42)
                clf.fit(Xtr[tr_idx], ytr[tr_idx])
                v = cohen_kappa_score(ytr[va_idx], clf.predict(Xtr[va_idx]), weights="quadratic")
                vs.append(v)
            if np.mean(vs) > best_avg:
                best_avg = np.mean(vs)
                best_C = C
        clf = LogisticRegression(C=best_C, class_weight="balanced", max_iter=10000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        return clf, float(best_avg), best_C
    else:
        best = None
        for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                     solver="lbfgs", random_state=42)
            clf.fit(Xtr, ytr)
            v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
            if best is None or v > best["v"]:
                best = {"C": C, "v": float(v), "clf": clf}
        return best["clf"], best["v"], best["C"]


def get_train_oof_probs(Xtr, ytr, n_folds=5):
    """Out-of-fold predictions on train using K-fold CV (for stacking).
    Returns (N_train, 4) probability matrix."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    out = np.zeros((len(ytr), 4), dtype=np.float32)
    for tr_idx, va_idx in kf.split(Xtr):
        clf, _, _ = fit_lr(Xtr[tr_idx], ytr[tr_idx], Xtr[va_idx], ytr[va_idx])
        out[va_idx] = clf.predict_proba(Xtr[va_idx])
    return out


def metrics(yt, yp):
    return {
        "kappa_q": float(cohen_kappa_score(yt, yp, weights="quadratic")),
        "kappa_q_ci": boot_kq(yt, yp),
        "accuracy": float(accuracy_score(yt, yp)),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "pred_dist": {int(k): int((yp == k).sum()) for k in range(4)},
    }


def align(target_ids, source_ids, source_feats):
    s_idx = {c: i for i, c in enumerate(source_ids)}
    out = np.zeros((len(target_ids), source_feats.shape[1]), dtype=np.float32)
    for j, c in enumerate(target_ids):
        if c in s_idx:
            out[j] = source_feats[s_idx[c]]
    return out


def main():
    print("Loading...")
    tmp = np.load(TEMPORAL, allow_pickle=True)
    clip_ids = tmp["clip_id"]; splits = tmp["split"]
    eng = tmp["engagement"].astype(np.int64)
    temporal_explicit = np.nan_to_num(np.concatenate([
        tmp["blendshapes_mean"], tmp["blendshapes_std"], tmp["blendshapes_delta"],
        tmp["head_pose_mean"], tmp["head_pose_std"],
        tmp["eye_gaze_mean"], tmp["eye_gaze_std"],
        tmp["landmark_summary_mean"]], axis=1))

    pose_d = np.load(POSE, allow_pickle=True)
    pose_feats = align(clip_ids, pose_d["clip_id"], np.nan_to_num(pose_d["pose_features"]))

    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"

    sc_t = StandardScaler().fit(temporal_explicit[tr])
    temporal_explicit_z = sc_t.transform(temporal_explicit).astype(np.float32)
    sc_p = StandardScaler().fit(pose_feats[tr])
    pose_z = sc_p.transform(pose_feats).astype(np.float32)

    feats = {"face_temporal": temporal_explicit_z, "body_pose": pose_z}
    for label, path in [("SigLIP_L", SIGLIP), ("CLIP_L", CLIPL), ("DINOv2", DINO)]:
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            key = "feat" if "feat" in d.files else "cls_feat"
            feats[label] = align(clip_ids, d["clip_id"], d[key].astype(np.float32))

    print(f"Train={tr.sum()} Val={va.sum()} Test={te.sum()}")

    # Get OOF train probs for each probe (for stacking)
    print("\nComputing out-of-fold train probabilities for each probe...")
    train_probs = {}
    val_probs = {}
    test_probs = {}
    for name, X in feats.items():
        print(f"  {name} ...")
        # OOF probs on train
        train_probs[name] = get_train_oof_probs(X[tr], eng[tr])
        # Probe trained on full train; eval on val and test
        clf, val_v, C = fit_lr(X[tr], eng[tr], X[va], eng[va])
        val_probs[name] = clf.predict_proba(X[va])
        test_probs[name] = clf.predict_proba(X[te])

    # Concatenate probe probabilities
    print("\nStacking ensemble (Ridge meta-learner)...")
    Z_tr = np.concatenate([train_probs[n] for n in train_probs], axis=1)
    Z_va = np.concatenate([val_probs[n] for n in val_probs], axis=1)
    Z_te = np.concatenate([test_probs[n] for n in test_probs], axis=1)
    print(f"  meta-feature dim: {Z_tr.shape[1]}")

    out = {}

    # Try Ridge meta-learner (uses real-valued labels)
    best_rd = None
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(Z_tr, eng[tr].astype(np.float32))
        v = cohen_kappa_score(eng[va], np.clip(np.round(reg.predict(Z_va)), 0, 3).astype(int), weights="quadratic")
        if best_rd is None or v > best_rd["v"]:
            best_rd = {"a": alpha, "v": v, "reg": reg}
    yhat = np.clip(np.round(best_rd["reg"].predict(Z_te)), 0, 3).astype(int)
    out["stacking_ridge"] = {**metrics(eng[te], yhat), "alpha": best_rd["a"], "val_kq": best_rd["v"]}
    print(f"  stacking_ridge α={best_rd['a']} κ_q={out['stacking_ridge']['kappa_q']:.3f} {out['stacking_ridge']['kappa_q_ci']}  val={best_rd['v']:.3f}")

    # Try L2-regularized LR meta-learner
    best_lr = None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        meta = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                  solver="lbfgs", random_state=42)
        meta.fit(Z_tr, eng[tr])
        v = cohen_kappa_score(eng[va], meta.predict(Z_va), weights="quadratic")
        if best_lr is None or v > best_lr["v"]:
            best_lr = {"C": C, "v": v, "meta": meta}
    yhat = best_lr["meta"].predict(Z_te)
    out["stacking_logreg"] = {**metrics(eng[te], yhat), "C": best_lr["C"], "val_kq": best_lr["v"]}
    print(f"  stacking_logreg C={best_lr['C']} κ_q={out['stacking_logreg']['kappa_q']:.3f} {out['stacking_logreg']['kappa_q_ci']}  val={best_lr['v']:.3f}")

    # Subsets — drop each probe
    print("\nLeave-one-out ablation (drop one probe at a time, Ridge stacking):")
    probe_names = list(feats.keys())
    for drop_name in probe_names:
        keep_names = [n for n in probe_names if n != drop_name]
        Z_tr_d = np.concatenate([train_probs[n] for n in keep_names], axis=1)
        Z_va_d = np.concatenate([val_probs[n] for n in keep_names], axis=1)
        Z_te_d = np.concatenate([test_probs[n] for n in keep_names], axis=1)
        best = None
        for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
            reg = Ridge(alpha=alpha, random_state=42)
            reg.fit(Z_tr_d, eng[tr].astype(np.float32))
            v = cohen_kappa_score(eng[va], np.clip(np.round(reg.predict(Z_va_d)), 0, 3).astype(int), weights="quadratic")
            if best is None or v > best["v"]:
                best = {"a": alpha, "v": v, "reg": reg}
        yhat = np.clip(np.round(best["reg"].predict(Z_te_d)), 0, 3).astype(int)
        m = metrics(eng[te], yhat)
        out[f"drop_{drop_name}"] = {**m, "alpha": best["a"], "val_kq": best["v"]}
        print(f"  drop {drop_name:15} κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  val={best['v']:.3f}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
