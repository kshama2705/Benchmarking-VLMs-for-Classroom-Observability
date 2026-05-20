"""
Rigorous K-fold cross-validated late fusion.

Problem: with subject-disjoint Val/Test, val and test population statistics
differ enough that single-shot weight selection on val overfits. K-fold CV
on (Train+Val) gives a more honest weight estimate.

Procedure:
  1. Pool Train and Val clips. Keep Test held out.
  2. Run K-fold CV on the pool:
     - For each fold, train base probes on (K-1)/K and predict on fold-out.
     - Collect out-of-fold predictions for the entire pool.
  3. On the OOF predictions, sweep fusion weight to maximize κ.
  4. Train final base probes on ALL (Train+Val) data.
  5. Apply learned weight on test.
  6. Bootstrap test CIs.

This protocol prevents val-overfitting because the weight is selected on
held-out samples within each fold, then averaged.

Output:
  results/sota/kfold_late_fusion.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPORAL = os.path.join(BASE, "features", "daisee_face_signals_temporal.npz")
SINGLE = os.path.join(BASE, "features", "daisee_face_signals.npz")
POSE = os.path.join(BASE, "features", "daisee_pose_signals.npz")
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CLIPL = os.path.join(BASE, "features", "daisee_clip_l_14_features.npz")
DINO = os.path.join(BASE, "features", "dinov2_vitb14_features.npz")
OUT = os.path.join(BASE, "results", "sota", "kfold_late_fusion.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)


def boot_kq(yt, yp, n=1000):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def fit_lr_pick_C(Xtr, ytr, n_folds=3):
    """Pick C via K-fold CV on the given training data."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    best_C = None; best_avg = -1e9
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        vs = []
        for tr_idx, va_idx in kf.split(Xtr):
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                     solver="lbfgs", random_state=42)
            clf.fit(Xtr[tr_idx], ytr[tr_idx])
            v = cohen_kappa_score(ytr[va_idx], clf.predict(Xtr[va_idx]), weights="quadratic")
            vs.append(v)
        if np.mean(vs) > best_avg:
            best_avg = np.mean(vs); best_C = C
    clf = LogisticRegression(C=best_C, class_weight="balanced", max_iter=10000,
                             solver="lbfgs", random_state=42)
    clf.fit(Xtr, ytr)
    return clf, float(best_avg), best_C


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


def kfold_oof_probs(X, y, n_folds=5):
    """OOF predictions on X, y via K-fold CV."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    out = np.zeros((len(y), 4), dtype=np.float32)
    for tr_idx, va_idx in kf.split(X):
        clf, _, _ = fit_lr_pick_C(X[tr_idx], y[tr_idx])
        out[va_idx] = clf.predict_proba(X[va_idx])
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

    sf = np.load(SINGLE, allow_pickle=True)
    single_explicit = align(clip_ids, sf["clip_id"], np.nan_to_num(
        np.concatenate([sf["blendshapes"], sf["head_pose"], sf["eye_gaze"], sf["landmark_summary"]], axis=1)))

    pose_d = np.load(POSE, allow_pickle=True)
    pose_raw = align(clip_ids, pose_d["clip_id"], np.nan_to_num(pose_d["pose_features"]))

    # Combine train+val into one "pool" for K-fold selection
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    pool = tr | va
    print(f"Pool (Train+Val) = {pool.sum()}; Test = {te.sum()}")

    # Z-score features using pool stats (train+val)
    sc_t = StandardScaler().fit(temporal_explicit[pool])
    temporal_z = sc_t.transform(temporal_explicit).astype(np.float32)
    sc_s = StandardScaler().fit(single_explicit[pool])
    single_z = sc_s.transform(single_explicit).astype(np.float32)
    sc_p = StandardScaler().fit(pose_raw[pool])
    pose_z = sc_p.transform(pose_raw).astype(np.float32)

    # Frozen encoder features
    feats = {"face_temporal": temporal_z, "face_single": single_z, "body_pose": pose_z}
    for label, path in [("SigLIP_L", SIGLIP), ("CLIP_L", CLIPL), ("DINOv2", DINO)]:
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            key = "feat" if "feat" in d.files else "cls_feat"
            feats[label] = align(clip_ids, d["clip_id"], d[key].astype(np.float32))

    # Step 1: Get OOF probabilities for each probe on Train+Val pool
    print("\nComputing OOF probabilities on Train+Val pool (5-fold)...")
    pool_probs = {}  # OOF predictions, aligned to pool order
    test_probs = {}  # test predictions from full-pool-trained probes
    for name, X in feats.items():
        print(f"  {name} ...", flush=True)
        # OOF on pool
        pool_oof = kfold_oof_probs(X[pool], eng[pool], n_folds=5)
        pool_probs[name] = pool_oof
        # Final probe trained on full pool, predict on test
        clf, val_v, C = fit_lr_pick_C(X[pool], eng[pool])
        test_probs[name] = clf.predict_proba(X[te])
        print(f"    OOF mean κ_q (across 5 folds): tracking only headline. C={C}")

    # Step 2: Sweep fusion weight on OOF predictions (pool); apply to test
    y_pool = eng[pool]
    y_te = eng[te]

    out = {}

    # Solo metrics first (test)
    print("\n=== Solo (full-pool-trained → test) ===")
    for name in feats:
        yhat = test_probs[name].argmax(axis=1)
        m = metrics(y_te, yhat)
        out[f"solo_{name}"] = m
        print(f"  solo_{name:18} κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}")

    # Pairwise scalar fusion using OOF for weight selection
    print("\n=== K-fold-CV-selected scalar fusion (SigLIP_L + each) ===")
    if "SigLIP_L" in pool_probs:
        base_pool = pool_probs["SigLIP_L"]
        base_te = test_probs["SigLIP_L"]
        for name in pool_probs:
            if name == "SigLIP_L":
                continue
            other_pool = pool_probs[name]
            other_te = test_probs[name]
            # Sweep w using OOF on pool
            best = None
            for w in np.linspace(0.0, 1.0, 41):
                p = w * other_pool + (1 - w) * base_pool
                v = cohen_kappa_score(y_pool, p.argmax(axis=1), weights="quadratic")
                if best is None or v > best["v"]:
                    best = {"w": float(w), "v": float(v)}
            wb = best["w"]
            yhat = (wb * other_te + (1 - wb) * base_te).argmax(axis=1)
            m = metrics(y_te, yhat)
            tag = f"SigLIP_L+{name}_kfoldscalar"
            out[tag] = {**m, "w": wb, "pool_kq": best["v"]}
            print(f"  {tag:45} w={wb:.2f}  κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  pool={best['v']:.3f}")

    # Multi-probe scalar fusion (random dirichlet search on pool)
    print("\n=== K-fold-CV-selected multi-probe fusion (4-5 probes) ===")
    multi_names = [n for n in ["SigLIP_L", "DINOv2", "face_temporal", "face_single", "body_pose"]
                   if n in pool_probs]
    print(f"  using probes: {multi_names}")
    probs_pool_list = [pool_probs[n] for n in multi_names]
    probs_te_list = [test_probs[n] for n in multi_names]
    rng = np.random.default_rng(42)
    best = None
    for _ in range(10000):
        ws = rng.dirichlet(np.ones(len(multi_names)) * 1.0)
        p = sum(probs_pool_list[j] * ws[j] for j in range(len(multi_names)))
        v = cohen_kappa_score(y_pool, p.argmax(axis=1), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"ws": ws.tolist(), "v": float(v)}
    p_te = sum(probs_te_list[j] * best["ws"][j] for j in range(len(multi_names)))
    yhat = p_te.argmax(axis=1)
    m = metrics(y_te, yhat)
    out["multi_kfoldscalar"] = {**m, "weights": best["ws"], "probes": multi_names, "pool_kq": best["v"]}
    print(f"  multi_kfoldscalar  κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  pool={best['v']:.3f}")
    print(f"    weights: {dict(zip(multi_names, [f'{w:.3f}' for w in best['ws']]))}")

    # Stacking with Ridge meta (use OOF train + val together)
    print("\n=== Stacking (Ridge meta on K-fold OOF probs) ===")
    Z_pool = np.concatenate([pool_probs[n] for n in multi_names], axis=1)
    Z_te = np.concatenate([test_probs[n] for n in multi_names], axis=1)
    # K-fold CV to pick alpha
    from sklearn.linear_model import Ridge
    best_rd = None
    for alpha in [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        vs = []
        for tr_idx, va_idx in kf.split(Z_pool):
            reg = Ridge(alpha=alpha, random_state=42)
            reg.fit(Z_pool[tr_idx], y_pool[tr_idx].astype(np.float32))
            yhat_va = np.clip(np.round(reg.predict(Z_pool[va_idx])), 0, 3).astype(int)
            v = cohen_kappa_score(y_pool[va_idx], yhat_va, weights="quadratic")
            vs.append(v)
        avg = np.mean(vs)
        if best_rd is None or avg > best_rd["avg"]:
            best_rd = {"alpha": alpha, "avg": float(avg), "fold_vs": [float(v) for v in vs]}
    # Fit final on full pool
    reg = Ridge(alpha=best_rd["alpha"], random_state=42)
    reg.fit(Z_pool, y_pool.astype(np.float32))
    yhat = np.clip(np.round(reg.predict(Z_te)), 0, 3).astype(int)
    m = metrics(y_te, yhat)
    out["stacking_ridge_kfold"] = {**m, "alpha": best_rd["alpha"],
                                    "fold_kq_mean": best_rd["avg"],
                                    "fold_vs": best_rd["fold_vs"]}
    print(f"  stacking_ridge_kfold α={best_rd['alpha']} κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  "
          f"fold_kq={best_rd['avg']:.3f}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
