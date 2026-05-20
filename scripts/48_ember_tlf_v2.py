"""
EMBER-TLF v2 — more thorough class-aware fusion + cross-validated weight selection.

Improvements:
  1. Bigger random search for class-aware weights (5000 samples)
  2. K-fold cross-validation on the meta-fit: split val into K folds, choose
     weights that perform best on average across folds (reduces val-overfit)
  3. Encoder ensemble: combine SigLIP-L + CLIP-L + DINOv2 + explicit predictions
  4. Final test bootstrap CI

Output:
  results/ember/ember_tlf_v2_results.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPORAL = os.path.join(BASE, "features", "daisee_face_signals_temporal.npz")
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CLIPL = os.path.join(BASE, "features", "daisee_clip_l_14_features.npz")
DINO = os.path.join(BASE, "features", "dinov2_vitb14_features.npz")
OUT = os.path.join(BASE, "results", "ember", "ember_tlf_v2_results.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)


def boot_kq(yt, yp, n=1000):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


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


def class_aware_search(probs_list, y_va, n_iter=5000, seed=42):
    """probs_list: list of (N, 4) probability arrays from different probes.
    Search per-class blending weights. We treat the first probs as 'base' and
    sweep per-class blend weights for each additional probe."""
    rng = np.random.default_rng(seed)
    n_probes = len(probs_list)
    # Per-class weights for each non-base probe: (n_probes-1, 4)
    best = None
    for _ in range(n_iter):
        # Sample n_probes weights per class via Dirichlet
        ws = rng.dirichlet(np.ones(n_probes) * 2.0, size=4)  # (4, n_probes)
        # ws[c, j] is the weight on probe j for class c
        # Compute fused probs as sum_j ws[c, j] * probs_j[:, c]
        p_fused = np.zeros_like(probs_list[0])
        for j, p in enumerate(probs_list):
            p_fused += p * ws[None, :, j]  # broadcast: ws[:, j] is per-class
        v = cohen_kappa_score(y_va, p_fused.argmax(axis=1), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"ws": ws.tolist(), "v": float(v)}
    return best


def kfold_class_aware(probs_va_list, y_va, n_folds=5, n_iter=2000, seed=42):
    """K-fold CV on val: choose weights that perform best across folds.
    Returns the final weights (averaged across folds) and per-fold scores."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_weights = []
    fold_vs = []
    for fold_i, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(y_va)))):
        # Use train_idx (subset of val) to fit weights, val_idx for sanity
        probs_sub = [p[train_idx] for p in probs_va_list]
        y_sub = y_va[train_idx]
        best = class_aware_search(probs_sub, y_sub, n_iter=n_iter, seed=seed + fold_i)
        fold_weights.append(np.array(best["ws"]))
        fold_vs.append(best["v"])
    # Average weights across folds
    avg_ws = np.mean(np.stack(fold_weights), axis=0)
    return avg_ws, fold_vs, fold_weights


def fuse(probs_list, ws_per_class):
    """ws_per_class: (4, n_probes). Returns fused (N, 4)."""
    out = np.zeros_like(probs_list[0])
    for j, p in enumerate(probs_list):
        out += p * ws_per_class[None, :, j]
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
    print(f"  temporal_explicit dim: {temporal_explicit.shape[1]}")

    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    sc = StandardScaler().fit(temporal_explicit[tr])
    explicit = sc.transform(temporal_explicit).astype(np.float32)

    feats = {}
    for label, path in [("SigLIP_L", SIGLIP), ("CLIP_L", CLIPL), ("DINOv2", DINO)]:
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            key = "feat" if "feat" in d.files else "cls_feat"
            feats[label] = align(clip_ids, d["clip_id"], d[key].astype(np.float32))

    # Fit base probes
    print("\nFitting base probes...")
    probes = {}
    print("  explicit ...")
    clf_e, _, _ = fit_lr(explicit[tr], eng[tr], explicit[va], eng[va])
    probes["explicit"] = {"va": clf_e.predict_proba(explicit[va]),
                          "te": clf_e.predict_proba(explicit[te])}
    for label, X in feats.items():
        print(f"  {label} ...")
        clf, _, _ = fit_lr(X[tr], eng[tr], X[va], eng[va])
        probes[label] = {"va": clf.predict_proba(X[va]),
                         "te": clf.predict_proba(X[te])}

    out = {}

    # ==== Pair fusions (SigLIP_L + explicit, etc.) ====
    print("\n=== Pair fusions (CLS + explicit) — 5000-iter class-aware ===")
    for enc_label in ["SigLIP_L", "CLIP_L", "DINOv2"]:
        if enc_label not in probes:
            continue
        probs_va_list = [probes[enc_label]["va"], probes["explicit"]["va"]]
        probs_te_list = [probes[enc_label]["te"], probes["explicit"]["te"]]

        # 5000-iter random search on val
        best = class_aware_search(probs_va_list, eng[va], n_iter=5000)
        ws = np.array(best["ws"])
        p_te = fuse(probs_te_list, ws)
        yhat = p_te.argmax(axis=1)
        tag = f"{enc_label}+explicit_classaware_5k"
        out[tag] = {**metrics(eng[te], yhat),
                    "weights": ws.tolist(), "val_kq": best["v"]}
        print(f"  {tag:40} κ_q={out[tag]['kappa_q']:.3f} {out[tag]['kappa_q_ci']}  val={best['v']:.3f}")

        # 5-fold CV class-aware
        avg_ws, fold_vs, fold_weights = kfold_class_aware(
            probs_va_list, eng[va], n_folds=5, n_iter=2000)
        p_te = fuse(probs_te_list, avg_ws)
        yhat = p_te.argmax(axis=1)
        tag = f"{enc_label}+explicit_kfold_classaware"
        out[tag] = {**metrics(eng[te], yhat),
                    "fold_kqs": fold_vs, "avg_weights": avg_ws.tolist(),
                    "fold_kq_mean": float(np.mean(fold_vs)),
                    "fold_kq_std": float(np.std(fold_vs))}
        print(f"  {tag:40} κ_q={out[tag]['kappa_q']:.3f} {out[tag]['kappa_q_ci']}  "
              f"fold_kq={np.mean(fold_vs):.3f}±{np.std(fold_vs):.3f}")

    # ==== Triple ensemble: SigLIP_L + CLIP_L + DINOv2 + explicit ====
    print("\n=== Triple ensemble: SigLIP_L + CLIP_L + DINOv2 + explicit ===")
    probs_va_list = [probes[k]["va"] for k in ["SigLIP_L", "CLIP_L", "DINOv2", "explicit"] if k in probes]
    probs_te_list = [probes[k]["te"] for k in ["SigLIP_L", "CLIP_L", "DINOv2", "explicit"] if k in probes]
    if len(probs_va_list) == 4:
        best = class_aware_search(probs_va_list, eng[va], n_iter=10000)
        ws = np.array(best["ws"])
        p_te = fuse(probs_te_list, ws)
        yhat = p_te.argmax(axis=1)
        tag = "ensemble_all4_classaware_10k"
        out[tag] = {**metrics(eng[te], yhat), "weights": ws.tolist(), "val_kq": best["v"]}
        print(f"  {tag:40} κ_q={out[tag]['kappa_q']:.3f} {out[tag]['kappa_q_ci']}  val={best['v']:.3f}")

        avg_ws, fold_vs, _ = kfold_class_aware(probs_va_list, eng[va], n_folds=5, n_iter=2000)
        p_te = fuse(probs_te_list, avg_ws)
        yhat = p_te.argmax(axis=1)
        tag = "ensemble_all4_kfold"
        out[tag] = {**metrics(eng[te], yhat),
                    "avg_weights": avg_ws.tolist(),
                    "fold_kq_mean": float(np.mean(fold_vs)),
                    "fold_kq_std": float(np.std(fold_vs))}
        print(f"  {tag:40} κ_q={out[tag]['kappa_q']:.3f} {out[tag]['kappa_q_ci']}  "
              f"fold_kq={np.mean(fold_vs):.3f}±{np.std(fold_vs):.3f}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
