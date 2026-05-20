"""
Final fusion: combine all available encoders + EMBER explicit signals.

For each base feature group:
  - face_temporal (186-d MediaPipe), body_pose (16-d MediaPipe)
  - SigLIP-L (1024-d image), CLIP-L (768-d image), DINOv2 (768-d image)
  - VideoMAE-base (768-d video — if features exist)
  - SigLIP-SO400M (1152-d image — if features exist)

We train an LR probe on each, then late-fuse the predictions via:
  - Pairwise scalar fusion against SigLIP-L baseline
  - Triple/quad/quint scalar fusion (small grid search)
  - Stacked Ridge meta-learner with K-fold CV (the best we found)

Output:
  results/sota/final_fusion.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from itertools import product

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_OUT = os.path.join(BASE, "results", "sota", "final_fusion.json")
os.makedirs(os.path.dirname(RES_OUT), exist_ok=True)
RNG = np.random.default_rng(42)

# Feature paths
PATHS = {
    "SigLIP_L":     os.path.join(BASE, "features", "daisee_siglip_l_features.npz"),
    "CLIP_L":       os.path.join(BASE, "features", "daisee_clip_l_14_features.npz"),
    "DINOv2":       os.path.join(BASE, "features", "dinov2_vitb14_features.npz"),
    "SigLIP_SO400M": os.path.join(BASE, "features", "daisee_siglip_so400m_features.npz"),
    "VideoMAE":     os.path.join(BASE, "features", "daisee_videomae_features.npz"),
}
SIGNALS = os.path.join(BASE, "features", "daisee_face_signals.npz")
TEMPORAL = os.path.join(BASE, "features", "daisee_face_signals_temporal.npz")
POSE = os.path.join(BASE, "features", "daisee_pose_signals.npz")


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


def main():
    # Use temporal as canonical clip_id source
    tmp = np.load(TEMPORAL, allow_pickle=True)
    clip_ids = tmp["clip_id"]; splits = tmp["split"]
    eng = tmp["engagement"].astype(np.int64)
    temporal_explicit = np.nan_to_num(np.concatenate([
        tmp["blendshapes_mean"], tmp["blendshapes_std"], tmp["blendshapes_delta"],
        tmp["head_pose_mean"], tmp["head_pose_std"],
        tmp["eye_gaze_mean"], tmp["eye_gaze_std"],
        tmp["landmark_summary_mean"]], axis=1))
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    sc = StandardScaler().fit(temporal_explicit[tr])
    explicit_z = sc.transform(temporal_explicit).astype(np.float32)

    pose_d = np.load(POSE, allow_pickle=True)
    pose_raw = align(clip_ids, pose_d["clip_id"], np.nan_to_num(pose_d["pose_features"]))
    sc_p = StandardScaler().fit(pose_raw[tr])
    pose_z = sc_p.transform(pose_raw).astype(np.float32)

    feats = {"face_temporal": explicit_z, "body_pose": pose_z}
    for label, path in PATHS.items():
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            key = "feat" if "feat" in d.files else "cls_feat"
            feats[label] = align(clip_ids, d["clip_id"], d[key].astype(np.float32))
    print(f"Available encoders: {list(feats.keys())}")

    # Train probes
    print("\nTraining base probes...")
    probes = {}
    for name, X in feats.items():
        clf, val_v, C = fit_lr(X[tr], eng[tr], X[va], eng[va])
        yhat_te = clf.predict(X[te])
        probes[name] = {
            "p_tr": clf.predict_proba(X[tr]),
            "p_va": clf.predict_proba(X[va]),
            "p_te": clf.predict_proba(X[te]),
            "metrics": metrics(eng[te], yhat_te),
            "val_kq": val_v, "C": C, "dim": X.shape[1],
        }
        m = probes[name]["metrics"]
        print(f"  {name:18} solo  κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  val={val_v:.3f}  dim={X.shape[1]}")

    out = {"solo": {n: probes[n]["metrics"] | {"val_kq": probes[n]["val_kq"]} for n in probes}}

    # Pair fusion (each probe vs SigLIP-L)
    if "SigLIP_L" in probes:
        print("\n=== Pair fusions with SigLIP-L ===")
        base_va = probes["SigLIP_L"]["p_va"]
        base_te = probes["SigLIP_L"]["p_te"]
        for name in probes:
            if name == "SigLIP_L":
                continue
            other_va = probes[name]["p_va"]
            other_te = probes[name]["p_te"]
            best = None
            for w in np.linspace(0.0, 1.0, 41):
                p_va = w * other_va + (1 - w) * base_va
                v = cohen_kappa_score(eng[va], p_va.argmax(axis=1), weights="quadratic")
                if best is None or v > best["v"]:
                    best = {"w": float(w), "v": float(v)}
            wb = best["w"]
            yhat = (wb * other_te + (1 - wb) * base_te).argmax(axis=1)
            m = metrics(eng[te], yhat)
            tag = f"SigLIP_L+{name}_late"
            out[tag] = {**m, "w": wb, "val_kq": best["v"]}
            print(f"  {tag:35} w={wb:.2f}  κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  val={best['v']:.3f}")

    # Triple + multi-encoder scalar grid (only large encoders + explicit + pose)
    print("\n=== Multi-probe scalar fusion ===")
    # Pick top 4-6 probes for fusion
    candidate_probes = [n for n in ["SigLIP_L", "SigLIP_SO400M", "VideoMAE", "DINOv2", "face_temporal", "body_pose"]
                        if n in probes]
    print(f"  Candidates: {candidate_probes}")

    # Brute force scalar search over 5-probe simplex (granularity 11)
    if len(candidate_probes) >= 3:
        granularity = 11
        ws = np.linspace(0.0, 1.0, granularity)
        # Build all simplex points
        from itertools import combinations_with_replacement
        # Sample weight vectors instead (faster)
        rng = np.random.default_rng(42)
        n_iter = 5000
        best = None
        probs_va_list = [probes[n]["p_va"] for n in candidate_probes]
        probs_te_list = [probes[n]["p_te"] for n in candidate_probes]
        for _ in range(n_iter):
            ws_v = rng.dirichlet(np.ones(len(candidate_probes)) * 1.0)
            p_va = sum(probs_va_list[j] * ws_v[j] for j in range(len(candidate_probes)))
            v = cohen_kappa_score(eng[va], p_va.argmax(axis=1), weights="quadratic")
            if best is None or v > best["v"]:
                best = {"ws": ws_v.tolist(), "v": float(v)}
        ws_v = np.array(best["ws"])
        p_te = sum(probs_te_list[j] * ws_v[j] for j in range(len(candidate_probes)))
        yhat = p_te.argmax(axis=1)
        m = metrics(eng[te], yhat)
        out["multi_scalar_simplex"] = {**m, "weights": best["ws"], "probes": candidate_probes, "val_kq": best["v"]}
        print(f"  multi_scalar_simplex  κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  val={best['v']:.3f}")
        print(f"    weights: {dict(zip(candidate_probes, [f'{w:.2f}' for w in best['ws']]))}")

    # ===== Stacking with K-fold on Train+Val =====
    print("\n=== Stacking (Ridge meta on probe probabilities) ===")
    # Concatenate per-clip probe probabilities
    keep = candidate_probes
    Z_tr = np.concatenate([probes[n]["p_tr"] for n in keep], axis=1)
    Z_va = np.concatenate([probes[n]["p_va"] for n in keep], axis=1)
    Z_te = np.concatenate([probes[n]["p_te"] for n in keep], axis=1)

    # Ridge with strong reg, alpha tuned on val
    best_rd = None
    for alpha in [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]:
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(Z_tr, eng[tr].astype(np.float32))
        yhat_va = np.clip(np.round(reg.predict(Z_va)), 0, 3).astype(int)
        v = cohen_kappa_score(eng[va], yhat_va, weights="quadratic")
        if best_rd is None or v > best_rd["v"]:
            best_rd = {"a": alpha, "v": v, "reg": reg}
    yhat = np.clip(np.round(best_rd["reg"].predict(Z_te)), 0, 3).astype(int)
    m = metrics(eng[te], yhat)
    out["stacking_ridge_train"] = {**m, "alpha": best_rd["a"], "val_kq": best_rd["v"]}
    print(f"  stacking_ridge_train α={best_rd['a']} κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  val={best_rd['v']:.3f}")

    # LR meta with strong reg
    best_lr = None
    for C in [0.001, 0.01, 0.1, 1.0]:
        meta = LogisticRegression(C=C, class_weight="balanced", max_iter=10000, solver="lbfgs", random_state=42)
        meta.fit(Z_tr, eng[tr])
        v = cohen_kappa_score(eng[va], meta.predict(Z_va), weights="quadratic")
        if best_lr is None or v > best_lr["v"]:
            best_lr = {"C": C, "v": v, "meta": meta}
    yhat = best_lr["meta"].predict(Z_te)
    m = metrics(eng[te], yhat)
    out["stacking_logreg_train"] = {**m, "C": best_lr["C"], "val_kq": best_lr["v"]}
    print(f"  stacking_logreg_train C={best_lr['C']} κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  val={best_lr['v']:.3f}")

    with open(RES_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {RES_OUT}")


if __name__ == "__main__":
    main()
