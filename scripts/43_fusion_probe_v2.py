"""
EMBER v2 fusion probe — fixes the StandardScaler-on-CLS issue.

Frozen VLM CLS features are already L2-normalized (open_clip / DINOv2 convention).
StandardScaler breaks them. We:
  - Z-score-normalize ONLY the explicit signals (AUs, pose, gaze, landmark summary)
  - Leave the frozen CLS features as-is
  - Concatenate

Output:
  results/ember/fusion_results_v2.json
"""
import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGNALS = os.path.join(BASE, "features", "daisee_face_signals.npz")
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CLIPL = os.path.join(BASE, "features", "daisee_clip_l_14_features.npz")
DINO = os.path.join(BASE, "features", "dinov2_vitb14_features.npz")
OUT_DIR = os.path.join(BASE, "results", "ember")
RES = os.path.join(OUT_DIR, "fusion_results_v2.json")
os.makedirs(OUT_DIR, exist_ok=True)
RNG = np.random.default_rng(42)


def boot_kq(yt, yp, n=1000):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def probe(Xtr, ytr, Xva, yva, Xte, yte, label):
    """No internal scaling — caller has already prepared features."""
    best = None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": v, "clf": clf}
    yhat_lr = best["clf"].predict(Xte)
    m_lr = {
        "kappa_q": float(cohen_kappa_score(yte, yhat_lr, weights="quadratic")),
        "kappa_q_ci": boot_kq(yte, yhat_lr),
        "accuracy": float(accuracy_score(yte, yhat_lr)),
        "f1_macro": float(f1_score(yte, yhat_lr, average="macro", zero_division=0)),
        "pred_dist": {int(k): int((yhat_lr == k).sum()) for k in range(4)},
        "best_C": best["C"],
    }
    best = None
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(Xtr, ytr.astype(np.float32))
        v = cohen_kappa_score(yva, np.clip(np.round(reg.predict(Xva)), 0, 3).astype(int), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"a": alpha, "v": v, "reg": reg}
    yhat_rd = np.clip(np.round(best["reg"].predict(Xte)), 0, 3).astype(int)
    m_rd = {
        "kappa_q": float(cohen_kappa_score(yte, yhat_rd, weights="quadratic")),
        "kappa_q_ci": boot_kq(yte, yhat_rd),
        "accuracy": float(accuracy_score(yte, yhat_rd)),
        "f1_macro": float(f1_score(yte, yhat_rd, average="macro", zero_division=0)),
        "pred_dist": {int(k): int((yhat_rd == k).sum()) for k in range(4)},
        "best_alpha": best["a"],
    }
    print(f"  {label:35} LR    κ_q={m_lr['kappa_q']:.3f} {m_lr['kappa_q_ci']}  acc={m_lr['accuracy']:.3f}")
    print(f"  {label:35} Ridge κ_q={m_rd['kappa_q']:.3f} {m_rd['kappa_q_ci']}  acc={m_rd['accuracy']:.3f}")
    return {"logreg": m_lr, "ridge_ordinal": m_rd, "dim": int(Xtr.shape[1])}


def align_by_clip_id(target_clip_ids, source_clip_ids, source_feats):
    s_idx = {c: i for i, c in enumerate(source_clip_ids)}
    out = np.zeros((len(target_clip_ids), source_feats.shape[1]), dtype=np.float32)
    missing = 0
    for j, c in enumerate(target_clip_ids):
        if c in s_idx:
            out[j] = source_feats[s_idx[c]]
        else:
            missing += 1
    if missing:
        print(f"  WARN: {missing} clips missing")
    return out


def main():
    print("Loading face signals...")
    sig = np.load(SIGNALS, allow_pickle=True)
    clip_ids = sig["clip_id"]
    splits = sig["split"]
    eng = sig["engagement"].astype(np.int64)
    blend = sig["blendshapes"]
    pose = sig["head_pose"]
    gaze = sig["eye_gaze"]
    lm_sum = sig["landmark_summary"]
    all_explicit_raw = np.concatenate([blend, pose, gaze, lm_sum], axis=1)
    print(f"  all_explicit dim={all_explicit_raw.shape[1]}")

    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"

    # Z-score the explicit signals using only Train statistics
    print("  z-scoring explicit signals (train-only stats)...")
    sc = StandardScaler().fit(all_explicit_raw[tr])
    all_explicit = sc.transform(np.nan_to_num(all_explicit_raw))
    blend_z = StandardScaler().fit(blend[tr]).transform(np.nan_to_num(blend))
    pose_z = StandardScaler().fit(pose[tr]).transform(np.nan_to_num(pose))
    gaze_z = StandardScaler().fit(gaze[tr]).transform(np.nan_to_num(gaze))
    lm_z = StandardScaler().fit(lm_sum[tr]).transform(np.nan_to_num(lm_sum))

    # Frozen features (raw, no scaling)
    print("Loading frozen VLM features (raw)...")
    feature_sources = {}
    for label, path in [("SigLIP_L", SIGLIP), ("CLIP_L", CLIPL), ("DINOv2", DINO)]:
        if not os.path.exists(path):
            continue
        d = np.load(path, allow_pickle=True)
        key = "feat" if "feat" in d.files else "cls_feat"
        feature_sources[label] = align_by_clip_id(clip_ids, d["clip_id"], d[key].astype(np.float32))
        print(f"  {label}: {feature_sources[label].shape}")

    print(f"\nSplit sizes: Train={tr.sum()} Val={va.sum()} Test={te.sum()}\n")

    configs = [
        ("blendshapes (z)",                    blend_z),
        ("head_pose (z)",                      pose_z),
        ("eye_gaze (z)",                       gaze_z),
        ("landmark_summary (z)",               lm_z),
        ("all_explicit (z, 73d)",              all_explicit),
    ]
    for name, frozen in feature_sources.items():
        configs.append((f"{name}_cls_raw",                       frozen))
        configs.append((f"EMBER_{name}+explicit_z",              np.concatenate([all_explicit, frozen], axis=1)))
        configs.append((f"EMBER_{name}+blendshapes_only",        np.concatenate([blend_z, frozen], axis=1)))
        configs.append((f"EMBER_{name}+gaze_only",               np.concatenate([gaze_z, frozen], axis=1)))

    results = {}
    for label, X in configs:
        X = np.nan_to_num(X)
        results[label] = probe(X[tr], eng[tr], X[va], eng[va], X[te], eng[te], label)

    with open(RES, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {RES}")


if __name__ == "__main__":
    main()
