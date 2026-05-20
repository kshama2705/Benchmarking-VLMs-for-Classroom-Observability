"""
Fusion probe combining explicit face signals (AU-equivalent blendshapes, head
pose, eye gaze, landmark summary) with frozen-VLM CLS features.

Hypothesis: explicit identity-invariant signals (AUs/pose/gaze) carry orthogonal
information to frozen VLM features. Their fusion should break the κ ≈ 0.20
ceiling that pure frozen-VLM methods hit.

Method name: EMBER — Engagement via Multi-Behavioral Explicit Routing.

Steps:
  1. Load face signals (blendshapes, head pose, eye gaze, landmark summary).
  2. Load SigLIP-L and CLIP-L CLS features (already cached from overnight).
  3. Standardize explicit features (z-score).
  4. Run linear probe (LR + Ridge ordinal) on all combinations:
       - blendshapes alone
       - head_pose alone
       - eye_gaze alone
       - landmark_summary alone
       - all-explicit (concat 73-dim)
       - SigLIP-L CLS alone (reference baseline)
       - all-explicit + SigLIP-L CLS (the EMBER candidate)
       - all-explicit + CLIP-L CLS
  5. Subject-disjoint Train/Val/Test splits + bootstrap CIs.

Output:
  results/ember/fusion_results.json
"""
import os, json, time
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
RES = os.path.join(OUT_DIR, "fusion_results.json")
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
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr); Xva_s = sc.transform(Xva); Xte_s = sc.transform(Xte)

    best = None
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr_s, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva_s), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": v, "clf": clf}
    yhat_lr = best["clf"].predict(Xte_s)
    m_lr = {
        "kappa_q": float(cohen_kappa_score(yte, yhat_lr, weights="quadratic")),
        "kappa_q_ci": boot_kq(yte, yhat_lr),
        "accuracy": float(accuracy_score(yte, yhat_lr)),
        "f1_macro": float(f1_score(yte, yhat_lr, average="macro", zero_division=0)),
        "pred_dist": {int(k): int((yhat_lr == k).sum()) for k in range(4)},
        "best_C": best["C"],
    }
    best = None
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(Xtr_s, ytr.astype(np.float32))
        v = cohen_kappa_score(yva, np.clip(np.round(reg.predict(Xva_s)), 0, 3).astype(int), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"a": alpha, "v": v, "reg": reg}
    yhat_rd = np.clip(np.round(best["reg"].predict(Xte_s)), 0, 3).astype(int)
    m_rd = {
        "kappa_q": float(cohen_kappa_score(yte, yhat_rd, weights="quadratic")),
        "kappa_q_ci": boot_kq(yte, yhat_rd),
        "accuracy": float(accuracy_score(yte, yhat_rd)),
        "f1_macro": float(f1_score(yte, yhat_rd, average="macro", zero_division=0)),
        "pred_dist": {int(k): int((yhat_rd == k).sum()) for k in range(4)},
        "best_alpha": best["a"],
    }
    print(f"  {label:30} LR    κ_q={m_lr['kappa_q']:.3f} {m_lr['kappa_q_ci']}  acc={m_lr['accuracy']:.3f}")
    print(f"  {label:30} Ridge κ_q={m_rd['kappa_q']:.3f} {m_rd['kappa_q_ci']}  acc={m_rd['accuracy']:.3f}")
    return {"logreg": m_lr, "ridge_ordinal": m_rd, "dim": int(Xtr.shape[1])}


def align_by_clip_id(target_clip_ids, source_clip_ids, source_feats):
    """Align source features to target clip_ids order."""
    s_idx = {c: i for i, c in enumerate(source_clip_ids)}
    out = np.zeros((len(target_clip_ids), source_feats.shape[1]), dtype=np.float32)
    missing = 0
    for j, c in enumerate(target_clip_ids):
        if c in s_idx:
            out[j] = source_feats[s_idx[c]]
        else:
            missing += 1
    if missing:
        print(f"  WARN: {missing} clips missing in source")
    return out


def main():
    print("Loading face signals...")
    sig = np.load(SIGNALS, allow_pickle=True)
    clip_ids = sig["clip_id"]
    splits = sig["split"]
    eng = sig["engagement"].astype(np.int64)
    detected = sig["detected"]
    blend = sig["blendshapes"]      # (N, 52)
    pose = sig["head_pose"]          # (N, 3)
    gaze = sig["eye_gaze"]           # (N, 6)
    lm_sum = sig["landmark_summary"] # (N, 12)
    print(f"  N={len(clip_ids)}, detected={detected.sum()} ({100*detected.sum()/len(clip_ids):.1f}%)")
    print(f"  blendshapes shape: {blend.shape}, head_pose: {pose.shape}, gaze: {gaze.shape}")

    all_explicit = np.concatenate([blend, pose, gaze, lm_sum], axis=1)
    print(f"  all_explicit dim: {all_explicit.shape[1]}")

    print("\nLoading frozen VLM features (SigLIP-L, CLIP-L, DINOv2)...")
    feature_sources = {}
    if os.path.exists(SIGLIP):
        d = np.load(SIGLIP, allow_pickle=True)
        feature_sources["SigLIP_L"] = align_by_clip_id(clip_ids, d["clip_id"], d["feat"].astype(np.float32))
    if os.path.exists(CLIPL):
        d = np.load(CLIPL, allow_pickle=True)
        # daisee_clip_l_14_features uses "cls_feat" key (from PPEP) — but we also have feat from overnight
        if "feat" in d.files:
            feature_sources["CLIP_L"] = align_by_clip_id(clip_ids, d["clip_id"], d["feat"].astype(np.float32))
        elif "cls_feat" in d.files:
            feature_sources["CLIP_L"] = align_by_clip_id(clip_ids, d["clip_id"], d["cls_feat"].astype(np.float32))
    if os.path.exists(DINO):
        d = np.load(DINO, allow_pickle=True)
        feature_sources["DINOv2"] = align_by_clip_id(clip_ids, d["clip_id"], d["feat"].astype(np.float32))

    for k, v in feature_sources.items():
        print(f"  {k}: {v.shape}")

    # Subject-disjoint masks
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    print(f"\nSplit sizes: Train={tr.sum()} Val={va.sum()} Test={te.sum()}")

    # Replace NaN/Inf with 0 in all features
    def clean(X):
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    results = {}
    configs = [
        ("blendshapes",          blend),
        ("head_pose",            pose),
        ("eye_gaze",             gaze),
        ("landmark_summary",     lm_sum),
        ("all_explicit",         all_explicit),
    ]
    for name, frozen_feat in feature_sources.items():
        configs.append((f"{name}_cls_only", frozen_feat))
        configs.append((f"EMBER_{name}+explicit", np.concatenate([all_explicit, frozen_feat], axis=1)))

    print("\nRunning fusion probes...")
    for label, X in configs:
        X = clean(X)
        results[label] = probe(X[tr], eng[tr], X[va], eng[va], X[te], eng[te], label)

    with open(RES, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {RES}")


if __name__ == "__main__":
    main()
