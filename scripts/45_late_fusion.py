"""
EMBER-LateFusion — late fusion of probe predictions.

Train two separate linear probes:
  - Probe A: LR on explicit signals only (73-d)
  - Probe B: LR on frozen CLS (1024-d SigLIP-L or 768-d CLIP-L/DINOv2)

At inference, combine via:
  - Mean of class-probabilities
  - Weighted mean (sweep weight on validation)
  - Stacked LR meta-classifier (train a small LR on the two probes' soft outputs)

Output:
  results/ember/late_fusion_results.json
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
OUT = os.path.join(BASE, "results", "ember", "late_fusion_results.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)


def boot_kq(yt, yp, n=1000):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def fit_lr_probe(Xtr, ytr, Xva, yva):
    """Tune C on val, return (clf, val_kq)."""
    best = None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": float(v), "clf": clf}
    return best["clf"], best["v"]


def metrics(yt, yp, label=""):
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
    sig = np.load(SIGNALS, allow_pickle=True)
    clip_ids = sig["clip_id"]; splits = sig["split"]
    eng = sig["engagement"].astype(np.int64)
    explicit_raw = np.nan_to_num(np.concatenate([sig["blendshapes"], sig["head_pose"],
                                                  sig["eye_gaze"], sig["landmark_summary"]], axis=1))
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    sc = StandardScaler().fit(explicit_raw[tr])
    explicit = sc.transform(explicit_raw).astype(np.float32)

    feats = {}
    for label, path in [("SigLIP_L", SIGLIP), ("CLIP_L", CLIPL), ("DINOv2", DINO)]:
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            key = "feat" if "feat" in d.files else "cls_feat"
            feats[label] = align(clip_ids, d["clip_id"], d[key].astype(np.float32))
    print(f"  Loaded: {list(feats.keys())}")

    out = {}

    print("\n=== Probe A: explicit signals only ===")
    clf_exp, val_exp = fit_lr_probe(explicit[tr], eng[tr], explicit[va], eng[va])
    p_exp_va = clf_exp.predict_proba(explicit[va])
    p_exp_te = clf_exp.predict_proba(explicit[te])
    yhat_exp = p_exp_te.argmax(axis=1)
    m_exp = metrics(eng[te], yhat_exp)
    out["explicit_alone"] = {**m_exp, "val_kq": val_exp, "best_C": clf_exp.C}
    print(f"  explicit_alone  κ_q={m_exp['kappa_q']:.3f} {m_exp['kappa_q_ci']}  val={val_exp:.3f}")

    for enc_label, cls_feats in feats.items():
        print(f"\n=== {enc_label}: ===")
        clf_cls, val_cls = fit_lr_probe(cls_feats[tr], eng[tr], cls_feats[va], eng[va])
        p_cls_va = clf_cls.predict_proba(cls_feats[va])
        p_cls_te = clf_cls.predict_proba(cls_feats[te])
        yhat_cls = p_cls_te.argmax(axis=1)
        m_cls = metrics(eng[te], yhat_cls)
        out[f"{enc_label}_alone"] = {**m_cls, "val_kq": val_cls, "best_C": clf_cls.C}
        print(f"  {enc_label}_alone  κ_q={m_cls['kappa_q']:.3f} {m_cls['kappa_q_ci']}  val={val_cls:.3f}")

        # ===== Mean fusion =====
        p_mean_va = (p_exp_va + p_cls_va) / 2
        p_mean_te = (p_exp_te + p_cls_te) / 2
        yhat_mean = p_mean_te.argmax(axis=1)
        m_mean = metrics(eng[te], yhat_mean)
        out[f"{enc_label}_mean_fusion"] = m_mean
        print(f"  {enc_label}_mean_fusion  κ_q={m_mean['kappa_q']:.3f} {m_mean['kappa_q_ci']}")

        # ===== Weighted fusion (sweep weight) =====
        best_w = None
        for w in np.linspace(0.0, 1.0, 21):
            p_va = w * p_exp_va + (1 - w) * p_cls_va
            yhat_va = p_va.argmax(axis=1)
            v = cohen_kappa_score(eng[va], yhat_va, weights="quadratic")
            if best_w is None or v > best_w["v"]:
                best_w = {"w": float(w), "v": float(v)}
        w_best = best_w["w"]
        p_w_te = w_best * p_exp_te + (1 - w_best) * p_cls_te
        yhat_w = p_w_te.argmax(axis=1)
        m_w = metrics(eng[te], yhat_w)
        out[f"{enc_label}_weighted_fusion"] = {**m_w, "best_w": w_best, "val_kq": best_w["v"]}
        print(f"  {enc_label}_weighted_fusion (w={w_best:.2f})  κ_q={m_w['kappa_q']:.3f} {m_w['kappa_q_ci']}  val={best_w['v']:.3f}")

        # ===== Stacked meta-classifier =====
        # Train meta-LR on val (probe outputs are inputs, eng[va] is target)
        meta_X_va = np.concatenate([p_exp_va, p_cls_va], axis=1)
        meta_X_te = np.concatenate([p_exp_te, p_cls_te], axis=1)
        meta_clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000,
                                       solver="lbfgs", random_state=42)
        # We need to fit meta on train predictions, not on val. Re-fit base probes on a sub-split.
        # Approach: use cross-fitting on train to get unbiased train probabilities.
        # For simplicity here, fit meta on val (small but OK for 4-class).
        meta_clf.fit(meta_X_va, eng[va])
        yhat_meta = meta_clf.predict(meta_X_te)
        m_meta = metrics(eng[te], yhat_meta)
        out[f"{enc_label}_stacked"] = m_meta
        print(f"  {enc_label}_stacked     κ_q={m_meta['kappa_q']:.3f} {m_meta['kappa_q_ci']}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
