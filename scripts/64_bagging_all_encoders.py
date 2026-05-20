"""
Apply bagging (K=10) to all encoders to see if breakthrough generalizes.

For each encoder (SigLIP-L, SigLIP-SO400M, CLIP-L, DINOv2):
  - Bootstrap train K=10 times
  - Train LR on each bootstrap (val-tuned C)
  - Average test predictions
  - Report κ_q with bootstrap CI

Also: bagged + EMBER late fusion (face_explicit added)

Output:
  results/sota/bagging_all_encoders.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP_L = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
SIGLIP_SO400M = os.path.join(BASE, "features", "daisee_siglip_so400m_features.npz")
CLIPL = os.path.join(BASE, "features", "daisee_clip_l_14_features.npz")
DINO = os.path.join(BASE, "features", "dinov2_vitb14_features.npz")
SIGNALS = os.path.join(BASE, "features", "daisee_face_signals.npz")
OUT = os.path.join(BASE, "results", "sota", "bagging_all_encoders.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)

K_BAGS = 10
OUTER_SEED = 0


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


def bag_probe(X, eng, tr_mask, va_mask, te_mask, K, outer_seed):
    """Returns averaged test probability matrix."""
    rng = np.random.default_rng(outer_seed)
    Xtr = X[tr_mask]; ytr = eng[tr_mask]
    Xva = X[va_mask]; yva = eng[va_mask]
    Xte = X[te_mask]
    n_tr = len(ytr)
    p_te_bags = []
    p_va_bags = []
    for k in range(K):
        idx = rng.integers(0, n_tr, size=n_tr)
        clf, _, _ = fit_lr(Xtr[idx], ytr[idx], Xva, yva)
        p_te_bags.append(clf.predict_proba(Xte))
        p_va_bags.append(clf.predict_proba(Xva))
    return np.stack(p_te_bags).mean(axis=0), np.stack(p_va_bags).mean(axis=0)


def main():
    print("Loading...")
    # Use SigLIP-L manifest as canonical (all encoder feature files use same clip_ids)
    d = np.load(SIGLIP_L, allow_pickle=True)
    clip_ids = d["clip_id"]; splits = d["split"]; eng = d["engagement"].astype(np.int64)
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    print(f"  Train={tr.sum()} Val={va.sum()} Test={te.sum()}")

    encoders = {"SigLIP_L": SIGLIP_L, "SigLIP_SO400M": SIGLIP_SO400M,
                "CLIP_L": CLIPL, "DINOv2": DINO}
    feats = {}
    for label, path in encoders.items():
        if not os.path.exists(path):
            print(f"  MISSING: {label} at {path}")
            continue
        dd = np.load(path, allow_pickle=True)
        key = "feat" if "feat" in dd.files else "cls_feat"
        feats[label] = align(clip_ids, dd["clip_id"], dd[key].astype(np.float32))
        print(f"  {label}: {feats[label].shape}")

    # Explicit signals
    d_sf = np.load(SIGNALS, allow_pickle=True)
    explicit_raw = np.nan_to_num(np.concatenate([d_sf["blendshapes"], d_sf["head_pose"],
                                                  d_sf["eye_gaze"], d_sf["landmark_summary"]], axis=1))
    explicit = align(clip_ids, d_sf["clip_id"], explicit_raw)
    sc = StandardScaler().fit(explicit[tr])
    explicit_z = sc.transform(explicit).astype(np.float32)
    print(f"  explicit (z): {explicit_z.shape}")

    out = {}

    print(f"\n=== Bagging K={K_BAGS}, outer_seed={OUTER_SEED} ===")
    for enc_label, X in feats.items():
        print(f"\n--- {enc_label} ---")
        # Solo (no bag)
        clf, _, _ = fit_lr(X[tr], eng[tr], X[va], eng[va])
        yhat = clf.predict(X[te])
        m_solo = metrics(eng[te], yhat)
        out[f"{enc_label}_solo"] = m_solo
        print(f"  solo  κ_q={m_solo['kappa_q']:.3f} {m_solo['kappa_q_ci']}")
        # Bagged
        p_te, p_va = bag_probe(X, eng, tr, va, te, K_BAGS, OUTER_SEED)
        yhat_bag = p_te.argmax(axis=1)
        m_bag = metrics(eng[te], yhat_bag)
        out[f"{enc_label}_bagged"] = m_bag
        print(f"  bagged κ_q={m_bag['kappa_q']:.3f} {m_bag['kappa_q_ci']}")

        # Bagged + EMBER late fusion (combine with explicit probe — also bagged)
        p_exp_te, p_exp_va = bag_probe(explicit_z, eng, tr, va, te, K_BAGS, OUTER_SEED + 100)
        best_w = None
        for w in np.linspace(0.0, 1.0, 41):
            pv = w * p_exp_va + (1 - w) * p_va
            v = cohen_kappa_score(eng[va], pv.argmax(axis=1), weights="quadratic")
            if best_w is None or v > best_w["v"]:
                best_w = {"w": float(w), "v": float(v)}
        wb = best_w["w"]
        p_fuse = wb * p_exp_te + (1 - wb) * p_te
        yhat_fuse = p_fuse.argmax(axis=1)
        m_fuse = metrics(eng[te], yhat_fuse)
        out[f"{enc_label}_bagged_EMBER"] = {**m_fuse, "w": wb, "val_kq": best_w["v"]}
        print(f"  bagged+EMBER (w={wb:.2f})  κ_q={m_fuse['kappa_q']:.3f} {m_fuse['kappa_q_ci']}  val={best_w['v']:.3f}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
