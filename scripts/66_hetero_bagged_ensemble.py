"""
Heterogeneous-encoder bagged ensemble.

Combine bagged predictions from SigLIP-L (best solo), CLIP-L, DINOv2.
Hypothesis: different encoders bring diverse predictions; bagged version of each
captures within-encoder ensemble variance; late fusion combines them.

Pipeline:
  - For each encoder, K=30 bags × 5 outer seeds → mean prediction
  - Late-fuse across encoders with val-tuned weights

Output:
  results/sota/hetero_bagged.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS = {
    "SigLIP_L":   os.path.join(BASE, "features", "daisee_siglip_l_features.npz"),
    "CLIP_L":     os.path.join(BASE, "features", "daisee_clip_l_14_features.npz"),
    "DINOv2":     os.path.join(BASE, "features", "dinov2_vitb14_features.npz"),
}
OUT = os.path.join(BASE, "results", "sota", "hetero_bagged.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)
OUTER_SEEDS = [0, 7, 42, 2025, 1024]
K_BAGS = 30


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


def bagged_predictions(X, eng, tr, va, te, outer_seeds=OUTER_SEEDS, K=K_BAGS):
    """Mega-ensemble: average across outer_seeds × K bags."""
    Xtr = X[tr]; ytr = eng[tr]
    Xva = X[va]; yva = eng[va]
    Xte = X[te]
    n_tr = len(ytr)
    all_p_va, all_p_te = [], []
    for outer_seed in outer_seeds:
        rng = np.random.default_rng(outer_seed)
        seed_va = []; seed_te = []
        for _ in range(K):
            idx = rng.integers(0, n_tr, size=n_tr)
            clf, _, _ = fit_lr(Xtr[idx], ytr[idx], Xva, yva)
            seed_va.append(clf.predict_proba(Xva))
            seed_te.append(clf.predict_proba(Xte))
        all_p_va.append(np.stack(seed_va).mean(axis=0))
        all_p_te.append(np.stack(seed_te).mean(axis=0))
    return np.stack(all_p_va).mean(axis=0), np.stack(all_p_te).mean(axis=0)


def main():
    print("Loading...")
    d0 = np.load(PATHS["SigLIP_L"], allow_pickle=True)
    clip_ids = d0["clip_id"]; splits = d0["split"]; eng = d0["engagement"].astype(np.int64)
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    yte = eng[te]; yva = eng[va]

    feats = {}
    for label, path in PATHS.items():
        d = np.load(path, allow_pickle=True)
        key = "feat" if "feat" in d.files else "cls_feat"
        feats[label] = align(clip_ids, d["clip_id"], d[key].astype(np.float32))
        print(f"  {label}: {feats[label].shape}")

    out = {}
    bagged_p_va = {}
    bagged_p_te = {}
    for label, X in feats.items():
        print(f"\nBagging {label} ({len(OUTER_SEEDS)} seeds × K={K_BAGS} = {len(OUTER_SEEDS)*K_BAGS} probes)...")
        t0 = time.time()
        p_va, p_te = bagged_predictions(X, eng, tr, va, te)
        bagged_p_va[label] = p_va
        bagged_p_te[label] = p_te
        yhat = p_te.argmax(axis=1)
        m = metrics(yte, yhat)
        out[f"{label}_bagged_mega"] = m
        dt = time.time() - t0
        print(f"  {label} bagged mega: κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  ({dt:.0f}s)")

    # Pair fusions of bagged probes
    print("\n=== Pair fusions of bagged probes ===")
    pairs = [("SigLIP_L", "CLIP_L"), ("SigLIP_L", "DINOv2"), ("CLIP_L", "DINOv2")]
    for a, b in pairs:
        best_w = None
        for w in np.linspace(0.0, 1.0, 41):
            pv = w * bagged_p_va[a] + (1 - w) * bagged_p_va[b]
            v = cohen_kappa_score(yva, pv.argmax(axis=1), weights="quadratic")
            if best_w is None or v > best_w["v"]:
                best_w = {"w": float(w), "v": float(v)}
        wb = best_w["w"]
        p_te = wb * bagged_p_te[a] + (1 - wb) * bagged_p_te[b]
        m = metrics(yte, p_te.argmax(axis=1))
        out[f"pair_{a}+{b}"] = {**m, "w": wb, "val_kq": best_w["v"]}
        print(f"  {a}+{b} (w={wb:.2f}): κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  val={best_w['v']:.3f}")

    # Triple fusion (scalar simplex via grid)
    print("\n=== Triple fusion (SigLIP_L + CLIP_L + DINOv2) ===")
    best_t = None
    pa, pb, pc = bagged_p_va["SigLIP_L"], bagged_p_va["CLIP_L"], bagged_p_va["DINOv2"]
    pat, pbt, pct = bagged_p_te["SigLIP_L"], bagged_p_te["CLIP_L"], bagged_p_te["DINOv2"]
    G = 11
    for w1 in np.linspace(0, 1, G):
        for w2 in np.linspace(0, 1 - w1, G):
            w3 = 1 - w1 - w2
            pv = w1 * pa + w2 * pb + w3 * pc
            v = cohen_kappa_score(yva, pv.argmax(axis=1), weights="quadratic")
            if best_t is None or v > best_t["v"]:
                best_t = {"ws": [float(w1), float(w2), float(w3)], "v": float(v)}
    ws = best_t["ws"]
    p_te = ws[0] * pat + ws[1] * pbt + ws[2] * pct
    m = metrics(yte, p_te.argmax(axis=1))
    out["triple_SigLIP+CLIP+DINO"] = {**m, "weights": ws, "val_kq": best_t["v"]}
    print(f"  triple ws=[{ws[0]:.2f},{ws[1]:.2f},{ws[2]:.2f}]: κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  val={best_t['v']:.3f}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
