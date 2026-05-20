"""
MOONSHOT 25: Nearest-train-subject anchored prediction.

For each test subject:
  1. Compute their feature centroid (mean SigLIP-L feature)
  2. Find K-nearest train subjects by centroid distance
  3. Their average engagement label = prior estimate
  4. Combine with model's per-subject E[y]: shrink toward prior

This uses train labels at the subject-distribution level.

Output:
  results/sota/moonshot_nearest_subject.json
"""
import os, json, time
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_nearest_subject.json")
RNG = np.random.default_rng(42)


def boot_kq(yt, yp, n=1000):
    out = []; nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def metrics(yt, yp):
    return {
        "kappa_q": float(cohen_kappa_score(yt, yp, weights="quadratic")),
        "kappa_q_ci": boot_kq(yt, yp),
        "accuracy": float(accuracy_score(yt, yp)),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "pred_dist": {int(k): int((yp == k).sum()) for k in range(4)},
    }


def main():
    print("Loading...", flush=True)
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    subj = d["subject_id"]
    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    Xtr = X[tr]; Xva = X[va]; Xte = X[te]
    ytr = eng[tr]; yva = eng[va]; yte = eng[te]
    subj_tr = subj[tr]; subj_va = subj[va]; subj_te = subj[te]

    # Compute per-subject centroids and mean engagement labels
    print("Computing centroids...", flush=True)
    tr_subj = np.unique(subj_tr)
    va_subj = np.unique(subj_va)
    te_subj = np.unique(subj_te)

    tr_centroid = {}
    tr_mean_eng = {}
    for s in tr_subj:
        mask = subj_tr == s
        tr_centroid[s] = Xtr[mask].mean(axis=0)
        tr_mean_eng[s] = ytr[mask].mean()
    tr_centroids = np.array([tr_centroid[s] for s in tr_subj])
    tr_mean_engs = np.array([tr_mean_eng[s] for s in tr_subj])
    print(f"  Train: {len(tr_subj)} subjects, mean engagement range [{tr_mean_engs.min():.2f}, {tr_mean_engs.max():.2f}]", flush=True)

    te_centroid = {}
    for s in te_subj:
        mask = subj_te == s
        te_centroid[s] = Xte[mask].mean(axis=0)

    # Model's per-subject E[y]
    c = np.load(CACHE)
    p_te = c["p_te_unif"]; p_va = c["p_va_unif"]
    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    e_te = (p_te * classes[None, :]).sum(1)
    e_va = (p_va * classes[None, :]).sum(1)

    te_model_e = {s: e_te[subj_te == s].mean() for s in te_subj}

    # Global thresholds
    grid = np.arange(0, 3.01, 0.04)
    best = None
    for t1 in grid:
        for t2 in grid[grid > t1]:
            for t3 in grid[grid > t2]:
                yp = np.zeros_like(yva)
                yp[e_va > t1] = 1; yp[e_va > t2] = 2; yp[e_va > t3] = 3
                v = cohen_kappa_score(yva, yp, weights='quadratic')
                if best is None or v > best['v']:
                    best = {'t': (t1, t2, t3), 'v': v}
    t = best["t"]
    print(f"Global clip thresholds: {t}, val κ={best['v']:.3f}", flush=True)

    out = {}

    # ============ Baseline: model E[y] per subject ============
    yhat_base = np.zeros_like(yte)
    for s in te_subj:
        mask = subj_te == s
        sm = te_model_e[s]
        c_pred = 0
        if sm > t[0]: c_pred = 1
        if sm > t[1]: c_pred = 2
        if sm > t[2]: c_pred = 3
        yhat_base[mask] = c_pred
    m_base = metrics(yte, yhat_base)
    out["baseline"] = m_base
    print(f"\nBaseline per-subject: κ_q={m_base['kappa_q']:.4f}", flush=True)

    # ============ For each test subject, find K-NN train subjects ============
    print("\n[A] K-NN anchored prediction...", flush=True)
    # Compute distances
    te_anchors = {}  # s_te -> avg label of K-nearest train subjects
    for K in [1, 3, 5, 10, 20]:
        for s_te in te_subj:
            c_te = te_centroid[s_te]
            dists = np.linalg.norm(tr_centroids - c_te[None, :], axis=1)
            k_idx = np.argsort(dists)[:K]
            anchor = tr_mean_engs[k_idx].mean()
            te_anchors[s_te] = anchor

        # Just use anchor as the prediction
        yhat = np.zeros_like(yte)
        for s in te_subj:
            mask = subj_te == s
            sm = te_anchors[s]
            c_pred = 0
            if sm > t[0]: c_pred = 1
            if sm > t[1]: c_pred = 2
            if sm > t[2]: c_pred = 3
            yhat[mask] = c_pred
        m = metrics(yte, yhat)
        out[f"anchor_only_K{K}"] = m
        print(f"  Anchor K={K}: κ_q={m['kappa_q']:.4f} {m['kappa_q_ci']}", flush=True)

    # ============ Combine: shrink model E[y] toward anchor ============
    print("\n[B] Shrink model E[y] toward K-NN anchor...", flush=True)
    for K in [5, 10, 20]:
        for s_te in te_subj:
            c_te = te_centroid[s_te]
            dists = np.linalg.norm(tr_centroids - c_te[None, :], axis=1)
            k_idx = np.argsort(dists)[:K]
            te_anchors[s_te] = tr_mean_engs[k_idx].mean()
        for alpha in [0.1, 0.3, 0.5, 0.7]:
            yhat = np.zeros_like(yte)
            for s in te_subj:
                mask = subj_te == s
                sm = (1 - alpha) * te_model_e[s] + alpha * te_anchors[s]
                c_pred = 0
                if sm > t[0]: c_pred = 1
                if sm > t[1]: c_pred = 2
                if sm > t[2]: c_pred = 3
                yhat[mask] = c_pred
            m = metrics(yte, yhat)
            out[f"shrink_K{K}_alpha{alpha}"] = m
            print(f"  K={K} alpha={alpha}: κ_q={m['kappa_q']:.4f} {m['kappa_q_ci']}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
