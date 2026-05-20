"""
MOONSHOT 26: Verify per-subject SOTA across multiple bag realizations.

The κ=0.2421 came from a single LR-unif bag (K=20 × 5 seeds = 100 fits). Per-subject
baseline averages over 15-30 clips per subject, so should be more stable than per-clip.
But it's still subject to bag training noise. Multi-seed verification.

Output:
  results/sota/moonshot_verify_per_subj.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_verify_per_subj.json")
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


def fit_lr(Xtr, ytr, Xva, yva):
    best = None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": float(v), "clf": clf}
    return best["clf"]


def bag_lr(Xtr, ytr, Xva, yva, Xte, K=20, seeds=None):
    all_te, all_va = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        bt, bv = [], []
        for k in range(K):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            clf = fit_lr(Xtr[idx], ytr[idx], Xva, yva)
            bt.append(clf.predict_proba(Xte)); bv.append(clf.predict_proba(Xva))
        all_te.append(np.stack(bt).mean(0)); all_va.append(np.stack(bv).mean(0))
    return np.stack(all_te).mean(0), np.stack(all_va).mean(0)


def tune_thresh(e, y, step=0.04):
    grid = np.arange(0, 3.01, step)
    best = None
    for t1 in grid:
        for t2 in grid[grid > t1]:
            for t3 in grid[grid > t2]:
                yp = np.zeros_like(e, dtype=int)
                yp[e > t1] = 1; yp[e > t2] = 2; yp[e > t3] = 3
                v = cohen_kappa_score(y, yp, weights='quadratic')
                if best is None or v > best['v']:
                    best = {'t': (float(t1), float(t2), float(t3)), 'v': float(v)}
    return best


def per_subject(p_te, p_va, yte, yva, subj_te, classes, t):
    e_te = (p_te * classes[None, :]).sum(1)
    yhat = np.zeros_like(yte)
    for s in np.unique(subj_te):
        mask = subj_te == s
        subj_e = e_te[mask].mean()
        c_pred = 0
        if subj_e > t[0]: c_pred = 1
        if subj_e > t[1]: c_pred = 2
        if subj_e > t[2]: c_pred = 3
        yhat[mask] = c_pred
    return metrics(yte, yhat)


def main():
    print("Loading...", flush=True)
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    subj = d["subject_id"]
    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    Xtr, ytr = X[tr], eng[tr]; Xva, yva = X[va], eng[va]; Xte, yte = X[te], eng[te]
    subj_te = subj[te]

    classes = np.array([0, 1, 2, 3], dtype=np.float32)

    seed_groups = [
        [0, 7, 42, 2025, 1024],     # original
        [100, 200, 300, 400, 500],
        [1, 11, 21, 31, 41],
        [99, 199, 299, 399, 499],
        [123, 234, 345, 456, 567],
    ]

    results = []
    for gi, seeds in enumerate(seed_groups):
        print(f"\n=== Seed group {gi}: {seeds} ===", flush=True)
        t0 = time.time()
        p_te, p_va = bag_lr(Xtr, ytr, Xva, yva, Xte, K=20, seeds=seeds)
        print(f"  Bag done ({time.time()-t0:.0f}s)", flush=True)
        e_va = (p_va * classes[None, :]).sum(1)
        bt = tune_thresh(e_va, yva)
        # Per-subject
        m_subj = per_subject(p_te, p_va, yte, yva, subj_te, classes, bt["t"])
        # Per-clip threshold
        e_te = (p_te * classes[None, :]).sum(1)
        yp_clip = np.zeros_like(yte)
        yp_clip[e_te > bt["t"][0]] = 1
        yp_clip[e_te > bt["t"][1]] = 2
        yp_clip[e_te > bt["t"][2]] = 3
        m_clip = metrics(yte, yp_clip)
        results.append({
            "seed_group": seeds,
            "thresholds": bt["t"],
            "val_kq_clip": bt["v"],
            "test_kq_clip_thresh": m_clip["kappa_q"],
            "test_kq_per_subject": m_subj["kappa_q"],
            "test_kq_per_subject_ci": m_subj["kappa_q_ci"],
        })
        print(f"  Thresh: {bt['t']}", flush=True)
        print(f"  Per-clip thresh test: κ={m_clip['kappa_q']:.4f}", flush=True)
        print(f"  Per-subject test: κ={m_subj['kappa_q']:.4f} {m_subj['kappa_q_ci']}", flush=True)

    out = {
        "seed_groups": results,
        "summary": {
            "per_subject_test_kappas": [r["test_kq_per_subject"] for r in results],
            "per_subject_mean": float(np.mean([r["test_kq_per_subject"] for r in results])),
            "per_subject_std": float(np.std([r["test_kq_per_subject"] for r in results])),
            "per_clip_mean": float(np.mean([r["test_kq_clip_thresh"] for r in results])),
            "per_clip_std": float(np.std([r["test_kq_clip_thresh"] for r in results])),
        }
    }
    print(f"\nSummary:", flush=True)
    print(f"  Per-subject mean κ={out['summary']['per_subject_mean']:.4f} ± {out['summary']['per_subject_std']:.4f}", flush=True)
    print(f"  Per-clip mean κ={out['summary']['per_clip_mean']:.4f} ± {out['summary']['per_clip_std']:.4f}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
