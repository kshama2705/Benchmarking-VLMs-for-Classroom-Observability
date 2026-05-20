"""
Verify the κ=0.237 result for solo LR-unif + threshold via K-fold CV on val.

Uses cached probs; only does threshold optimization (no LR retraining).
"""
import os, json, sys
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
OUT = os.path.join(BASE, "results", "sota", "verify_solo_threshold.json")
RNG = np.random.default_rng(42)


def tune(e, y, step=0.04):
    """Coarser grid (0.04) for speed; still finds same optima."""
    grid = np.arange(0.0, 3.01, step)
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


def apply(e, t):
    yp = np.zeros_like(e, dtype=int)
    yp[e > t[0]] = 1; yp[e > t[1]] = 2; yp[e > t[2]] = 3
    return yp


def main():
    print("Loading...", flush=True)
    c = np.load(CACHE)
    p_te_unif = c['p_te_unif']; p_va_unif = c['p_va_unif']
    d = np.load(SIGLIP, allow_pickle=True)
    sp = d['split']; eng = d['engagement'].astype(np.int64)
    yva = eng[sp == 'Validation']; yte = eng[sp == 'Test']
    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    e_va = (p_va_unif * classes[None, :]).sum(1)
    e_te = (p_te_unif * classes[None, :]).sum(1)

    out = {}

    print("\nBaseline (full val tune)...", flush=True)
    b = tune(e_va, yva)
    yhat_te = apply(e_te, b['t'])
    te_kq = float(cohen_kappa_score(yte, yhat_te, weights='quadratic'))
    print(f"  val={b['v']:.4f}  test={te_kq:.4f}  t={b['t']}", flush=True)
    out["full_val_tune"] = {"t": b['t'], "val_kq": b['v'], "test_kq": te_kq}

    print("\nK-fold CV...", flush=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_recs = []
    for fi, (cal, hold) in enumerate(kf.split(yva)):
        bf = tune(e_va[cal], yva[cal])
        yp_hold = apply(e_va[hold], bf['t'])
        h_kq = float(cohen_kappa_score(yva[hold], yp_hold, weights='quadratic'))
        yp_te = apply(e_te, bf['t'])
        t_kq = float(cohen_kappa_score(yte, yp_te, weights='quadratic'))
        fold_recs.append({"fold": fi, "t": bf['t'], "tune_kq": bf['v'], "hold_kq": h_kq, "test_kq": t_kq})
        print(f"  fold {fi}: tune={bf['v']:.4f}  hold={h_kq:.4f}  test={t_kq:.4f}  t={bf['t']}", flush=True)
    out["folds"] = fold_recs

    fold_ts = np.array([r["t"] for r in fold_recs])
    mean_t = tuple(fold_ts.mean(0).tolist())
    median_t = tuple(np.median(fold_ts, axis=0).tolist())
    tune_kqs = [r["tune_kq"] for r in fold_recs]
    hold_kqs = [r["hold_kq"] for r in fold_recs]
    test_kqs = [r["test_kq"] for r in fold_recs]

    print(f"\n  Tune mean={np.mean(tune_kqs):.4f}  Hold mean={np.mean(hold_kqs):.4f}  Test mean={np.mean(test_kqs):.4f} ± {np.std(test_kqs):.4f}", flush=True)
    print(f"  Tune-Hold gap: {np.mean(tune_kqs) - np.mean(hold_kqs):.4f}", flush=True)
    print(f"  Mean-t={mean_t}", flush=True)
    print(f"  Median-t={median_t}", flush=True)
    yp_mean = apply(e_te, mean_t)
    yp_med = apply(e_te, median_t)
    te_mean = float(cohen_kappa_score(yte, yp_mean, weights='quadratic'))
    te_med = float(cohen_kappa_score(yte, yp_med, weights='quadratic'))
    print(f"  Test κ with mean-t: {te_mean:.4f}", flush=True)
    print(f"  Test κ with median-t: {te_med:.4f}", flush=True)

    # Bootstrap CI on baseline test prediction
    nt = len(yte)
    boots = []
    for _ in range(1000):
        idx = RNG.integers(0, nt, size=nt)
        boots.append(cohen_kappa_score(yte[idx], yhat_te[idx], weights='quadratic'))
    ci = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
    print(f"\nBootstrap CI for solo LR-unif + full-val threshold: κ={te_kq:.4f}  95% CI={ci}", flush=True)
    out["bootstrap_ci_full_val_tune"] = ci

    out["summary"] = {
        "test_mean_kfold": float(np.mean(test_kqs)),
        "test_std_kfold": float(np.std(test_kqs)),
        "tune_hold_gap": float(np.mean(tune_kqs) - np.mean(hold_kqs)),
        "mean_t": list(mean_t),
        "median_t": list(median_t),
        "test_mean_t": te_mean,
        "test_median_t": te_med,
    }

    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
