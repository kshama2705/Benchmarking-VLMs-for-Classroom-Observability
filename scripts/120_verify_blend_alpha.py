"""
MOONSHOT 31: Verify the α=0.5 blend result via K-fold CV on val.

The hybrid_clip_subj script showed:
  α=0.5 → test κ=0.271 (oracle-best)
  Val-tuned α=1.0 → test κ=0.238

Is α=0.5 a fluke or a real val/test mismatch?

Approach:
  K-fold val: for each α, tune thresholds on K-1 folds, evaluate on held-out fold
  Plot α vs mean held-out κ
  If held-out κ peaks at α=0.5, then val is misrepresentative
  If held-out κ peaks at α=1.0, then α=0.5 is a fluke

Output:
  results/sota/moonshot_verify_blend.json
"""
import os, json
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
from sklearn.model_selection import KFold

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_verify_blend.json")
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


def apply_t(e, t):
    yp = np.zeros_like(e, dtype=int)
    yp[e > t[0]] = 1; yp[e > t[1]] = 2; yp[e > t[2]] = 3
    return yp


def main():
    print("Loading...", flush=True)
    d = np.load(SIGLIP, allow_pickle=True)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    subj = d["subject_id"]
    va = sp == "Validation"; te = sp == "Test"
    yva = eng[va]; yte = eng[te]
    subj_va = subj[va]; subj_te = subj[te]

    c = np.load(CACHE)
    p_te = c["p_te_unif"]; p_va = c["p_va_unif"]
    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    e_va = (p_va * classes[None, :]).sum(1)
    e_te = (p_te * classes[None, :]).sum(1)

    # Compute subject-mean E[y]
    e_va_subj = np.zeros_like(e_va)
    for s in np.unique(subj_va):
        mask = subj_va == s
        e_va_subj[mask] = e_va[mask].mean()
    e_te_subj = np.zeros_like(e_te)
    for s in np.unique(subj_te):
        mask = subj_te == s
        e_te_subj[mask] = e_te[mask].mean()

    out = {}
    alphas = np.linspace(0, 1, 21)

    # Full val tune to baseline
    print("\n=== Full-val test results per α (using clip-level threshold tune per α) ===", flush=True)
    full_results = []
    for alpha in alphas:
        e_va_blend = alpha * e_va + (1 - alpha) * e_va_subj
        e_te_blend = alpha * e_te + (1 - alpha) * e_te_subj
        bt = tune_thresh(e_va_blend, yva)
        yhat = apply_t(e_te_blend, bt["t"])
        kq_te = cohen_kappa_score(yte, yhat, weights="quadratic")
        full_results.append({"alpha": float(alpha), "val_kq": float(bt["v"]),
                              "test_kq": float(kq_te), "t": bt["t"]})
        print(f"  α={alpha:.2f}: val κ={bt['v']:.4f}  test κ={kq_te:.4f}  t={bt['t']}", flush=True)
    out["full_val_per_alpha"] = full_results

    # K-fold CV on val
    print("\n=== K-fold CV (K=5) on val per α ===", flush=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results_by_alpha = {float(a): [] for a in alphas}
    for fold_idx, (cal, hold) in enumerate(kf.split(yva)):
        for alpha in alphas:
            e_va_blend = alpha * e_va + (1 - alpha) * e_va_subj
            # Tune thresholds on cal fold
            bt = tune_thresh(e_va_blend[cal], yva[cal])
            # Evaluate on held-out fold
            yhat_hold = apply_t(e_va_blend[hold], bt["t"])
            kq_hold = cohen_kappa_score(yva[hold], yhat_hold, weights="quadratic")
            # Evaluate on test
            e_te_blend = alpha * e_te + (1 - alpha) * e_te_subj
            yhat_te = apply_t(e_te_blend, bt["t"])
            kq_te = cohen_kappa_score(yte, yhat_te, weights="quadratic")
            fold_results_by_alpha[float(alpha)].append({
                "fold": fold_idx, "hold_kq": float(kq_hold), "test_kq": float(kq_te)
            })
        print(f"  fold {fold_idx} done", flush=True)

    summary = []
    for alpha in alphas:
        recs = fold_results_by_alpha[float(alpha)]
        hold_kqs = np.array([r["hold_kq"] for r in recs])
        test_kqs = np.array([r["test_kq"] for r in recs])
        summary.append({
            "alpha": float(alpha),
            "hold_mean": float(hold_kqs.mean()),
            "hold_std": float(hold_kqs.std()),
            "test_mean": float(test_kqs.mean()),
            "test_std": float(test_kqs.std()),
        })
    out["kfold_summary"] = summary

    print("\n=== K-fold CV results ===", flush=True)
    print("  α     hold_mean ± std   test_mean ± std", flush=True)
    for s in summary:
        print(f"  {s['alpha']:.2f}: {s['hold_mean']:.4f} ± {s['hold_std']:.4f}  {s['test_mean']:.4f} ± {s['test_std']:.4f}", flush=True)

    # Best α by hold-out CV
    best_hold = max(summary, key=lambda s: s["hold_mean"])
    print(f"\nBest α by hold-out CV: α={best_hold['alpha']:.2f}  hold_mean={best_hold['hold_mean']:.4f}  test_mean={best_hold['test_mean']:.4f}", flush=True)
    out["best_by_hold_cv"] = best_hold

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
