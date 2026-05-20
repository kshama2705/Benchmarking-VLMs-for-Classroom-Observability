"""
MOONSHOT 32: Verify α=0.5 blend across multiple bag realizations.

Test α=0.5 blend (per-clip + per-subject E[y]) using:
  - Cached LR-unif K=30×5 (script 80 cache)
  - Super-bag K=50×10 (script 101 cache)
  - Maybe more if available

For each bag, tune α and thresholds via K-fold CV on val, evaluate on test.
Report mean ± std of test κ.

Output:
  results/sota/moonshot_blend_multi_bag.json
"""
import os, json
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
SUPER = os.path.join(BASE, "results", "sota", "_super_bag_cache.npz")
DINO = os.path.join(BASE, "results", "sota", "_dinov2_bag_cache.npz")
MLP = os.path.join(BASE, "results", "sota", "_mlp_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_blend_multi_bag.json")
RNG = np.random.default_rng(42)


def boot_kq(yt, yp, n=1000):
    out = []; nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


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


def evaluate_blend_kfold(p_te, p_va, yte, yva, subj_te, subj_va):
    """For each α, K-fold CV on val. Return mean test κ per α."""
    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    e_va = (p_va * classes[None, :]).sum(1)
    e_te = (p_te * classes[None, :]).sum(1)
    e_va_subj = np.zeros_like(e_va)
    for s in np.unique(subj_va):
        mask = subj_va == s; e_va_subj[mask] = e_va[mask].mean()
    e_te_subj = np.zeros_like(e_te)
    for s in np.unique(subj_te):
        mask = subj_te == s; e_te_subj[mask] = e_te[mask].mean()

    alphas = np.linspace(0, 1, 11)  # coarser grid (11 vs 21) for speed
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    summary = []
    for alpha in alphas:
        e_va_blend = alpha * e_va + (1 - alpha) * e_va_subj
        e_te_blend = alpha * e_te + (1 - alpha) * e_te_subj
        hold_kqs = []; test_kqs = []
        for cal, hold in kf.split(yva):
            bt = tune_thresh(e_va_blend[cal], yva[cal])
            yhat_hold = apply_t(e_va_blend[hold], bt["t"])
            yhat_te = apply_t(e_te_blend, bt["t"])
            hold_kqs.append(cohen_kappa_score(yva[hold], yhat_hold, weights="quadratic"))
            test_kqs.append(cohen_kappa_score(yte, yhat_te, weights="quadratic"))
        summary.append({
            "alpha": float(alpha),
            "hold_mean": float(np.mean(hold_kqs)),
            "test_mean": float(np.mean(test_kqs)),
            "test_std": float(np.std(test_kqs)),
        })
    return summary


def main():
    d = np.load(SIGLIP, allow_pickle=True)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    subj = d["subject_id"]
    va = sp == "Validation"; te = sp == "Test"
    yva = eng[va]; yte = eng[te]
    subj_va = subj[va]; subj_te = subj[te]

    bags = {}
    if os.path.exists(CACHE):
        c = np.load(CACHE)
        bags["LR_unif_K30x5"] = (c["p_te_unif"], c["p_va_unif"])
        bags["LR_RSB45_K30x5"] = (c["p_te_rsb"], c["p_va_rsb"])
    if os.path.exists(SUPER):
        s = np.load(SUPER)
        bags["super_bag_K50x10"] = (s["p_te"], s["p_va"])
    if os.path.exists(DINO):
        s = np.load(DINO)
        bags["DINOv2_K20x5"] = (s["p_te"], s["p_va"])
    if os.path.exists(MLP):
        s = np.load(MLP)
        bags["MLP_K10x5"] = (s["p_te"], s["p_va"])

    out = {}
    for name, (p_te, p_va) in bags.items():
        print(f"\n=== Bag: {name} ===", flush=True)
        summary = evaluate_blend_kfold(p_te, p_va, yte, yva, subj_te, subj_va)
        best_hold = max(summary, key=lambda s: s["hold_mean"])
        print(f"  Best α by hold-out: α={best_hold['alpha']:.2f}  hold={best_hold['hold_mean']:.4f}  test={best_hold['test_mean']:.4f} ± {best_hold['test_std']:.4f}", flush=True)
        out[name] = {"summary": summary, "best_by_hold": best_hold}
        # Show test mean at α=0.5
        a05 = next((s for s in summary if abs(s["alpha"] - 0.5) < 0.01), None)
        if a05:
            print(f"  α=0.50: hold={a05['hold_mean']:.4f}  test={a05['test_mean']:.4f}", flush=True)
        # Show baseline α=1.0
        a10 = next((s for s in summary if abs(s["alpha"] - 1.0) < 0.01), None)
        if a10:
            print(f"  α=1.00: hold={a10['hold_mean']:.4f}  test={a10['test_mean']:.4f}", flush=True)

    # Aggregate across bags: blend at α=0.5 vs α=1.0
    blend_05 = [b["summary"][5]["test_mean"] for b in out.values() if len(b["summary"]) > 5]  # α=0.5 is index 5 in 11-grid
    clip_10 = [b["summary"][-1]["test_mean"] for b in out.values()]
    if blend_05 and clip_10:
        print(f"\nAcross {len(blend_05)} bags:", flush=True)
        print(f"  α=0.5 blend: mean test κ = {np.mean(blend_05):.4f} ± {np.std(blend_05):.4f}", flush=True)
        print(f"  α=1.0 clip:  mean test κ = {np.mean(clip_10):.4f} ± {np.std(clip_10):.4f}", flush=True)
        out["aggregate"] = {
            "blend_05_mean": float(np.mean(blend_05)),
            "blend_05_std": float(np.std(blend_05)),
            "clip_10_mean": float(np.mean(clip_10)),
            "clip_10_std": float(np.std(clip_10)),
            "n_bags": len(blend_05),
        }

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
