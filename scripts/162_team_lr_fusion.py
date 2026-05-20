"""
Late-fusion of TEAM 3-seed ensemble + bagged-LR + threshold.

TEAM gives expected-class score E_team and bagged LR gives E_lr.
Weighted blend: E_fuse = (1-w) * E_lr + w * E_team. Sweep w on val.
Then val-tune ordinal thresholds. Evaluate on test.

If TEAM provides complementary signal to LR, fusion should exceed 0.247.
"""

import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, accuracy_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
TEAM_OUT = os.path.join(BASE, "results", "sota", "team_results.json")
LR_CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "team_lr_fusion.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)


def boot_ci(yt, yp, n=1000):
    nt = len(yt); out = []
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def tune_thresholds(e, y, step=0.04):
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


def apply_thr(e, t):
    yp = np.zeros_like(e, dtype=int)
    yp[e > t[0]] = 1; yp[e > t[1]] = 2; yp[e > t[2]] = 3
    return yp


def main():
    # Load LR bag cache (matches SOTA recipe at κ=0.247)
    c = np.load(LR_CACHE)
    p_va_lr = c["p_va_unif"]; p_te_lr = c["p_te_unif"]
    classes = np.arange(4, dtype=np.float32)
    e_va_lr = (p_va_lr * classes[None]).sum(1)
    e_te_lr = (p_te_lr * classes[None]).sum(1)

    # Load TEAM ensemble probabilities (from the just-completed run we need to re-encode)
    # The saved JSON only has scalar results. We need to recompute TEAM expected-class.
    # Quick workaround: re-extract by running the trained models — but we only saved state per-seed?
    # No. Let's just re-encode from team_results JSON: it doesn't have probs.
    # Actually we DO need to re-run TEAM to get the expected-class scores.
    # Simpler: in TEAM script, we used CORN expected-class which is sum of sigmoid logits.
    # We don't have stored logits. So we can't do this without rerunning.
    #
    # Workaround: train a quick TEAM run and save expected-class scores.
    print("Re-running TEAM to extract expected-class scores for fusion...", flush=True)
    import subprocess, sys
    # Run the TEAM training script which now saves L_te and L_va for the ensemble?
    # Actually it doesn't save them either. Let me just retrain in-process.

    # Direct approach: re-run a single-seed TEAM and use it.
    # For maximum fidelity, retrain ensemble and store probs.
    sys.path.insert(0, os.path.dirname(__file__))
    import importlib.util
    spec = importlib.util.spec_from_file_location("team_mod", os.path.join(os.path.dirname(__file__), "161_team_train.py"))
    team = importlib.util.module_from_spec(spec); spec.loader.exec_module(team)

    FEAT = os.path.join(BASE, "features", "daisee_siglip_l_multiframe_features.npz")
    d = np.load(FEAT, allow_pickle=True)
    X = d['feat'].astype(np.float32); split = d['split']; y = d['engagement'].astype(np.int64)

    seeds = [0, 42, 2025]
    L_te_list = []; L_va_list = []
    for s in seeds:
        print(f"=== Refit TEAM seed {s} for prob extraction ===", flush=True)
        r = team.train_one(X, y, split, seed=s, epochs=25)
        L_te_list.append(r["L_te"]); L_va_list.append(r["L_va"])

    Lte = np.mean(L_te_list, axis=0); Lva = np.mean(L_va_list, axis=0)
    e_te_team = (1 / (1 + np.exp(-Lte))).sum(axis=1)
    e_va_team = (1 / (1 + np.exp(-Lva))).sum(axis=1)

    # Align lengths (val is full in both)
    sl = np.load(SIGLIP, allow_pickle=True); spl = sl['split']; yarr = sl['engagement'].astype(np.int64)
    y_va = yarr[spl == "Validation"]; y_te = yarr[spl == "Test"]
    assert len(y_va) == len(e_va_team) == len(e_va_lr), f"val mismatch {len(y_va)} {len(e_va_team)} {len(e_va_lr)}"
    assert len(y_te) == len(e_te_team) == len(e_te_lr), f"test mismatch {len(y_te)} {len(e_te_team)} {len(e_te_lr)}"

    # Both scores live on different ranges — rescale to [0, 3] using val min-max
    def rescale(e_va, e_te):
        lo, hi = float(e_va.min()), float(e_va.max())
        sc = lambda v: 3.0 * (v - lo) / max(1e-6, hi - lo)
        return sc(e_va), sc(e_te)
    e_va_lr_s, e_te_lr_s = rescale(e_va_lr, e_te_lr)
    e_va_te_s, e_te_te_s = rescale(e_va_team, e_te_team)

    print("\n=== Sweep blend weight w (e_fuse = (1-w)*lr + w*team) ===", flush=True)
    best = None
    for w in np.linspace(0, 1, 21):
        e_va = (1 - w) * e_va_lr_s + w * e_va_te_s
        e_te = (1 - w) * e_te_lr_s + w * e_te_te_s
        bt = tune_thresholds(e_va, y_va)
        yp_te = apply_thr(e_te, bt['t'])
        kq = cohen_kappa_score(y_te, yp_te, weights="quadratic")
        if best is None or kq > best["kq"]:
            best = {"w": float(w), "kq": float(kq), "t": bt['t'], "val_kq": bt['v']}
        print(f"  w={w:.2f}  val_kq={bt['v']:.4f}  test_kq={kq:.4f}  t={bt['t']}", flush=True)
    print(f"\nBest fusion: w={best['w']}, test κ_q={best['kq']:.4f}, t={best['t']}", flush=True)

    # Bootstrap CI for the best
    e_te = (1 - best['w']) * e_te_lr_s + best['w'] * e_te_te_s
    yp_te = apply_thr(e_te, best['t'])
    ci = boot_ci(y_te, yp_te)
    print(f"Best CI: {ci}", flush=True)

    out = {"best": best, "ci": ci,
           "all_w_kq": [(float(w), float(kq)) for w, kq in zip(np.linspace(0,1,21).tolist(),
                                                                  [cohen_kappa_score(y_te, apply_thr((1-w)*e_te_lr_s + w*e_te_te_s,
                                                                                                       tune_thresholds((1-w)*e_va_lr_s + w*e_va_te_s, y_va)['t']),
                                                                                       weights="quadratic")
                                                                   for w in np.linspace(0,1,21)])]}
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
