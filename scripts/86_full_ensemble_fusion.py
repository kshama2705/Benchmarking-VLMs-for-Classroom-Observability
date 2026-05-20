"""
Final full ensemble: 4 cached probs from prior runs.
  1. SigLIP-L LR uniform bag
  2. SigLIP-L LR RSB-45 bag
  3. SigLIP-L bagged MLP
  4. CLIP-L/14 LR uniform bag

Per-fold tuning (4-way Dirichlet weights + thresholds), majority vote across folds.
This is the final ensemble of best methods found in the search.

Output:
  results/sota/full_ensemble_fusion.json
"""
import os, json, time
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import KFold
from scipy.stats import mode

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
CLIPL_CACHE = os.path.join(BASE, "results", "sota", "_clipl_bag_cache.npz")
SUBJ_CACHE = os.path.join(BASE, "results", "sota", "_subj_bag_cache.npz")
MLP_CACHE = os.path.join(BASE, "results", "sota", "_mlp_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "full_ensemble_fusion.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)


def boot_kq(yt, yp, n=1000):
    out = []
    nt = len(yt)
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


def apply_thresholds(e, t1, t2, t3):
    yp = np.zeros_like(e, dtype=int)
    yp[e > t1] = 1; yp[e > t2] = 2; yp[e > t3] = 3
    return yp


def tune_thresholds(e, y, grid_step=0.02):
    grid = np.arange(0.0, 3.01, grid_step)
    best = None
    for t1 in grid:
        for t2 in grid[grid > t1]:
            for t3 in grid[grid > t2]:
                yp = apply_thresholds(e, t1, t2, t3)
                v = cohen_kappa_score(y, yp, weights="quadratic")
                if best is None or v > best["v"]:
                    best = {"t1": float(t1), "t2": float(t2), "t3": float(t3), "v": float(v)}
    return best


def main():
    print("Loading all cached probs...")
    c = np.load(CACHE); c2 = np.load(CLIPL_CACHE)
    cm = np.load(MLP_CACHE)
    p_va_list = [c["p_va_unif"], c["p_va_rsb"], c2["p_va"], cm["p_va"]]
    p_te_list = [c["p_te_unif"], c["p_te_rsb"], c2["p_te"], cm["p_te"]]
    names = ["LR-unif", "LR-RSB45", "CLIP-L", "MLP"]
    M = len(names)
    print(f"  Members: {names}")

    d = np.load(SIGLIP, allow_pickle=True)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    yva = eng[sp == "Validation"]; yte = eng[sp == "Test"]

    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    out = {"members": names}

    # Per-member solo solo
    print("\n=== Solo per-member (threshold-tuned) ===")
    out["solo"] = {}
    for i, nm in enumerate(names):
        e_va = (p_va_list[i] * classes[None, :]).sum(1)
        e_te = (p_te_list[i] * classes[None, :]).sum(1)
        bt = tune_thresholds(e_va, yva)
        m = metrics(yte, apply_thresholds(e_te, bt["t1"], bt["t2"], bt["t3"]))
        out["solo"][nm] = {**m, **bt}
        print(f"  {nm:>10}: κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  t=({bt['t1']:.2f},{bt['t2']:.2f},{bt['t3']:.2f})")

    # 4-way 5-fold majority vote: per-fold joint search over Dirichlet weights + thresholds
    print("\n=== 4-way 5-fold majority vote ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_records = []
    fold_preds = []
    for fold_idx, (cal_idx, _hold_idx) in enumerate(kf.split(yva)):
        rng = np.random.default_rng(2000 + fold_idx)
        best = None
        for _ in range(25000):
            a = rng.dirichlet(np.ones(M))
            pv = sum(a[i] * p_va_list[i][cal_idx] for i in range(M))
            e = (pv * classes[None, :]).sum(1)
            ts = sorted(rng.uniform(0, 3, size=3))
            yp = apply_thresholds(e, *ts)
            v = cohen_kappa_score(yva[cal_idx], yp, weights="quadratic")
            if best is None or v > best["v"]:
                best = {"w": a.tolist(), "t": ts, "v": float(v)}
        # Apply to test
        pe = sum(best["w"][i] * p_te_list[i] for i in range(M))
        e_te = (pe * classes[None, :]).sum(1)
        yp_te = apply_thresholds(e_te, *best["t"])
        kq_te = float(cohen_kappa_score(yte, yp_te, weights="quadratic"))
        fold_records.append({"fold": fold_idx, "w": best["w"], "t": best["t"], "tune_kq": best["v"], "test_kq": kq_te})
        fold_preds.append(yp_te)
        wstr = ", ".join(f"{nm}={best['w'][i]:.2f}" for i, nm in enumerate(names))
        print(f"  fold {fold_idx}: tune={best['v']:.3f} test={kq_te:.3f}  {wstr}")
    fold_preds = np.stack(fold_preds)
    yp_vote = mode(fold_preds, axis=0, keepdims=False).mode
    m_vote = metrics(yte, yp_vote)
    out["fold_vote_4way"] = m_vote
    out["fold_records_4way"] = fold_records
    print(f"  4-way vote κ_q={m_vote['kappa_q']:.3f} {m_vote['kappa_q_ci']}")

    # Best subset via joint search per combination (Dirichlet)
    print("\n=== Best subset (Dirichlet joint search, 15K iter per combo) ===")
    from itertools import combinations
    best_sub = None
    subset_records = []
    for r in range(2, M+1):
        for combo in combinations(range(M), r):
            rng = np.random.default_rng(2026 + sum(combo))
            best_combo = None
            for _ in range(15000):
                a = rng.dirichlet(np.ones(r))
                pv = sum(a[j] * p_va_list[combo[j]] for j in range(r))
                e = (pv * classes[None, :]).sum(1)
                ts = sorted(rng.uniform(0, 3, size=3))
                yp = apply_thresholds(e, *ts)
                v = cohen_kappa_score(yva, yp, weights="quadratic")
                if best_combo is None or v > best_combo["v"]:
                    best_combo = {"w": a.tolist(), "t": ts, "v": float(v)}
            pe = sum(best_combo["w"][j] * p_te_list[combo[j]] for j in range(r))
            e_te = (pe * classes[None, :]).sum(1)
            yp_te = apply_thresholds(e_te, *best_combo["t"])
            test_kq = float(cohen_kappa_score(yte, yp_te, weights="quadratic"))
            combo_names = [names[i] for i in combo]
            rec = {"combo": combo_names, "w": best_combo["w"], "t": best_combo["t"],
                    "val_kq": best_combo["v"], "test_kq": test_kq}
            subset_records.append(rec)
            print(f"  {'+'.join(combo_names):>30}  val={best_combo['v']:.3f} test={test_kq:.3f}")
            if best_sub is None or best_combo["v"] > best_sub["val_kq"]:
                best_sub = rec
    print(f"  Best by val: {'+'.join(best_sub['combo'])}  val={best_sub['val_kq']:.3f}  test={best_sub['test_kq']:.3f}")
    out["best_subset"] = best_sub
    out["subset_records"] = subset_records

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
