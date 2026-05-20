"""
Verify the 3-way joint search SOTA (κ=0.252) is real, not a 40K-trial random-search artifact.

Approaches:
[1] K-fold CV on val: tune (w, t) on K-1 folds, evaluate on held-out fold + test.
    If joint search overfits val, held-out val and test will degrade significantly.
[2] Repeat joint random search with different seeds. If the optimum is robust,
    different seeds find similar (w, t) and similar test κ.
[3] Constrained optimization: scipy.minimize on coordinate-descent space.
    More disciplined than random search.

Output:
  results/sota/verify_3way.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import KFold

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CLIPL = os.path.join(BASE, "features", "daisee_clip_l_14_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
CLIPL_CACHE = os.path.join(BASE, "results", "sota", "_clipl_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "verify_3way.json")
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


def joint_random_search(p_va_list, y_va, n_iter=40000, seed=42, classes=np.array([0,1,2,3], dtype=np.float32)):
    rng = np.random.default_rng(seed)
    M = len(p_va_list)
    best = None
    for _ in range(n_iter):
        a = rng.dirichlet(np.ones(M))
        pv = sum(w * p for w, p in zip(a, p_va_list))
        e = (pv * classes[None, :]).sum(1)
        ts = sorted(rng.uniform(0, 3, size=3))
        yp = apply_thresholds(e, *ts)
        v = cohen_kappa_score(y_va, yp, weights="quadratic")
        if best is None or v > best["v"]:
            best = {"w": tuple(map(float, a)), "t": tuple(map(float, ts)), "v": float(v)}
    return best


def main():
    print("Loading caches + labels...")
    c = np.load(CACHE); c2 = np.load(CLIPL_CACHE)
    p_te_unif = c["p_te_unif"]; p_va_unif = c["p_va_unif"]
    p_te_rsb = c["p_te_rsb"]; p_va_rsb = c["p_va_rsb"]
    p_te_clipl = c2["p_te"]; p_va_clipl = c2["p_va"]
    d = np.load(SIGLIP, allow_pickle=True)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    yva = eng[sp == "Validation"]; yte = eng[sp == "Test"]
    classes = np.array([0, 1, 2, 3], dtype=np.float32)

    out = {}

    # ============ [1] Seed-stability of joint random search ============
    print("\n=== Seed-stability of joint random search ===")
    seed_results = []
    for s in [42, 0, 7, 2025, 1024]:
        b = joint_random_search([p_va_unif, p_va_rsb, p_va_clipl], yva, n_iter=40000, seed=s)
        # Apply to test
        pe = b["w"][0]*p_te_unif + b["w"][1]*p_te_rsb + b["w"][2]*p_te_clipl
        e_te = (pe * classes[None, :]).sum(1)
        yp_te = apply_thresholds(e_te, *b["t"])
        kq_te = float(cohen_kappa_score(yte, yp_te, weights="quadratic"))
        seed_results.append({"seed": s, "val_kq": b["v"], "test_kq": kq_te,
                              "w": b["w"], "t": b["t"]})
        print(f"  seed {s}: val={b['v']:.3f}  test={kq_te:.3f}  w={[f'{x:.2f}' for x in b['w']]}  t={[f'{x:.2f}' for x in b['t']]}")
    out["seed_stability"] = seed_results
    test_kqs = np.array([r["test_kq"] for r in seed_results])
    val_kqs = np.array([r["val_kq"] for r in seed_results])
    out["seed_stability_summary"] = {
        "test_mean": float(test_kqs.mean()), "test_std": float(test_kqs.std()),
        "val_mean": float(val_kqs.mean()), "val_std": float(val_kqs.std()),
        "test_min": float(test_kqs.min()), "test_max": float(test_kqs.max()),
    }
    print(f"  Test κ mean={test_kqs.mean():.3f} ± {test_kqs.std():.3f}  (min={test_kqs.min():.3f}, max={test_kqs.max():.3f})")
    print(f"  Val κ mean={val_kqs.mean():.3f} ± {val_kqs.std():.3f}")

    # ============ [2] K-fold CV on val: holdout test ============
    print("\n=== 5-fold CV on val (tune on K-1 folds, eval on held-out + test) ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    for fold_idx, (cal_idx, hold_idx) in enumerate(kf.split(yva)):
        # Tune (w, t) on cal_idx
        b = joint_random_search(
            [p_va_unif[cal_idx], p_va_rsb[cal_idx], p_va_clipl[cal_idx]],
            yva[cal_idx], n_iter=20000, seed=42 + fold_idx
        )
        # Evaluate on held-out val
        pv_hold = b["w"][0]*p_va_unif[hold_idx] + b["w"][1]*p_va_rsb[hold_idx] + b["w"][2]*p_va_clipl[hold_idx]
        e_hold = (pv_hold * classes[None, :]).sum(1)
        yp_hold = apply_thresholds(e_hold, *b["t"])
        kq_hold = float(cohen_kappa_score(yva[hold_idx], yp_hold, weights="quadratic"))
        # Evaluate on test
        pe = b["w"][0]*p_te_unif + b["w"][1]*p_te_rsb + b["w"][2]*p_te_clipl
        e_te = (pe * classes[None, :]).sum(1)
        yp_te = apply_thresholds(e_te, *b["t"])
        kq_te = float(cohen_kappa_score(yte, yp_te, weights="quadratic"))
        fold_results.append({
            "fold": fold_idx, "tune_val_kq": b["v"], "hold_val_kq": kq_hold, "test_kq": kq_te,
            "w": b["w"], "t": b["t"],
        })
        print(f"  fold {fold_idx}: tune={b['v']:.3f}  hold={kq_hold:.3f}  test={kq_te:.3f}  w={[f'{x:.2f}' for x in b['w']]}")
    out["kfold_cv"] = fold_results
    hold_kqs = np.array([r["hold_val_kq"] for r in fold_results])
    test_kqs = np.array([r["test_kq"] for r in fold_results])
    tune_kqs = np.array([r["tune_val_kq"] for r in fold_results])
    out["kfold_summary"] = {
        "tune_mean": float(tune_kqs.mean()), "tune_std": float(tune_kqs.std()),
        "hold_mean": float(hold_kqs.mean()), "hold_std": float(hold_kqs.std()),
        "test_mean": float(test_kqs.mean()), "test_std": float(test_kqs.std()),
        "overfitting_gap": float(tune_kqs.mean() - hold_kqs.mean()),
    }
    print(f"  Tune mean={tune_kqs.mean():.3f}  Hold mean={hold_kqs.mean():.3f}  Test mean={test_kqs.mean():.3f}")
    print(f"  Overfit gap (tune − hold): {tune_kqs.mean() - hold_kqs.mean():.3f}")

    # ============ [3] Mean-of-folds prediction ensemble ============
    print("\n=== Use mean of K-fold (w, t) → single prediction ===")
    mean_w = np.array([r["w"] for r in fold_results]).mean(0)
    mean_t = np.array([r["t"] for r in fold_results]).mean(0)
    mean_w = mean_w / mean_w.sum()
    mean_t = np.sort(mean_t)
    pe_mean = mean_w[0]*p_te_unif + mean_w[1]*p_te_rsb + mean_w[2]*p_te_clipl
    e_te_mean = (pe_mean * classes[None, :]).sum(1)
    yp_te_mean = apply_thresholds(e_te_mean, *mean_t)
    m_mean = metrics(yte, yp_te_mean)
    out["mean_fold_params"] = {**m_mean, "w": mean_w.tolist(), "t": mean_t.tolist()}
    print(f"  Mean w={mean_w}  t={mean_t}")
    print(f"  κ_q={m_mean['kappa_q']:.3f} {m_mean['kappa_q_ci']}")

    # ============ [4] Vote of K-fold preds ============
    print("\n=== Ensemble: majority vote of K-fold predictions on test ===")
    fold_preds = []
    for r in fold_results:
        pe = r["w"][0]*p_te_unif + r["w"][1]*p_te_rsb + r["w"][2]*p_te_clipl
        e = (pe * classes[None, :]).sum(1)
        yp = apply_thresholds(e, *r["t"])
        fold_preds.append(yp)
    fold_preds = np.stack(fold_preds)  # (5, 1784)
    # Mode
    from scipy.stats import mode
    yp_vote = mode(fold_preds, axis=0, keepdims=False).mode
    m_vote = metrics(yte, yp_vote)
    out["fold_vote"] = m_vote
    print(f"  κ_q={m_vote['kappa_q']:.3f} {m_vote['kappa_q_ci']}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
