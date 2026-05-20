"""
TTA + bagged LR-unif + threshold tuning.

Apply each trained LR-unif bag classifier to all 4 TTA test variants
(original, hflip, crop_center_110, crop_center_90), average across bags AND
across TTA variants, then threshold-tune the resulting E[y].

Prior TTA result (solo LR, no bagging, no threshold): 0.199 → 0.205 with TTA.
This combines TTA with the SOTA recipe (bagged + threshold).

Output:
  results/sota/tta_bagged_solo.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
TTA = os.path.join(BASE, "features", "daisee_siglip_l_tta_test.npz")
OUT = os.path.join(BASE, "results", "sota", "tta_bagged_solo.json")
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


def bag_lr_with_tta(Xtr, ytr, Xva, yva, Xte_variants, K=20, seeds=[0, 7, 42, 2025, 1024]):
    """Train LR bag, predict on each TTA variant separately."""
    n_variants = len(Xte_variants)
    all_te = [[] for _ in range(n_variants)]  # one list per TTA variant
    all_va = []
    for s in seeds:
        rng = np.random.default_rng(s)
        bv = []
        bt = [[] for _ in range(n_variants)]
        for k in range(K):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            clf = fit_lr(Xtr[idx], ytr[idx], Xva, yva)
            bv.append(clf.predict_proba(Xva))
            for vi, Xte in enumerate(Xte_variants):
                bt[vi].append(clf.predict_proba(Xte))
            if (k + 1) % 10 == 0:
                print(f"    seed={s} k={k+1}/{K}", flush=True)
        all_va.append(np.stack(bv).mean(0))
        for vi in range(n_variants):
            all_te[vi].append(np.stack(bt[vi]).mean(0))
    p_va = np.stack(all_va).mean(0)
    p_te_per_variant = [np.stack(all_te[vi]).mean(0) for vi in range(n_variants)]
    return p_va, p_te_per_variant


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


def apply_thresh(e, t):
    yp = np.zeros_like(e, dtype=int)
    yp[e > t[0]] = 1; yp[e > t[1]] = 2; yp[e > t[2]] = 3
    return yp


def main():
    print("Loading features + TTA...", flush=True)
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    Xtr, ytr = X[tr], eng[tr]; Xva, yva = X[va], eng[va]; Xte, yte = X[te], eng[te]

    tta = np.load(TTA)
    variants = ["original", "hflip", "crop_center_110", "crop_center_90"]
    Xte_variants = [tta[v].astype(np.float32) for v in variants]
    # Verify "original" matches our Xte ordering (clip-by-clip)
    # (TTA features should follow same test ordering as in SIGLIP — they were extracted from same clips)
    diff = np.abs(Xte - Xte_variants[0]).mean()
    print(f"  Original-TTA vs SIGLIP test feat: mean abs diff = {diff:.6f}", flush=True)
    if diff > 0.1:
        print("  WARNING: features differ — TTA ordering may not match!", flush=True)

    classes = np.array([0, 1, 2, 3], dtype=np.float32)

    print(f"\nTraining bag with TTA application (K=20 × 5 seeds × {len(variants)} TTA variants)...", flush=True)
    t0 = time.time()
    p_va, p_te_variants = bag_lr_with_tta(Xtr, ytr, Xva, yva, Xte_variants, K=20)
    print(f"  Done ({time.time()-t0:.0f}s)", flush=True)

    out = {"variants": variants}

    print("\n=== Per-variant solo + threshold ===", flush=True)
    e_va = (p_va * classes[None, :]).sum(1)
    bt_full = tune_thresholds(e_va, yva, step=0.04)
    print(f"  Full val tune: t={bt_full['t']}  val κ={bt_full['v']:.4f}", flush=True)
    for vi, vname in enumerate(variants):
        p_te = p_te_variants[vi]
        e_te = (p_te * classes[None, :]).sum(1)
        # Use SAME thresholds tuned on val (since val is original features only)
        yp = apply_thresh(e_te, bt_full['t'])
        m = metrics(yte, yp)
        out[f"variant_{vname}"] = {**m, "t": bt_full['t']}
        print(f"  {vname:>20}: κ_q={m['kappa_q']:.4f} {m['kappa_q_ci']}", flush=True)

    # TTA average (4-variant mean)
    print("\n=== TTA-averaged + threshold ===", flush=True)
    p_te_tta = np.mean(p_te_variants, axis=0)
    e_te_tta = (p_te_tta * classes[None, :]).sum(1)
    yp_tta = apply_thresh(e_te_tta, bt_full['t'])
    m_tta = metrics(yte, yp_tta)
    out["tta_mean4_threshold"] = {**m_tta, "t": bt_full['t']}
    print(f"  TTA-mean κ_q={m_tta['kappa_q']:.4f} {m_tta['kappa_q_ci']}", flush=True)

    # Geometric mean across TTA
    p_te_geo = np.exp(np.log(np.stack(p_te_variants) + 1e-12).mean(0))
    p_te_geo /= p_te_geo.sum(1, keepdims=True)
    e_te_geo = (p_te_geo * classes[None, :]).sum(1)
    yp_geo = apply_thresh(e_te_geo, bt_full['t'])
    m_geo = metrics(yte, yp_geo)
    out["tta_geom4_threshold"] = {**m_geo}
    print(f"  TTA-geom κ_q={m_geo['kappa_q']:.4f} {m_geo['kappa_q_ci']}", flush=True)

    # Tune thresholds re-on val (still original) and apply to TTA test (no val TTA available)
    # — same as above since val unchanged. Just for sanity.

    # 3-of-4: drop crop_center_90 (most aggressive crop)
    print("\n=== TTA 3-of-4 (drop crop_center_90) ===", flush=True)
    p_te_3 = np.mean([p_te_variants[0], p_te_variants[1], p_te_variants[2]], axis=0)
    e_te_3 = (p_te_3 * classes[None, :]).sum(1)
    yp_3 = apply_thresh(e_te_3, bt_full['t'])
    m_3 = metrics(yte, yp_3)
    out["tta_3of4_threshold"] = m_3
    print(f"  3-of-4: κ_q={m_3['kappa_q']:.4f} {m_3['kappa_q_ci']}", flush=True)

    # Weighted average via search (since val TTA not available, search on test directly is cheating)
    # Skip this — we only have val for original.

    # Save bag probs cache
    cache = os.path.join(BASE, "results", "sota", "_tta_bag_cache.npz")
    np.savez_compressed(cache, p_va=p_va,
                        p_te_original=p_te_variants[0], p_te_hflip=p_te_variants[1],
                        p_te_c110=p_te_variants[2], p_te_c90=p_te_variants[3])

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
