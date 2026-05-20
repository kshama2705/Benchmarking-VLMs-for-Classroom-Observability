"""
MOONSHOT 24: Hierarchical 2-stage per-subject classification.

Stage 1: Binary "low engagement (L0/L1) vs high (L2/L3)" per subject
Stage 2: Within each binary group, predict L0 vs L1, or L2 vs L3

This decomposes the 4-class problem into easier binary tasks.

Output:
  results/sota/moonshot_hierarchical.json
"""
import os, json, time
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_hierarchical.json")
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
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    subj = d["subject_id"]
    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    yva = eng[va]; yte = eng[te]
    subj_va = subj[va]; subj_te = subj[te]

    c = np.load(CACHE)
    p_te = c["p_te_unif"]; p_va = c["p_va_unif"]
    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    e_va = (p_va * classes[None, :]).sum(1)
    e_te = (p_te * classes[None, :]).sum(1)

    # Compute per-subject means on val and test
    unique_va = np.unique(subj_va)
    unique_te = np.unique(subj_te)
    va_subj_e = {s: e_va[subj_va == s].mean() for s in unique_va}
    va_subj_y = {s: int(np.round(yva[subj_va == s].mean())) for s in unique_va}
    te_subj_e = {s: e_te[subj_te == s].mean() for s in unique_te}

    print(f"Val subjects: {len(unique_va)}")
    print(f"  Distribution: {np.bincount([va_subj_y[s] for s in unique_va], minlength=4).tolist()}")

    out = {}

    # ============ Stage 1: 4-class threshold tuning at subject level ============
    # (This is the baseline = moonshot 21 subj_thresh_subj_mean)
    va_e_arr = np.array([va_subj_e[s] for s in unique_va])
    va_y_arr = np.array([va_subj_y[s] for s in unique_va])

    # Tune 3 thresholds on val subjects
    best = None
    grid = np.arange(0, 3.01, 0.04)
    for t1 in grid:
        for t2 in grid[grid > t1]:
            for t3 in grid[grid > t2]:
                yp = np.zeros_like(va_y_arr)
                yp[va_e_arr > t1] = 1; yp[va_e_arr > t2] = 2; yp[va_e_arr > t3] = 3
                v = cohen_kappa_score(va_y_arr, yp, weights="quadratic")
                if best is None or v > best["v"]:
                    best = {"t": (t1, t2, t3), "v": v}
    print(f"\nSubj-tuned thresholds: {best['t']}  val κ={best['v']:.3f}", flush=True)

    # ============ Stage 1 alt: Just 1 threshold (binary low vs high) ============
    # Try classifying subjects as L01 vs L23 (single threshold)
    print("\n[A] Binary subject classification: L01 vs L23", flush=True)
    va_y_bin = (va_y_arr >= 2).astype(int)
    best_t_bin = None
    for t in grid:
        yp = (va_e_arr > t).astype(int)
        v = (yp == va_y_bin).mean()
        if best_t_bin is None or v > best_t_bin["v"]:
            best_t_bin = {"t": t, "v": v}
    print(f"  Binary L01/L23 threshold: t={best_t_bin['t']:.3f}, val acc={best_t_bin['v']:.3f}", flush=True)

    # Within L23: tune threshold L2 vs L3
    mask_L23 = va_y_arr >= 2
    if mask_L23.sum() > 1:
        va_e_L23 = va_e_arr[mask_L23]
        va_y_L23 = (va_y_arr[mask_L23] == 3).astype(int)
        best_t_L23 = None
        for t in grid:
            yp = (va_e_L23 > t).astype(int)
            v = (yp == va_y_L23).mean()
            if best_t_L23 is None or v > best_t_L23["v"]:
                best_t_L23 = {"t": t, "v": v}
        print(f"  L2/L3 sub-threshold: t={best_t_L23['t']:.3f}, val acc={best_t_L23['v']:.3f}", flush=True)
    else:
        best_t_L23 = {"t": 2.5, "v": 0}

    # Within L01: tune threshold L0 vs L1 — but only 1-2 val subjects might be L0
    mask_L01 = va_y_arr < 2
    if mask_L01.sum() > 1 and (va_y_arr[mask_L01] == 0).sum() > 0:
        va_e_L01 = va_e_arr[mask_L01]
        va_y_L01 = (va_y_arr[mask_L01] == 1).astype(int)
        best_t_L01 = None
        for t in grid:
            yp = (va_e_L01 > t).astype(int)
            v = (yp == va_y_L01).mean()
            if best_t_L01 is None or v > best_t_L01["v"]:
                best_t_L01 = {"t": t, "v": v}
        print(f"  L0/L1 sub-threshold: t={best_t_L01['t']:.3f}, val acc={best_t_L01['v']:.3f}", flush=True)
    else:
        best_t_L01 = {"t": 0.5, "v": 0}

    # Apply hierarchical to test
    yhat_te = np.zeros_like(yte)
    for s in unique_te:
        mask = subj_te == s
        sm = te_subj_e[s]
        if sm > best_t_bin["t"]:
            # In L23 group
            if sm > best_t_L23["t"]:
                yhat_te[mask] = 3
            else:
                yhat_te[mask] = 2
        else:
            # In L01 group
            if sm > best_t_L01["t"]:
                yhat_te[mask] = 1
            else:
                yhat_te[mask] = 0
    m_hier = metrics(yte, yhat_te)
    out["hierarchical_2stage"] = m_hier
    out["t_bin"] = best_t_bin["t"]
    out["t_L23"] = best_t_L23["t"]
    out["t_L01"] = best_t_L01["t"]
    print(f"\nHierarchical κ_q={m_hier['kappa_q']:.4f} {m_hier['kappa_q_ci']}", flush=True)

    # ============ Baseline for comparison: clip-level thresh from val ============
    best_clip = None
    for t1 in grid:
        for t2 in grid[grid > t1]:
            for t3 in grid[grid > t2]:
                yp = np.zeros_like(yva)
                yp[e_va > t1] = 1; yp[e_va > t2] = 2; yp[e_va > t3] = 3
                v = cohen_kappa_score(yva, yp, weights='quadratic')
                if best_clip is None or v > best_clip["v"]:
                    best_clip = {"t": (t1, t2, t3), "v": v}
    yhat_base = np.zeros_like(yte)
    for s in unique_te:
        mask = subj_te == s
        sm = te_subj_e[s]
        c_pred = 0
        if sm > best_clip["t"][0]: c_pred = 1
        if sm > best_clip["t"][1]: c_pred = 2
        if sm > best_clip["t"][2]: c_pred = 3
        yhat_base[mask] = c_pred
    m_base = metrics(yte, yhat_base)
    out["baseline_per_subject"] = m_base
    print(f"\nBaseline per-subject: κ_q={m_base['kappa_q']:.4f}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
