"""
MOONSHOT 21: Per-subject calibration v2.

Improvements:
  - Tune thresholds on SUBJECT MEANS (not per-clip) via val
  - Each val subject's mean engagement label = average of clips per subject
  - Threshold-tune subject-mean E[y] against subject-mean true label
  - Apply to test subject means
  - Try multiple probe sources

Output:
  results/sota/moonshot_per_subject_v2.json
"""
import os, json, time
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
CLIPL_CACHE = os.path.join(BASE, "results", "sota", "_clipl_bag_cache.npz")
DINO_CACHE = os.path.join(BASE, "results", "sota", "_dinov2_bag_cache.npz")
MLP_CACHE = os.path.join(BASE, "results", "sota", "_mlp_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_per_subject_v2.json")
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


def per_subject_classify(p_va, p_te, yva, yte, subj_va, subj_te, source_name, classes):
    """Apply per-subject baseline classification.
    Returns dict of metrics for v1 (clip-level threshold on subject mean) and v2 (subject-level threshold tuning).
    """
    e_va = (p_va * classes[None, :]).sum(1)
    e_te = (p_te * classes[None, :]).sum(1)

    out = {}

    # ============ Variant A: Clip-level threshold applied to subject mean ============
    bt_clip = tune_thresh(e_va, yva)
    # Compute test subject means
    unique_te = np.unique(subj_te)
    yhat_te = np.zeros_like(yte)
    for s in unique_te:
        mask = subj_te == s
        subj_e = e_te[mask].mean()
        c_pred = 0
        if subj_e > bt_clip["t"][0]: c_pred = 1
        if subj_e > bt_clip["t"][1]: c_pred = 2
        if subj_e > bt_clip["t"][2]: c_pred = 3
        yhat_te[mask] = c_pred
    m_a = metrics(yte, yhat_te)
    out["clip_thresh_subj_mean"] = {**m_a, **bt_clip}

    # ============ Variant B: Subject-level threshold tuning ============
    # On val: aggregate per-subject (mean E[y], modal label)
    unique_va = np.unique(subj_va)
    va_subj_e = []
    va_subj_y = []
    for s in unique_va:
        mask = subj_va == s
        va_subj_e.append(e_va[mask].mean())
        va_subj_y.append(int(np.round(yva[mask].mean())))  # round to int for kappa
    va_subj_e = np.array(va_subj_e)
    va_subj_y = np.array(va_subj_y)
    bt_subj = tune_thresh(va_subj_e, va_subj_y)
    # Apply to test subject means
    yhat_te2 = np.zeros_like(yte)
    for s in unique_te:
        mask = subj_te == s
        subj_e = e_te[mask].mean()
        c_pred = 0
        if subj_e > bt_subj["t"][0]: c_pred = 1
        if subj_e > bt_subj["t"][1]: c_pred = 2
        if subj_e > bt_subj["t"][2]: c_pred = 3
        yhat_te2[mask] = c_pred
    m_b = metrics(yte, yhat_te2)
    out["subj_thresh_subj_mean"] = {**m_b, **bt_subj}

    # ============ Variant C: Subject-level thresh on MEAN, applied per-clip ============
    # Use subject-level thresholds, but apply to per-clip e_te
    yhat_te3 = apply_t(e_te, bt_subj["t"])
    m_c = metrics(yte, yhat_te3)
    out["subj_thresh_per_clip"] = {**m_c, **bt_subj}

    # Diagnostics
    pred_subj_dist = {}
    for s in unique_te:
        mask = subj_te == s
        true_class = int(np.round(yte[mask].mean()))
        pred_class = yhat_te[mask][0]  # all clips of subject same prediction
        key = f"true{true_class}_pred{pred_class}"
        pred_subj_dist[key] = pred_subj_dist.get(key, 0) + 1
    out["subj_confusion_va"] = pred_subj_dist

    return out


def main():
    print("Loading...", flush=True)
    d = np.load(SIGLIP, allow_pickle=True)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    subj = d["subject_id"]
    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    yva = eng[va]; yte = eng[te]
    subj_va = subj[va]; subj_te = subj[te]

    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    out = {}

    sources = []
    if os.path.exists(CACHE):
        c = np.load(CACHE)
        sources.append(("LR-unif", c["p_te_unif"], c["p_va_unif"]))
        sources.append(("LR-RSB45", c["p_te_rsb"], c["p_va_rsb"]))
    if os.path.exists(CLIPL_CACHE):
        c2 = np.load(CLIPL_CACHE)
        sources.append(("CLIP-L", c2["p_te"], c2["p_va"]))
    if os.path.exists(DINO_CACHE):
        c3 = np.load(DINO_CACHE)
        sources.append(("DINOv2", c3["p_te"], c3["p_va"]))
    if os.path.exists(MLP_CACHE):
        c4 = np.load(MLP_CACHE)
        sources.append(("MLP", c4["p_te"], c4["p_va"]))

    for name, p_te, p_va in sources:
        print(f"\n=== Source: {name} ===", flush=True)
        res = per_subject_classify(p_va, p_te, yva, yte, subj_va, subj_te, name, classes)
        for variant, m in res.items():
            if isinstance(m, dict) and 'kappa_q' in m:
                print(f"  {variant:>25}: κ_q={m['kappa_q']:.4f} {m['kappa_q_ci']}  t={m.get('t', '')}", flush=True)
        out[name] = res

    # Best fusion: average E[y] across all sources, then per-subject
    print("\n=== Avg E[y] across all sources, per-subject ===", flush=True)
    e_va_all = np.mean([(p_va * classes[None, :]).sum(1) for _, _, p_va in sources], axis=0)
    e_te_all = np.mean([(p_te * classes[None, :]).sum(1) for _, p_te, _ in sources], axis=0)
    # Tune on val (clip-level)
    bt_avg = tune_thresh(e_va_all, yva)
    print(f"  Clip-level threshold: t={bt_avg['t']}  val κ={bt_avg['v']:.4f}", flush=True)
    yhat_te = np.zeros_like(yte)
    unique_te = np.unique(subj_te)
    for s in unique_te:
        mask = subj_te == s
        subj_e = e_te_all[mask].mean()
        c_pred = 0
        if subj_e > bt_avg["t"][0]: c_pred = 1
        if subj_e > bt_avg["t"][1]: c_pred = 2
        if subj_e > bt_avg["t"][2]: c_pred = 3
        yhat_te[mask] = c_pred
    m_avg = metrics(yte, yhat_te)
    out["avg_sources_per_subject"] = {**m_avg, **bt_avg}
    print(f"  Avg sources, per-subject: κ_q={m_avg['kappa_q']:.4f} {m_avg['kappa_q_ci']}", flush=True)

    # Subject-level threshold on avg sources
    unique_va = np.unique(subj_va)
    va_subj_e = []
    va_subj_y = []
    for s in unique_va:
        mask = subj_va == s
        va_subj_e.append(e_va_all[mask].mean())
        va_subj_y.append(int(np.round(yva[mask].mean())))
    va_subj_e = np.array(va_subj_e)
    va_subj_y = np.array(va_subj_y)
    bt_subj_avg = tune_thresh(va_subj_e, va_subj_y)
    yhat_te2 = np.zeros_like(yte)
    for s in unique_te:
        mask = subj_te == s
        subj_e = e_te_all[mask].mean()
        c_pred = 0
        if subj_e > bt_subj_avg["t"][0]: c_pred = 1
        if subj_e > bt_subj_avg["t"][1]: c_pred = 2
        if subj_e > bt_subj_avg["t"][2]: c_pred = 3
        yhat_te2[mask] = c_pred
    m_avg_subj = metrics(yte, yhat_te2)
    out["avg_sources_subj_threshold"] = {**m_avg_subj, **bt_subj_avg}
    print(f"  Avg sources, subj-level threshold: κ_q={m_avg_subj['kappa_q']:.4f} {m_avg_subj['kappa_q_ci']}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
