"""
MOONSHOT 23: Confidence-weighted subject baseline.

Per-subject baseline (moonshot 20) takes unweighted mean of E[y] across a
subject's clips. But some clips have low-confidence predictions (max prob near
0.25). Weighting by max prob may give a cleaner subject baseline estimate.

Variants:
  A) Unweighted (baseline, = moonshot 20)
  B) Weighted by max_prob (clip confidence)
  C) Weighted by max_prob^2
  D) Median instead of mean (robust to outliers)
  E) Trimmed mean (drop top/bottom 20%)
  F) Per-subject Ridge-shrunk to global mean (prevent extreme estimates)

Output:
  results/sota/moonshot_conf_weighted_subj.json
"""
import os, json, time
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_conf_weighted_subj.json")
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


def apply_classify_per_subject(e_te, subj_te, yte, t):
    """Apply subject-mean classification."""
    yhat = np.zeros_like(yte)
    for s in np.unique(subj_te):
        mask = subj_te == s
        subj_e = e_te[mask].mean() if isinstance(e_te[mask], np.ndarray) else e_te[mask]
        c_pred = 0
        if subj_e > t[0]: c_pred = 1
        if subj_e > t[1]: c_pred = 2
        if subj_e > t[2]: c_pred = 3
        yhat[mask] = c_pred
    return yhat


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
    conf_te = p_te.max(axis=1)
    conf_va = p_va.max(axis=1)

    bt = tune_thresh(e_va, yva)
    t = bt["t"]
    print(f"\nGlobal thresholds: {t}  val κ={bt['v']:.4f}", flush=True)

    out = {"baseline_thresholds": t, "baseline_val_kq": bt["v"]}

    # ============ A: Unweighted mean ============
    print("\n[A] Unweighted mean...", flush=True)
    yhat_a = np.zeros_like(yte)
    for s in np.unique(subj_te):
        mask = subj_te == s
        sm = e_te[mask].mean()
        c_pred = 0
        if sm > t[0]: c_pred = 1
        if sm > t[1]: c_pred = 2
        if sm > t[2]: c_pred = 3
        yhat_a[mask] = c_pred
    m_a = metrics(yte, yhat_a)
    out["unweighted_mean"] = m_a
    print(f"  κ_q={m_a['kappa_q']:.4f} {m_a['kappa_q_ci']}", flush=True)

    # ============ B: Weighted by max_prob ============
    print("\n[B] Weighted by confidence (max_prob)...", flush=True)
    yhat_b = np.zeros_like(yte)
    for s in np.unique(subj_te):
        mask = subj_te == s
        e_s = e_te[mask]
        w_s = conf_te[mask]
        sm = np.sum(e_s * w_s) / np.sum(w_s)
        c_pred = 0
        if sm > t[0]: c_pred = 1
        if sm > t[1]: c_pred = 2
        if sm > t[2]: c_pred = 3
        yhat_b[mask] = c_pred
    m_b = metrics(yte, yhat_b)
    out["conf_weighted_mean"] = m_b
    print(f"  κ_q={m_b['kappa_q']:.4f} {m_b['kappa_q_ci']}", flush=True)

    # ============ C: Weighted by max_prob^2 ============
    yhat_c = np.zeros_like(yte)
    for s in np.unique(subj_te):
        mask = subj_te == s
        e_s = e_te[mask]
        w_s = conf_te[mask] ** 2
        sm = np.sum(e_s * w_s) / np.sum(w_s)
        c_pred = 0
        if sm > t[0]: c_pred = 1
        if sm > t[1]: c_pred = 2
        if sm > t[2]: c_pred = 3
        yhat_c[mask] = c_pred
    m_c = metrics(yte, yhat_c)
    out["conf_squared_weighted_mean"] = m_c
    print(f"\n[C] κ_q={m_c['kappa_q']:.4f} {m_c['kappa_q_ci']}", flush=True)

    # ============ D: Median ============
    yhat_d = np.zeros_like(yte)
    for s in np.unique(subj_te):
        mask = subj_te == s
        sm = np.median(e_te[mask])
        c_pred = 0
        if sm > t[0]: c_pred = 1
        if sm > t[1]: c_pred = 2
        if sm > t[2]: c_pred = 3
        yhat_d[mask] = c_pred
    m_d = metrics(yte, yhat_d)
    out["median_subject"] = m_d
    print(f"\n[D] Median: κ_q={m_d['kappa_q']:.4f} {m_d['kappa_q_ci']}", flush=True)

    # ============ E: Trimmed mean ============
    yhat_e = np.zeros_like(yte)
    for s in np.unique(subj_te):
        mask = subj_te == s
        e_s = sorted(e_te[mask])
        n = len(e_s); trim = int(0.2 * n)
        if trim > 0 and n > 2 * trim:
            e_s = e_s[trim:-trim]
        sm = np.mean(e_s)
        c_pred = 0
        if sm > t[0]: c_pred = 1
        if sm > t[1]: c_pred = 2
        if sm > t[2]: c_pred = 3
        yhat_e[mask] = c_pred
    m_e = metrics(yte, yhat_e)
    out["trimmed_mean"] = m_e
    print(f"\n[E] Trimmed: κ_q={m_e['kappa_q']:.4f} {m_e['kappa_q_ci']}", flush=True)

    # ============ F: Ridge-shrunk subject means ============
    # Subject mean shrunk toward global mean: m_s' = (n_s * m_s + alpha * m_global) / (n_s + alpha)
    global_mean = e_te.mean()
    for alpha in [1.0, 3.0, 5.0, 10.0]:
        yhat_f = np.zeros_like(yte)
        for s in np.unique(subj_te):
            mask = subj_te == s
            n_s = mask.sum()
            m_s = e_te[mask].mean()
            sm = (n_s * m_s + alpha * global_mean) / (n_s + alpha)
            c_pred = 0
            if sm > t[0]: c_pred = 1
            if sm > t[1]: c_pred = 2
            if sm > t[2]: c_pred = 3
            yhat_f[mask] = c_pred
        m_f = metrics(yte, yhat_f)
        out[f"shrunk_subject_alpha{alpha}"] = m_f
        print(f"\n[F] α={alpha}: κ_q={m_f['kappa_q']:.4f} {m_f['kappa_q_ci']}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
