"""
MOONSHOT 30: Hybrid clip-prediction + subject-baseline.

For each test clip, choose between:
  - Clip-level prediction (high confidence)
  - Subject baseline (low confidence fallback)

Confidence weighting based on max_prob.

Output:
  results/sota/moonshot_hybrid_clip_subj.json
"""
import os, json
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_hybrid_clip_subj.json")
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
    conf_va = p_va.max(1)
    conf_te = p_te.max(1)

    # Baseline thresholds (clip-level)
    bt = tune_thresh(e_va, yva)
    t = bt["t"]
    print(f"Thresholds: {t}", flush=True)

    out = {}

    # Method A: clip-level (baseline)
    yhat_clip = apply_t(e_te, t)
    m_clip = metrics(yte, yhat_clip)
    out["clip_only"] = m_clip
    print(f"\nClip-only: κ={m_clip['kappa_q']:.4f} {m_clip['kappa_q_ci']}", flush=True)

    # Method B: subject baseline (SOTA = 0.2421)
    yhat_subj = np.zeros_like(yte)
    for s in np.unique(subj_te):
        mask = subj_te == s
        sm = e_te[mask].mean()
        cp = 0
        if sm > t[0]: cp = 1
        if sm > t[1]: cp = 2
        if sm > t[2]: cp = 3
        yhat_subj[mask] = cp
    m_subj = metrics(yte, yhat_subj)
    out["subject_only"] = m_subj
    print(f"Subject-only: κ={m_subj['kappa_q']:.4f} {m_subj['kappa_q_ci']}", flush=True)

    # Method C: per-clip prediction USE SUBJECT MEAN as the E[y] (override per-clip e_te with subj mean)
    e_te_subj = np.zeros_like(e_te)
    for s in np.unique(subj_te):
        mask = subj_te == s
        e_te_subj[mask] = e_te[mask].mean()

    # Method D: Hybrid with confidence threshold τ
    print("\n[D] Hybrid: clip prediction if max_prob > τ else subject baseline", flush=True)
    for tau in [0.3, 0.4, 0.5, 0.6, 0.7]:
        yhat_hybrid = np.where(conf_te > tau, yhat_clip, yhat_subj)
        m = metrics(yte, yhat_hybrid)
        out[f"hybrid_tau{tau}"] = m
        n_clip = int((conf_te > tau).sum())
        print(f"  τ={tau}: clip-used={n_clip}/1784  κ={m['kappa_q']:.4f} {m['kappa_q_ci']}", flush=True)

    # Method E: weighted average of clip-level e_te and subject-mean e_te_subj
    print("\n[E] Weighted average of e_te (clip) and e_te_subj (subject mean)", flush=True)
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        e_te_blend = alpha * e_te + (1 - alpha) * e_te_subj
        yhat_blend = apply_t(e_te_blend, t)
        m = metrics(yte, yhat_blend)
        out[f"blend_alpha{alpha}"] = m
        print(f"  α={alpha}: κ={m['kappa_q']:.4f} {m['kappa_q_ci']}", flush=True)

    # Method F: tune α on val
    print("\n[F] Tune blend α on val", flush=True)
    e_va_subj = np.zeros_like(e_va)
    for s in np.unique(subj_va):
        mask = subj_va == s
        e_va_subj[mask] = e_va[mask].mean()

    best_alpha = None
    for alpha in np.linspace(0, 1, 21):
        e_va_blend = alpha * e_va + (1 - alpha) * e_va_subj
        bt_a = tune_thresh(e_va_blend, yva)
        if best_alpha is None or bt_a["v"] > best_alpha["v"]:
            best_alpha = {"alpha": float(alpha), **bt_a}
    a = best_alpha["alpha"]
    e_te_blend = a * e_te + (1 - a) * e_te_subj
    yhat = apply_t(e_te_blend, best_alpha["t"])
    m = metrics(yte, yhat)
    out["blend_tuned"] = {**m, "alpha": a, **best_alpha}
    print(f"  Best α={a:.2f}  t={best_alpha['t']}  κ={m['kappa_q']:.4f} {m['kappa_q_ci']}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
