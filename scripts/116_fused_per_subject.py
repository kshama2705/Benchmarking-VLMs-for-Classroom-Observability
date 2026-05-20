"""
MOONSHOT 27: Per-subject baseline with model fusion.

Fuse E[y] across models (LR-unif, MLP, LR-RSB) with val-tuned weights,
then apply per-subject baseline.

Output:
  results/sota/moonshot_fused_per_subject.json
"""
import os, json, time
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
MLP_CACHE = os.path.join(BASE, "results", "sota", "_mlp_bag_cache.npz")
CLIPL_CACHE = os.path.join(BASE, "results", "sota", "_clipl_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_fused_per_subject.json")
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


def per_subject_classify(e_te, subj_te, yte, t):
    yhat = np.zeros_like(yte)
    for s in np.unique(subj_te):
        mask = subj_te == s
        sm = e_te[mask].mean()
        c_pred = 0
        if sm > t[0]: c_pred = 1
        if sm > t[1]: c_pred = 2
        if sm > t[2]: c_pred = 3
        yhat[mask] = c_pred
    return metrics(yte, yhat)


def main():
    print("Loading...", flush=True)
    d = np.load(SIGLIP, allow_pickle=True)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    subj = d["subject_id"]
    va = sp == "Validation"; te = sp == "Test"
    yva = eng[va]; yte = eng[te]
    subj_te = subj[te]; subj_va = subj[va]

    c = np.load(CACHE)
    cm = np.load(MLP_CACHE)
    cc = np.load(CLIPL_CACHE)
    classes = np.array([0, 1, 2, 3], dtype=np.float32)

    sources = {
        "LR-unif": (c["p_te_unif"], c["p_va_unif"]),
        "LR-RSB45": (c["p_te_rsb"], c["p_va_rsb"]),
        "MLP": (cm["p_te"], cm["p_va"]),
        "CLIP-L": (cc["p_te"], cc["p_va"]),
    }

    eys = {n: ((pt * classes[None, :]).sum(1), (pv * classes[None, :]).sum(1)) for n, (pt, pv) in sources.items()}

    out = {}

    # Solo per-subject for each source
    print("\n=== Solo per-subject (use SOURCE's own clip-thresh) ===", flush=True)
    for name in sources:
        e_te, e_va = eys[name]
        bt = tune_thresh(e_va, yva)
        m_subj = per_subject_classify(e_te, subj_te, yte, bt["t"])
        out[f"solo_{name}"] = {**m_subj, **bt}
        print(f"  {name}: per-subj κ={m_subj['kappa_q']:.4f} {m_subj['kappa_q_ci']}", flush=True)

    # Fusion: val-tuned weighted average E[y]
    print("\n=== Fused E[y] (val-tuned weights via grid) → per-subject ===", flush=True)
    e_te_lr = eys["LR-unif"][0]; e_va_lr = eys["LR-unif"][1]
    e_te_mlp = eys["MLP"][0]; e_va_mlp = eys["MLP"][1]
    e_te_rsb = eys["LR-RSB45"][0]; e_va_rsb = eys["LR-RSB45"][1]

    # Try 2-way (LR-unif + MLP)
    best_w = None
    for w in np.linspace(0, 1, 51):
        e_va_f = w * e_va_lr + (1 - w) * e_va_mlp
        bt = tune_thresh(e_va_f, yva)
        if best_w is None or bt["v"] > best_w["v"]:
            best_w = {"w": w, **bt}
    w_lr_mlp = best_w["w"]
    e_te_f = w_lr_mlp * e_te_lr + (1 - w_lr_mlp) * e_te_mlp
    m_2way = per_subject_classify(e_te_f, subj_te, yte, best_w["t"])
    out["fusion_lr_mlp_per_subject"] = {**m_2way, "w_lr": float(w_lr_mlp), **best_w}
    print(f"  LR + MLP: w_lr={w_lr_mlp:.2f}  κ={m_2way['kappa_q']:.4f} {m_2way['kappa_q_ci']}", flush=True)

    # 3-way (LR-unif + LR-RSB + MLP) via Dirichlet random search
    print("\n  3-way LR+RSB+MLP Dirichlet search...", flush=True)
    rng = np.random.default_rng(42)
    best_3 = None
    for _ in range(20000):
        a = rng.dirichlet(np.ones(3))
        e_va_f = a[0]*e_va_lr + a[1]*e_va_rsb + a[2]*e_va_mlp
        bt = tune_thresh(e_va_f, yva, step=0.08)  # coarser for speed
        if best_3 is None or bt["v"] > best_3["v"]:
            best_3 = {"w": a.tolist(), **bt}
    w = best_3["w"]
    e_te_f = w[0]*e_te_lr + w[1]*e_te_rsb + w[2]*e_te_mlp
    # Refine threshold
    e_va_f = w[0]*e_va_lr + w[1]*e_va_rsb + w[2]*e_va_mlp
    bt_refined = tune_thresh(e_va_f, yva, step=0.04)
    m_3way = per_subject_classify(e_te_f, subj_te, yte, bt_refined["t"])
    out["fusion_3way_per_subject"] = {**m_3way, "w": w, **bt_refined}
    print(f"  3-way: w={['{:.2f}'.format(x) for x in w]}  κ={m_3way['kappa_q']:.4f} {m_3way['kappa_q_ci']}", flush=True)

    # 4-way Dirichlet
    print("\n  4-way LR+RSB+MLP+CLIP-L Dirichlet search...", flush=True)
    e_te_clip = eys["CLIP-L"][0]; e_va_clip = eys["CLIP-L"][1]
    best_4 = None
    for _ in range(30000):
        a = rng.dirichlet(np.ones(4))
        e_va_f = a[0]*e_va_lr + a[1]*e_va_rsb + a[2]*e_va_mlp + a[3]*e_va_clip
        bt = tune_thresh(e_va_f, yva, step=0.08)
        if best_4 is None or bt["v"] > best_4["v"]:
            best_4 = {"w": a.tolist(), **bt}
    w = best_4["w"]
    e_te_f = w[0]*e_te_lr + w[1]*e_te_rsb + w[2]*e_te_mlp + w[3]*e_te_clip
    e_va_f = w[0]*e_va_lr + w[1]*e_va_rsb + w[2]*e_va_mlp + w[3]*e_va_clip
    bt_refined = tune_thresh(e_va_f, yva, step=0.04)
    m_4way = per_subject_classify(e_te_f, subj_te, yte, bt_refined["t"])
    out["fusion_4way_per_subject"] = {**m_4way, "w": w, **bt_refined}
    print(f"  4-way: w={['{:.2f}'.format(x) for x in w]}  κ={m_4way['kappa_q']:.4f} {m_4way['kappa_q_ci']}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
