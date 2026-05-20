"""
Multi-encoder fusion at the bagging level.

Cached: SigLIP-L LR-uniform + LR-RSB-45 bags (from script 80).
New: build CLIP-L/14 LR-uniform bag (K=20).
Fuse all three on val with weights w1+w2+w3=1; threshold tune E[y].

Hypothesis: CLIP-L/14 (768-d, contrastive image-text) captures different feature axes than
SigLIP-L. If the errors are at all decorrelated, the fusion will push κ past 0.228.

Output:
  results/sota/multi_encoder_fusion.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CLIPL = os.path.join(BASE, "features", "daisee_clip_l_14_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
CLIPL_CACHE = os.path.join(BASE, "results", "sota", "_clipl_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "multi_encoder_fusion.json")
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


def bag_lr(Xtr, ytr, Xva, yva, Xte, K=20, seeds=[0, 7, 42, 2025, 1024]):
    all_te, all_va = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        bt, bv = [], []
        for _ in range(K):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            clf = fit_lr(Xtr[idx], ytr[idx], Xva, yva)
            bt.append(clf.predict_proba(Xte)); bv.append(clf.predict_proba(Xva))
        all_te.append(np.stack(bt).mean(0)); all_va.append(np.stack(bv).mean(0))
    return np.stack(all_te).mean(0), np.stack(all_va).mean(0)


def tune_thresholds(e, y, grid_step=0.02):
    grid = np.arange(0.0, 3.01, grid_step)
    best = None
    for t1 in grid:
        for t2 in grid[grid > t1]:
            for t3 in grid[grid > t2]:
                yp = np.zeros_like(e, dtype=int)
                yp[e > t1] = 1
                yp[e > t2] = 2
                yp[e > t3] = 3
                v = cohen_kappa_score(y, yp, weights="quadratic")
                if best is None or v > best["v"]:
                    best = {"t1": float(t1), "t2": float(t2), "t3": float(t3), "v": float(v)}
    return best


def apply_thresholds(e, t1, t2, t3):
    yp = np.zeros_like(e, dtype=int)
    yp[e > t1] = 1; yp[e > t2] = 2; yp[e > t3] = 3
    return yp


def main():
    print("Loading SigLIP-L cached bags...")
    if not os.path.exists(CACHE):
        print("ERROR: SigLIP-L bag cache missing.")
        return
    c = np.load(CACHE)
    p_te_unif = c["p_te_unif"]; p_va_unif = c["p_va_unif"]
    p_te_rsb = c["p_te_rsb"]; p_va_rsb = c["p_va_rsb"]
    print(f"  SigLIP-L bag shapes: te={p_te_unif.shape} va={p_va_unif.shape}")

    print("Loading CLIP-L/14 features...")
    d2 = np.load(CLIPL, allow_pickle=True)
    Xc = d2["feat"].astype(np.float32)
    sp = d2["split"]; eng = d2["engagement"].astype(np.int64)
    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    Xctr, yctr = Xc[tr], eng[tr]; Xcva, ycva = Xc[va], eng[va]; Xcte, ycte = Xc[te], eng[te]

    if os.path.exists(CLIPL_CACHE):
        c2 = np.load(CLIPL_CACHE)
        p_te_clipl = c2["p_te"]; p_va_clipl = c2["p_va"]
        print(f"  CLIP-L bag loaded from cache: te={p_te_clipl.shape}")
    else:
        print("Building CLIP-L/14 LR bag (K=20 × 5 seeds)...")
        t0 = time.time()
        p_te_clipl, p_va_clipl = bag_lr(Xctr, yctr, Xcva, ycva, Xcte, K=20)
        print(f"  CLIP-L bag done ({time.time()-t0:.0f}s)")
        np.savez_compressed(CLIPL_CACHE, p_te=p_te_clipl, p_va=p_va_clipl)

    # Sanity: yva matches between the two .npz files?
    # The SigLIP-L cached probs assume eng[va]; we used the same engagement labels (DAiSEE single-frame manifest)
    # Verify yva equality:
    d1 = np.load(SIGLIP, allow_pickle=True)
    sp1 = d1["split"]; eng1 = d1["engagement"].astype(np.int64)
    yva = eng1[sp1 == "Validation"]; yte = eng1[sp1 == "Test"]
    assert (yva == ycva).all() and (yte == ycte).all(), "Label mismatch between encoder splits!"

    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    out = {}

    print("\n=== Solo CLIP-L bagged ===")
    m = metrics(yte, p_te_clipl.argmax(1))
    out["solo_clipl"] = m
    print(f"  κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}")

    print("\n=== SigLIP-L pair baseline (uniform+RSB) ===")
    # Re-derive baseline
    best_w = None
    for w in np.linspace(0, 1, 21):
        pv = w * p_va_unif + (1 - w) * p_va_rsb
        v = cohen_kappa_score(yva, pv.argmax(1), weights="quadratic")
        if best_w is None or v > best_w["v"]:
            best_w = {"w_unif": float(w), "v": float(v)}
    wu = best_w["w_unif"]
    p_va_siglip = wu * p_va_unif + (1 - wu) * p_va_rsb
    p_te_siglip = wu * p_te_unif + (1 - wu) * p_te_rsb
    e_va_s = (p_va_siglip * classes[None, :]).sum(1)
    e_te_s = (p_te_siglip * classes[None, :]).sum(1)
    bt_s = tune_thresholds(e_va_s, yva)
    m_s = metrics(yte, apply_thresholds(e_te_s, bt_s["t1"], bt_s["t2"], bt_s["t3"]))
    out["siglip_pair_threshold"] = {**m_s, **bt_s, "w_unif": wu}
    print(f"  w_unif={wu:.2f}  t=({bt_s['t1']:.2f},{bt_s['t2']:.2f},{bt_s['t3']:.2f})  "
          f"κ_q={m_s['kappa_q']:.3f} {m_s['kappa_q_ci']}")

    print("\n=== 3-way fusion: SigLIP-uniform + SigLIP-RSB + CLIP-L (val-grid) ===")
    # Stage 1: 3-way argmax on probabilities
    best_w3 = None
    grid = np.linspace(0, 1, 21)
    for w1 in grid:
        for w2 in grid:
            w3 = 1 - w1 - w2
            if w3 < -1e-9 or w3 > 1 + 1e-9: continue
            pv = w1 * p_va_unif + w2 * p_va_rsb + w3 * p_va_clipl
            v = cohen_kappa_score(yva, pv.argmax(1), weights="quadratic")
            if best_w3 is None or v > best_w3["v"]:
                best_w3 = {"w": (float(w1), float(w2), float(w3)), "v": float(v)}
    w1, w2, w3 = best_w3["w"]
    p_va_3 = w1 * p_va_unif + w2 * p_va_rsb + w3 * p_va_clipl
    p_te_3 = w1 * p_te_unif + w2 * p_te_rsb + w3 * p_te_clipl
    m_arg = metrics(yte, p_te_3.argmax(1))
    out["fusion3_argmax"] = {**m_arg, "w": best_w3["w"], "val_kq": best_w3["v"]}
    print(f"  argmax: w=({w1:.2f},{w2:.2f},{w3:.2f})  κ_q={m_arg['kappa_q']:.3f} {m_arg['kappa_q_ci']}  val={best_w3['v']:.3f}")

    # Stage 2: + threshold
    e_va_3 = (p_va_3 * classes[None, :]).sum(1)
    e_te_3 = (p_te_3 * classes[None, :]).sum(1)
    bt_3 = tune_thresholds(e_va_3, yva)
    m_thr = metrics(yte, apply_thresholds(e_te_3, bt_3["t1"], bt_3["t2"], bt_3["t3"]))
    out["fusion3_threshold"] = {**m_thr, **bt_3, "w": best_w3["w"]}
    print(f"  +thresh: t=({bt_3['t1']:.2f},{bt_3['t2']:.2f},{bt_3['t3']:.2f})  κ_q={m_thr['kappa_q']:.3f} {m_thr['kappa_q_ci']}  val={bt_3['v']:.3f}")

    # Stage 3: jointly optimize (w1, w2, w3, t1, t2, t3) on val — randomized
    print("\n=== Joint random search over (w1,w2,w3,t1,t2,t3) ===")
    rng = np.random.default_rng(42)
    best_j = None
    for _ in range(40000):
        a = rng.dirichlet(np.ones(3))
        w1, w2, w3 = a
        pv = w1 * p_va_unif + w2 * p_va_rsb + w3 * p_va_clipl
        e = (pv * classes[None, :]).sum(1)
        # Random thresholds
        ts = sorted(rng.uniform(0, 3, size=3))
        t1, t2, t3 = ts
        yp = apply_thresholds(e, t1, t2, t3)
        v = cohen_kappa_score(yva, yp, weights="quadratic")
        if best_j is None or v > best_j["v"]:
            best_j = {"w": (float(w1), float(w2), float(w3)),
                      "t": (float(t1), float(t2), float(t3)),
                      "v": float(v)}
    w1, w2, w3 = best_j["w"]; t1, t2, t3 = best_j["t"]
    e_te_j = (w1 * p_te_unif + w2 * p_te_rsb + w3 * p_te_clipl) * classes[None, :]
    e_te_j = e_te_j.sum(1)
    m_j = metrics(yte, apply_thresholds(e_te_j, t1, t2, t3))
    out["fusion3_joint"] = {**m_j, "w": best_j["w"], "t": best_j["t"], "val_kq": best_j["v"]}
    print(f"  joint: w=({w1:.2f},{w2:.2f},{w3:.2f})  t=({t1:.2f},{t2:.2f},{t3:.2f})  "
          f"κ_q={m_j['kappa_q']:.3f} {m_j['kappa_q_ci']}  val={best_j['v']:.3f}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
