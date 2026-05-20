"""
Rank-averaging fusion across heterogeneous probes.

Probability averaging is sensitive to calibration. Rank averaging is robust:
- Per-model, convert per-class probability into rank across test set
- Average ranks across models, take argmax of summed ranks
- Decouples model confidence from class assignment

Also test:
  - Geometric mean of probs (vs arithmetic)
  - Power-mean (alpha=2, alpha=0.5) averaging
  - Per-class temperature-scaled fusion
  - Stratified bagging (class-balanced bootstrap) as a new bag-member

Output:
  results/sota/rank_fusion.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from scipy.stats import rankdata

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
OUT = os.path.join(BASE, "results", "sota", "rank_fusion.json")
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


def stratified_bootstrap(ytr, rng):
    """Class-balanced bootstrap: equal samples per class."""
    classes = np.unique(ytr)
    n_per = len(ytr) // len(classes)
    idx = []
    for c in classes:
        c_idx = np.where(ytr == c)[0]
        idx.extend(rng.choice(c_idx, size=n_per, replace=True))
    return np.array(idx)


def bag_lr_uniform(Xtr, ytr, Xva, yva, Xte, K=30, seeds=[0, 7, 42, 2025, 1024]):
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


def bag_lr_stratified(Xtr, ytr, Xva, yva, Xte, K=30, seeds=[0, 7, 42, 2025, 1024]):
    all_te, all_va = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        bt, bv = [], []
        for _ in range(K):
            idx = stratified_bootstrap(ytr, rng)
            clf = fit_lr(Xtr[idx], ytr[idx], Xva, yva)
            bt.append(clf.predict_proba(Xte)); bv.append(clf.predict_proba(Xva))
        all_te.append(np.stack(bt).mean(0)); all_va.append(np.stack(bv).mean(0))
    return np.stack(all_te).mean(0), np.stack(all_va).mean(0)


def bag_rsb(Xtr, ytr, Xva, yva, Xte, K=30, frac=0.45, seeds=[0, 7, 42, 2025, 1024]):
    """Random subspace bagging."""
    D = Xtr.shape[1]
    n_feat = int(D * frac)
    all_te, all_va = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        bt, bv = [], []
        for _ in range(K):
            r_idx = rng.integers(0, len(ytr), size=len(ytr))
            f_idx = rng.choice(D, size=n_feat, replace=False)
            clf = fit_lr(Xtr[r_idx][:, f_idx], ytr[r_idx], Xva[:, f_idx], yva)
            bt.append(clf.predict_proba(Xte[:, f_idx]))
            bv.append(clf.predict_proba(Xva[:, f_idx]))
        all_te.append(np.stack(bt).mean(0)); all_va.append(np.stack(bv).mean(0))
    return np.stack(all_te).mean(0), np.stack(all_va).mean(0)


def to_ranks(p):
    """Rank-transform per-class across samples. Higher prob → higher rank."""
    r = np.zeros_like(p)
    for c in range(p.shape[1]):
        r[:, c] = rankdata(p[:, c])
    return r / p.shape[0]  # normalize to [0,1]


def power_mean(probs_list, alpha):
    """Generalized mean: alpha=1 arith, alpha→0 geom, alpha→-inf min, alpha→inf max."""
    P = np.stack(probs_list)  # (M, N, K)
    if abs(alpha) < 1e-6:
        return np.exp(np.log(P + 1e-12).mean(0))
    return (P ** alpha).mean(0) ** (1.0 / alpha)


def main():
    print("Loading SigLIP-L features...")
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    splits = d["split"]; eng = d["engagement"].astype(np.int64)
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    Xtr, ytr = X[tr], eng[tr]; Xva, yva = X[va], eng[va]; Xte, yte = X[te], eng[te]

    out = {}

    print("\n[1/4] Bag LR (uniform bootstrap)...")
    t0 = time.time()
    p_te_unif, p_va_unif = bag_lr_uniform(Xtr, ytr, Xva, yva, Xte)
    m = metrics(yte, p_te_unif.argmax(1))
    out["bag_uniform"] = m
    print(f"  κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  ({time.time()-t0:.0f}s)")

    print("\n[2/4] Bag LR (stratified, class-balanced bootstrap)...")
    t0 = time.time()
    p_te_strat, p_va_strat = bag_lr_stratified(Xtr, ytr, Xva, yva, Xte)
    m = metrics(yte, p_te_strat.argmax(1))
    out["bag_stratified"] = m
    print(f"  κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  ({time.time()-t0:.0f}s)")

    print("\n[3/4] Bag LR (random subspace 0.45)...")
    t0 = time.time()
    p_te_rsb, p_va_rsb = bag_rsb(Xtr, ytr, Xva, yva, Xte, frac=0.45)
    m = metrics(yte, p_te_rsb.argmax(1))
    out["bag_rsb_45"] = m
    print(f"  κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  ({time.time()-t0:.0f}s)")

    print("\n[4/4] Fusion experiments...")
    # Build a roster of probs
    members = {
        "uniform": (p_te_unif, p_va_unif),
        "stratified": (p_te_strat, p_va_strat),
        "rsb45": (p_te_rsb, p_va_rsb),
    }

    # (a) Arithmetic mean — 3-way
    pa_te = np.mean([m[0] for m in members.values()], axis=0)
    pa_va = np.mean([m[1] for m in members.values()], axis=0)
    out["arith_3way"] = metrics(yte, pa_te.argmax(1))
    out["arith_3way"]["val_kappa"] = float(cohen_kappa_score(yva, pa_va.argmax(1), weights="quadratic"))
    print(f"  Arith 3-way: κ_q={out['arith_3way']['kappa_q']:.3f} {out['arith_3way']['kappa_q_ci']}")

    # (b) Geometric mean (power_mean alpha→0)
    pg_te = power_mean([m[0] for m in members.values()], alpha=1e-6)
    pg_te /= pg_te.sum(1, keepdims=True)
    out["geom_3way"] = metrics(yte, pg_te.argmax(1))
    print(f"  Geom 3-way:  κ_q={out['geom_3way']['kappa_q']:.3f} {out['geom_3way']['kappa_q_ci']}")

    # (c) Rank averaging
    pr_te = sum(to_ranks(m[0]) for m in members.values()) / len(members)
    out["rank_3way"] = metrics(yte, pr_te.argmax(1))
    print(f"  Rank 3-way:  κ_q={out['rank_3way']['kappa_q']:.3f} {out['rank_3way']['kappa_q_ci']}")

    # (d) Power-mean sweep
    best_pm = None
    for alpha in [-2, -1, -0.5, 0, 0.5, 1, 2, 4]:
        if alpha == 0:
            pm_te = power_mean([m[0] for m in members.values()], alpha=1e-6)
            pm_va = power_mean([m[1] for m in members.values()], alpha=1e-6)
        else:
            pm_te = power_mean([m[0] for m in members.values()], alpha=alpha)
            pm_va = power_mean([m[1] for m in members.values()], alpha=alpha)
        v = cohen_kappa_score(yva, pm_va.argmax(1), weights="quadratic")
        t = cohen_kappa_score(yte, pm_te.argmax(1), weights="quadratic")
        print(f"    alpha={alpha:>4}: val={v:.3f}  test={t:.3f}")
        if best_pm is None or v > best_pm["v"]:
            best_pm = {"a": alpha, "v": float(v), "t": float(t), "pm_te": pm_te}
    out["power_mean_best"] = metrics(yte, best_pm["pm_te"].argmax(1))
    out["power_mean_best"]["alpha"] = best_pm["a"]
    out["power_mean_best"]["val_kappa"] = best_pm["v"]
    print(f"  Best power-mean: alpha={best_pm['a']}  κ_q={out['power_mean_best']['kappa_q']:.3f}")

    # (e) Val-tuned weighted arith
    best_w = None
    grid = np.linspace(0, 1, 11)
    for w1 in grid:
        for w2 in grid:
            w3 = 1 - w1 - w2
            if w3 < 0 or w3 > 1: continue
            pv = w1*p_va_unif + w2*p_va_strat + w3*p_va_rsb
            v = cohen_kappa_score(yva, pv.argmax(1), weights="quadratic")
            if best_w is None or v > best_w["v"]:
                best_w = {"w": (float(w1), float(w2), float(w3)), "v": float(v)}
    w1, w2, w3 = best_w["w"]
    pw_te = w1*p_te_unif + w2*p_te_strat + w3*p_te_rsb
    out["weighted_arith"] = metrics(yte, pw_te.argmax(1))
    out["weighted_arith"]["w"] = best_w["w"]
    out["weighted_arith"]["val_kappa"] = best_w["v"]
    print(f"  Weighted arith: w={best_w['w']}  κ_q={out['weighted_arith']['kappa_q']:.3f} {out['weighted_arith']['kappa_q_ci']}  val={best_w['v']:.3f}")

    # (f) Val-tuned weighted rank
    pu_te_r, pu_va_r = to_ranks(p_te_unif), to_ranks(p_va_unif)
    ps_te_r, ps_va_r = to_ranks(p_te_strat), to_ranks(p_va_strat)
    pr_te_r, pr_va_r = to_ranks(p_te_rsb), to_ranks(p_va_rsb)
    best_wr = None
    for w1 in grid:
        for w2 in grid:
            w3 = 1 - w1 - w2
            if w3 < 0 or w3 > 1: continue
            rv = w1*pu_va_r + w2*ps_va_r + w3*pr_va_r
            v = cohen_kappa_score(yva, rv.argmax(1), weights="quadratic")
            if best_wr is None or v > best_wr["v"]:
                best_wr = {"w": (float(w1), float(w2), float(w3)), "v": float(v)}
    w1, w2, w3 = best_wr["w"]
    rw_te = w1*pu_te_r + w2*ps_te_r + w3*pr_te_r
    out["weighted_rank"] = metrics(yte, rw_te.argmax(1))
    out["weighted_rank"]["w"] = best_wr["w"]
    out["weighted_rank"]["val_kappa"] = best_wr["v"]
    print(f"  Weighted rank:  w={best_wr['w']}  κ_q={out['weighted_rank']['kappa_q']:.3f} {out['weighted_rank']['kappa_q_ci']}  val={best_wr['v']:.3f}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
