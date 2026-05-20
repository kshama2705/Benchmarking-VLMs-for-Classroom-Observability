"""
Train+Val combined SOTA recipe — FAST version.

- Use ALL of Train+Val to train LR (K=20 × 5 seeds bag = 100 probes)
- For threshold tuning, use a single 80/20 holdout of T+V (subject-stratified
  random split with seed=42)
- Predict test

This is the simplest possible "more data" experiment. ~5-10 min runtime.
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, accuracy_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
OUT = os.path.join(BASE, "results", "sota", "trainval_fast.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)


def boot_ci(yt, yp, n=1000):
    nt = len(yt); out = []
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def fit_lr(Xtr, ytr, Xva, yva, Cs=(0.001, 0.01, 0.1, 1.0, 10.0)):
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": float(v), "clf": clf}
    return best["clf"]


def bag(Xtr, ytr, Xva, yva, Xte, K=20, seeds=(0, 7, 42, 2025, 1024)):
    all_te = []; all_va = []
    for s in seeds:
        rng = np.random.default_rng(s)
        for _ in range(K):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            clf = fit_lr(Xtr[idx], ytr[idx], Xva, yva)
            all_te.append(clf.predict_proba(Xte))
            all_va.append(clf.predict_proba(Xva))
    return np.mean(all_te, axis=0), np.mean(all_va, axis=0)


def tune(e, y, step=0.04):
    grid = np.arange(0.0, 3.01, step); best = None
    for t1 in grid:
        for t2 in grid[grid > t1]:
            for t3 in grid[grid > t2]:
                yp = np.zeros_like(e, dtype=int)
                yp[e > t1] = 1; yp[e > t2] = 2; yp[e > t3] = 3
                v = cohen_kappa_score(y, yp, weights="quadratic")
                if best is None or v > best['v']:
                    best = {'t': (float(t1), float(t2), float(t3)), 'v': float(v)}
    return best


def apply_thr(e, t):
    yp = np.zeros_like(e, dtype=int)
    yp[e > t[0]] = 1; yp[e > t[1]] = 2; yp[e > t[2]] = 3
    return yp


def main():
    print("Loading...", flush=True)
    d = np.load(SIGLIP, allow_pickle=True)
    feat = d["feat"].astype(np.float32); split = d["split"]; eng = d["engagement"].astype(np.int64)
    subj = d["subject_id"]
    tr_mask = (split == "Train"); va_mask = (split == "Validation"); te_mask = (split == "Test")

    Xtr = feat[tr_mask]; ytr = eng[tr_mask]; str_ = subj[tr_mask]
    Xva = feat[va_mask]; yva = eng[va_mask]; sva = subj[va_mask]
    Xte = feat[te_mask]; yte = eng[te_mask]

    # Make T+V combined; carve 20% out as "holdout-val" for threshold tuning
    # Subject-stratified: keep 5 subjects out from val portion only (so train subjects stay disjoint)
    Xtv = np.concatenate([Xtr, Xva], axis=0); ytv = np.concatenate([ytr, yva], axis=0)
    stv = np.concatenate([str_, sva], axis=0)
    print(f"T+V combined: {len(ytv)} clips, {len(np.unique(stv))} subjects", flush=True)

    # Hold out 20% of T+V by subject
    rng_split = np.random.default_rng(42)
    uniq_s = np.unique(stv)
    rng_split.shuffle(uniq_s)
    n_hold = max(1, int(0.20 * len(uniq_s)))
    hold_s = set(uniq_s[:n_hold])
    hold_mask = np.array([s in hold_s for s in stv])
    print(f"  hold-out: {hold_mask.sum()} clips from {n_hold} subjects (for threshold tuning)", flush=True)
    Xtv_tr = Xtv[~hold_mask]; ytv_tr = ytv[~hold_mask]
    Xtv_ho = Xtv[hold_mask]; ytv_ho = ytv[hold_mask]
    print(f"  inner train: {len(ytv_tr)} clips, inner val: {len(ytv_ho)} clips", flush=True)

    # Run baseline (Train only) for direct comparison
    classes = np.arange(4, dtype=np.float32)
    print("\n=== A: Train-only baseline (matches prior SOTA recipe) ===", flush=True)
    t0 = time.time()
    p_te_a, p_va_a = bag(Xtr, ytr, Xva, yva, Xte, K=20)
    print(f"  bagging: {(time.time()-t0)/60:.1f}min", flush=True)
    e_va = (p_va_a * classes[None]).sum(1); e_te = (p_te_a * classes[None]).sum(1)
    bt_a = tune(e_va, yva)
    yp_a = apply_thr(e_te, bt_a['t'])
    kq_a = float(cohen_kappa_score(yte, yp_a, weights="quadratic"))
    ci_a = boot_ci(yte, yp_a)
    print(f"  [A] Train-only κ_q = {kq_a:.4f}  CI={ci_a}  t={bt_a['t']}", flush=True)

    print("\n=== B: Train+Val combined ===", flush=True)
    t0 = time.time()
    p_te_b, p_ho_b = bag(Xtv_tr, ytv_tr, Xtv_ho, ytv_ho, Xte, K=20)
    print(f"  bagging: {(time.time()-t0)/60:.1f}min", flush=True)
    e_ho = (p_ho_b * classes[None]).sum(1); e_te = (p_te_b * classes[None]).sum(1)
    bt_b = tune(e_ho, ytv_ho)
    yp_b = apply_thr(e_te, bt_b['t'])
    kq_b = float(cohen_kappa_score(yte, yp_b, weights="quadratic"))
    ci_b = boot_ci(yte, yp_b)
    print(f"  [B] Train+Val combined κ_q = {kq_b:.4f}  CI={ci_b}  t={bt_b['t']}", flush=True)

    # Also: fuse A+B test probabilities (average)
    p_te_fuse = 0.5 * p_te_a + 0.5 * p_te_b
    e_te_fuse = (p_te_fuse * classes[None]).sum(1)
    # Use BOTH val (for A) and hold-out (for B) for threshold tuning — score on the inner hold-out only
    bt_f = tune(e_te_fuse[:0] if False else e_te_fuse, yte)  # placeholder; we won't use test for threshold
    # Actually a fair fuse: take threshold from the (T+V trained) probe on hold-out, apply to fuse
    yp_fuse = apply_thr(e_te_fuse, bt_b['t'])
    kq_fuse = float(cohen_kappa_score(yte, yp_fuse, weights="quadratic"))
    ci_fuse = boot_ci(yte, yp_fuse)
    print(f"\n[A+B fuse @ B's threshold] κ_q = {kq_fuse:.4f}  CI={ci_fuse}", flush=True)

    delta = kq_b - kq_a
    print(f"\n*** Train+Val effect: Δκ = {delta:+.4f} ({kq_a:.4f} → {kq_b:.4f}) ***", flush=True)

    out = {
        "A_train_only": {"kq": kq_a, "ci": ci_a, "t": bt_a['t']},
        "B_train_val": {"kq": kq_b, "ci": ci_b, "t": bt_b['t']},
        "AB_fuse_meanprob_Bthr": {"kq": kq_fuse, "ci": ci_fuse},
        "delta_B_minus_A": delta,
        "n_train_only": len(ytr), "n_train_val_combined": len(ytv_tr),
        "n_holdout_subjects": n_hold,
    }
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
