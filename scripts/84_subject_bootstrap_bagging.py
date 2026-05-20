"""
Subject-disjoint bagging (NEW ANGLE).

Current bagging samples clips with replacement (clip-level bootstrap). Given the
identity-engagement entanglement finding (subject-ID 99.5% recoverable from same
features), clip-level bootstraps within a subject create redundant samples — the
'effective n' is the number of subjects, not the number of clips.

Subject bootstrap: sample 70 train subjects with replacement (matching #train subjects),
then take ALL clips from sampled subjects. Bag size matches: ~5358 (with replacement
collapses to ~63% unique subjects, similar to clip bootstrap).

Hypothesis: subject-disjoint bagging better matches the structural variance of the
problem; ensemble diversity comes from "which subjects model trained on", which is
the actual source of test-time variance. Should give tighter CIs and possibly better κ.

Combine with: SigLIP-L cached bags, +CLIP-L/14 subject-bag, fold-vote.

Output:
  results/sota/subject_bagging.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
SUBJ_CACHE = os.path.join(BASE, "results", "sota", "_subj_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "subject_bagging.json")
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


def bag_subject_lr(Xtr, ytr, subj_tr, Xva, yva, Xte, K=20, seeds=[0, 7, 42, 2025, 1024]):
    """Bag by sampling subjects with replacement; take all clips from sampled subjects."""
    unique_subjects = np.unique(subj_tr)
    n_subj = len(unique_subjects)
    subj_to_idx = {s: np.where(subj_tr == s)[0] for s in unique_subjects}
    all_te, all_va = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        bt, bv = [], []
        for k in range(K):
            sampled = rng.choice(unique_subjects, size=n_subj, replace=True)
            rows = np.concatenate([subj_to_idx[ss] for ss in sampled])
            clf = fit_lr(Xtr[rows], ytr[rows], Xva, yva)
            bt.append(clf.predict_proba(Xte)); bv.append(clf.predict_proba(Xva))
            if (k + 1) % 10 == 0:
                print(f"    seed={s} k={k+1}/{K}  n_unique_subj_in_bag={len(np.unique(sampled))}", flush=True)
        all_te.append(np.stack(bt).mean(0)); all_va.append(np.stack(bv).mean(0))
    return np.stack(all_te).mean(0), np.stack(all_va).mean(0)


def tune_thresholds(e, y, grid_step=0.02):
    grid = np.arange(0.0, 3.01, grid_step)
    best = None
    for t1 in grid:
        for t2 in grid[grid > t1]:
            for t3 in grid[grid > t2]:
                yp = np.zeros_like(e, dtype=int)
                yp[e > t1] = 1; yp[e > t2] = 2; yp[e > t3] = 3
                v = cohen_kappa_score(y, yp, weights="quadratic")
                if best is None or v > best["v"]:
                    best = {"t1": float(t1), "t2": float(t2), "t3": float(t3), "v": float(v)}
    return best


def apply_thresholds(e, t1, t2, t3):
    yp = np.zeros_like(e, dtype=int)
    yp[e > t1] = 1; yp[e > t2] = 2; yp[e > t3] = 3
    return yp


def main():
    print("Loading SigLIP-L features...")
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    subj = d["subject_id"]
    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    Xtr, ytr = X[tr], eng[tr]; Xva, yva = X[va], eng[va]; Xte, yte = X[te], eng[te]
    subj_tr = subj[tr]
    print(f"  Train: {len(ytr)} clips, {len(np.unique(subj_tr))} subjects")

    if os.path.exists(SUBJ_CACHE):
        sc = np.load(SUBJ_CACHE)
        p_te_sub = sc["p_te"]; p_va_sub = sc["p_va"]
        print(f"  Subject-bag loaded from cache: {p_te_sub.shape}")
    else:
        print("\nBuilding subject-bootstrap LR bag (K=20 × 5 seeds)...")
        t0 = time.time()
        p_te_sub, p_va_sub = bag_subject_lr(Xtr, ytr, subj_tr, Xva, yva, Xte, K=20)
        print(f"  Done ({time.time()-t0:.0f}s)")
        np.savez_compressed(SUBJ_CACHE, p_te=p_te_sub, p_va=p_va_sub)

    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    out = {}

    print("\n=== Solo subject-bag ===")
    m = metrics(yte, p_te_sub.argmax(1))
    out["solo_subj_bag"] = m
    print(f"  κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}")

    print("\n=== Subject-bag + threshold ===")
    e_va = (p_va_sub * classes[None, :]).sum(1)
    e_te = (p_te_sub * classes[None, :]).sum(1)
    bt = tune_thresholds(e_va, yva)
    m2 = metrics(yte, apply_thresholds(e_te, bt["t1"], bt["t2"], bt["t3"]))
    out["subj_bag_threshold"] = {**m2, **bt}
    print(f"  t=({bt['t1']:.2f},{bt['t2']:.2f},{bt['t3']:.2f})  κ_q={m2['kappa_q']:.3f} {m2['kappa_q_ci']}")

    print("\n=== Fusion: clip-bag + subj-bag (SigLIP-L only) ===")
    c = np.load(CACHE)
    p_te_unif = c["p_te_unif"]; p_va_unif = c["p_va_unif"]
    p_te_rsb = c["p_te_rsb"]; p_va_rsb = c["p_va_rsb"]
    # 3-way val grid: w1 (unif), w2 (rsb), w3 (subj)
    best_w = None
    grid = np.linspace(0, 1, 21)
    for w1 in grid:
        for w2 in grid:
            w3 = 1 - w1 - w2
            if w3 < -1e-9 or w3 > 1+1e-9: continue
            pv = w1*p_va_unif + w2*p_va_rsb + w3*p_va_sub
            v = cohen_kappa_score(yva, pv.argmax(1), weights="quadratic")
            if best_w is None or v > best_w["v"]:
                best_w = {"w": (float(w1), float(w2), float(w3)), "v": float(v)}
    w1, w2, w3 = best_w["w"]
    p_va_3 = w1*p_va_unif + w2*p_va_rsb + w3*p_va_sub
    p_te_3 = w1*p_te_unif + w2*p_te_rsb + w3*p_te_sub
    m_arg = metrics(yte, p_te_3.argmax(1))
    out["fusion3_argmax_clip_rsb_subj"] = {**m_arg, "w": best_w["w"], "val_kq": best_w["v"]}
    print(f"  argmax: w=({w1:.2f},{w2:.2f},{w3:.2f})  κ_q={m_arg['kappa_q']:.3f}  val={best_w['v']:.3f}")
    # +threshold
    e_va_3 = (p_va_3 * classes[None, :]).sum(1)
    e_te_3 = (p_te_3 * classes[None, :]).sum(1)
    bt_3 = tune_thresholds(e_va_3, yva)
    m_thr = metrics(yte, apply_thresholds(e_te_3, bt_3["t1"], bt_3["t2"], bt_3["t3"]))
    out["fusion3_threshold_clip_rsb_subj"] = {**m_thr, **bt_3, "w": best_w["w"]}
    print(f"  +thresh: t=({bt_3['t1']:.2f},{bt_3['t2']:.2f},{bt_3['t3']:.2f})  κ_q={m_thr['kappa_q']:.3f} {m_thr['kappa_q_ci']}  val={bt_3['v']:.3f}")

    # Also try 5-fold majority vote on the 3-way
    from sklearn.model_selection import KFold
    print("\n=== 5-fold majority vote on (unif, rsb, subj) ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_preds = []
    for fold_idx, (cal_idx, _) in enumerate(kf.split(yva)):
        # Tune (w, t) on cal_idx via small joint search
        rng = np.random.default_rng(1000 + fold_idx)
        best = None
        for _ in range(15000):
            a = rng.dirichlet(np.ones(3))
            pv = a[0]*p_va_unif[cal_idx] + a[1]*p_va_rsb[cal_idx] + a[2]*p_va_sub[cal_idx]
            e = (pv * classes[None, :]).sum(1)
            ts = sorted(rng.uniform(0, 3, size=3))
            yp = apply_thresholds(e, *ts)
            v = cohen_kappa_score(yva[cal_idx], yp, weights="quadratic")
            if best is None or v > best["v"]:
                best = {"w": a.tolist(), "t": ts, "v": float(v)}
        pe = best["w"][0]*p_te_unif + best["w"][1]*p_te_rsb + best["w"][2]*p_te_sub
        e_te = (pe * classes[None, :]).sum(1)
        yp_te = apply_thresholds(e_te, *best["t"])
        fold_preds.append(yp_te)
        print(f"  fold {fold_idx}: w={[f'{x:.2f}' for x in best['w']]} test κ={cohen_kappa_score(yte, yp_te, weights='quadratic'):.3f}")
    fold_preds = np.stack(fold_preds)
    from scipy.stats import mode
    yp_vote = mode(fold_preds, axis=0, keepdims=False).mode
    m_vote = metrics(yte, yp_vote)
    out["fold_vote_clip_rsb_subj"] = m_vote
    print(f"  vote κ_q={m_vote['kappa_q']:.3f} {m_vote['kappa_q_ci']}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
