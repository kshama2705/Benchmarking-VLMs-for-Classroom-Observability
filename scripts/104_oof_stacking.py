"""
MOONSHOT 15: K-fold OOF (out-of-fold) stacking — proper stacking.

Prior stacking (script 95) trained meta-learner on val, which catastrophically
overfit (val 0.194 → test 0.108). Proper stacking uses OOF predictions on TRAIN:

  For each base model (LR-unif on SigLIP-L, LR on CLIP-L, MLP, etc):
    K-fold split train → for each fold, train on K-1 folds, predict on held-out fold
    Accumulate held-out predictions as 'OOF train predictions'
  Then train meta-learner on OOF train predictions, validate on val (still!),
  evaluate on test.

This avoids val-overfit because the meta-features are train-side.

Output:
  results/sota/moonshot_oof_stacking.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CLIPL = os.path.join(BASE, "features", "daisee_clip_l_14_features.npz")
DINO = os.path.join(BASE, "features", "dinov2_vitb14_features.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_oof_stacking.json")
OOF_CACHE = os.path.join(BASE, "results", "sota", "_oof_stacking_cache.npz")
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


def fit_lr_C1(Xtr, ytr):
    """Fixed C=1.0 LR with class_weight balanced (skip val tuning for speed in OOF)."""
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000,
                             solver="lbfgs", random_state=42)
    clf.fit(Xtr, ytr)
    return clf


def oof_predictions(X, y, X_holdout, n_folds=5, seed=42):
    """K-fold OOF predictions on X plus predictions on X_holdout (avg of K models)."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), 4))
    hold_preds = []
    for cal, hold in kf.split(X):
        clf = fit_lr_C1(X[cal], y[cal])
        oof[hold] = clf.predict_proba(X[hold])
        hold_preds.append(clf.predict_proba(X_holdout))
    return oof, np.mean(hold_preds, axis=0)


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
    print("Loading encoders...", flush=True)
    d1 = np.load(SIGLIP, allow_pickle=True); Xs = d1["feat"].astype(np.float32)
    d2 = np.load(CLIPL, allow_pickle=True); Xc = d2["feat"].astype(np.float32)
    d3 = np.load(DINO, allow_pickle=True); Xd = d3["feat"].astype(np.float32)
    sp = d1["split"]; eng = d1["engagement"].astype(np.int64)
    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    ytr = eng[tr]; yva = eng[va]; yte = eng[te]
    Xstr, Xsva, Xste = Xs[tr], Xs[va], Xs[te]
    Xctr, Xcva, Xcte = Xc[tr], Xc[va], Xc[te]
    Xdtr, Xdva, Xdte = Xd[tr], Xd[va], Xd[te]

    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    out = {}

    if os.path.exists(OOF_CACHE):
        c = np.load(OOF_CACHE)
        oof_s = c["oof_s"]; te_s = c["te_s"]; va_s = c["va_s"]
        oof_c = c["oof_c"]; te_c = c["te_c"]; va_c = c["va_c"]
        oof_d = c["oof_d"]; te_d = c["te_d"]; va_d = c["va_d"]
        print("Loaded OOF cache", flush=True)
    else:
        # Build OOF + holdout (val) + test predictions for each base encoder
        print("\n[1/3] OOF SigLIP-L (5-fold)...", flush=True)
        t0 = time.time()
        oof_s, va_s = oof_predictions(Xstr, ytr, Xsva, n_folds=5, seed=42)
        _, te_s = oof_predictions(Xstr, ytr, Xste, n_folds=5, seed=42)
        print(f"  Done ({time.time()-t0:.0f}s)", flush=True)

        print("\n[2/3] OOF CLIP-L (5-fold)...", flush=True)
        t0 = time.time()
        oof_c, va_c = oof_predictions(Xctr, ytr, Xcva, n_folds=5, seed=42)
        _, te_c = oof_predictions(Xctr, ytr, Xcte, n_folds=5, seed=42)
        print(f"  Done ({time.time()-t0:.0f}s)", flush=True)

        print("\n[3/3] OOF DINOv2 (5-fold)...", flush=True)
        t0 = time.time()
        oof_d, va_d = oof_predictions(Xdtr, ytr, Xdva, n_folds=5, seed=42)
        _, te_d = oof_predictions(Xdtr, ytr, Xdte, n_folds=5, seed=42)
        print(f"  Done ({time.time()-t0:.0f}s)", flush=True)

        np.savez_compressed(OOF_CACHE,
                            oof_s=oof_s, te_s=te_s, va_s=va_s,
                            oof_c=oof_c, te_c=te_c, va_c=va_c,
                            oof_d=oof_d, te_d=te_d, va_d=va_d)

    # Build meta-features
    Xmeta_tr = np.concatenate([oof_s, oof_c, oof_d], axis=1)  # (n_tr, 12)
    Xmeta_va = np.concatenate([va_s, va_c, va_d], axis=1)
    Xmeta_te = np.concatenate([te_s, te_c, te_d], axis=1)
    print(f"\nMeta features: tr {Xmeta_tr.shape}, va {Xmeta_va.shape}, te {Xmeta_te.shape}", flush=True)

    # ============ Train meta-LR on OOF train, validate on val ============
    print("\n=== Meta-LR on OOF train (sweep C, val) ===", flush=True)
    best_lr = None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xmeta_tr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xmeta_va), weights="quadratic")
        if best_lr is None or v > best_lr["v"]:
            best_lr = {"C": C, "v": float(v), "clf": clf}
        print(f"    C={C}: val κ={v:.4f}", flush=True)
    clf_meta = best_lr["clf"]
    yp_te = clf_meta.predict(Xmeta_te)
    m_lr = metrics(yte, yp_te)
    out["meta_lr"] = {**m_lr, "C": best_lr["C"]}
    p_te_meta = clf_meta.predict_proba(Xmeta_te)
    p_va_meta = clf_meta.predict_proba(Xmeta_va)
    print(f"  Meta-LR: κ_q={m_lr['kappa_q']:.4f} {m_lr['kappa_q_ci']}", flush=True)

    # +threshold
    e_va = (p_va_meta * classes[None, :]).sum(1)
    e_te = (p_te_meta * classes[None, :]).sum(1)
    bt = tune_thresh(e_va, yva)
    m_thr = metrics(yte, apply_t(e_te, bt["t"]))
    out["meta_lr_threshold"] = {**m_thr, **bt}
    print(f"  Meta-LR + threshold: κ_q={m_thr['kappa_q']:.4f} {m_thr['kappa_q_ci']}", flush=True)

    # ============ Meta-LR with engineered features ============
    # Add E[y] per source + entropy per source
    def engineer(probs_list):
        feats = list(probs_list)
        for p in probs_list:
            e = (p * classes[None, :]).sum(1)
            feats.append(e[:, None])
            ent = -((p + 1e-12) * np.log(p + 1e-12)).sum(1)
            feats.append(ent[:, None])
        return np.concatenate(feats, axis=1)
    Xmeta_tr_e = engineer([oof_s, oof_c, oof_d])
    Xmeta_va_e = engineer([va_s, va_c, va_d])
    Xmeta_te_e = engineer([te_s, te_c, te_d])
    print(f"\nEngineered meta features: tr {Xmeta_tr_e.shape}, va {Xmeta_va_e.shape}", flush=True)

    print("\n=== Meta-LR engineered (sweep C) ===", flush=True)
    best_lr2 = None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xmeta_tr_e, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xmeta_va_e), weights="quadratic")
        if best_lr2 is None or v > best_lr2["v"]:
            best_lr2 = {"C": C, "v": float(v), "clf": clf}
    clf_meta2 = best_lr2["clf"]
    yp_te2 = clf_meta2.predict(Xmeta_te_e)
    m_lr2 = metrics(yte, yp_te2)
    out["meta_lr_engineered"] = {**m_lr2, "C": best_lr2["C"]}
    p_te_m2 = clf_meta2.predict_proba(Xmeta_te_e)
    p_va_m2 = clf_meta2.predict_proba(Xmeta_va_e)
    e_va2 = (p_va_m2 * classes[None, :]).sum(1)
    e_te2 = (p_te_m2 * classes[None, :]).sum(1)
    bt2 = tune_thresh(e_va2, yva)
    m_thr2 = metrics(yte, apply_t(e_te2, bt2["t"]))
    out["meta_lr_engineered_threshold"] = {**m_thr2, **bt2}
    print(f"  Meta-LR engineered: κ_q={m_lr2['kappa_q']:.4f}", flush=True)
    print(f"  Meta-LR engineered + threshold: κ_q={m_thr2['kappa_q']:.4f} {m_thr2['kappa_q_ci']}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
