"""
MOONSHOT 22: Subject-centered features.

For each subject, subtract their mean feature vector. The residual encodes
"within-subject deviation" — engagement variations relative to subject's baseline.

Bag LR on:
  A) raw features (baseline)
  B) subject-centered features only
  C) concat[raw, subject-centered] features
  D) raw + per-subject mean as broadcast feature

Threshold tune and compare. Then per-subject baseline.

Output:
  results/sota/moonshot_subject_centered.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_subject_centered.json")
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


def fit_lr(Xtr, ytr, Xva, yva):
    best = None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
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
        for k in range(K):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            clf = fit_lr(Xtr[idx], ytr[idx], Xva, yva)
            bt.append(clf.predict_proba(Xte)); bv.append(clf.predict_proba(Xva))
            if (k+1) % 10 == 0:
                print(f"    seed={s} k={k+1}/{K}", flush=True)
        all_te.append(np.stack(bt).mean(0)); all_va.append(np.stack(bv).mean(0))
    return np.stack(all_te).mean(0), np.stack(all_va).mean(0)


def subject_center(X, subj):
    """Subtract subject mean from each row."""
    Xc = X.copy()
    for s in np.unique(subj):
        mask = subj == s
        if mask.sum() > 1:
            Xc[mask] -= X[mask].mean(axis=0)
    return Xc


def subject_mean_broadcast(X, subj):
    """For each row, replace with its subject's mean."""
    Xm = np.zeros_like(X)
    for s in np.unique(subj):
        mask = subj == s
        Xm[mask] = X[mask].mean(axis=0)
    return Xm


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


def per_subject_classify(p_te, p_va, yva, yte, subj_te, subj_va, classes):
    e_va = (p_va * classes[None, :]).sum(1)
    e_te = (p_te * classes[None, :]).sum(1)
    bt = tune_thresh(e_va, yva)
    yhat = np.zeros_like(yte)
    for s in np.unique(subj_te):
        mask = subj_te == s
        subj_e = e_te[mask].mean()
        c_pred = 0
        if subj_e > bt["t"][0]: c_pred = 1
        if subj_e > bt["t"][1]: c_pred = 2
        if subj_e > bt["t"][2]: c_pred = 3
        yhat[mask] = c_pred
    m = metrics(yte, yhat)
    return {**m, **bt}


def main():
    print("Loading...", flush=True)
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    subj = d["subject_id"]
    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    Xtr, ytr = X[tr], eng[tr]; Xva, yva = X[va], eng[va]; Xte, yte = X[te], eng[te]
    subj_tr = subj[tr]; subj_va = subj[va]; subj_te = subj[te]
    print(f"  Train subjects: {len(np.unique(subj_tr))}, Val: {len(np.unique(subj_va))}, Test: {len(np.unique(subj_te))}", flush=True)

    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    out = {}

    # Variant B: subject-centered features only
    print("\n[B] Bag LR on subject-centered features (residual)...", flush=True)
    Xtr_c = subject_center(Xtr, subj_tr)
    Xva_c = subject_center(Xva, subj_va)
    Xte_c = subject_center(Xte, subj_te)
    t0 = time.time()
    p_te_b, p_va_b = bag_lr(Xtr_c, ytr, Xva_c, yva, Xte_c, K=20)
    print(f"  Done ({time.time()-t0:.0f}s)", flush=True)
    m_b = metrics(yte, p_te_b.argmax(1))
    e_va_b = (p_va_b * classes[None, :]).sum(1)
    e_te_b = (p_te_b * classes[None, :]).sum(1)
    bt_b = tune_thresh(e_va_b, yva)
    m_thr_b = metrics(yte, apply_t(e_te_b, bt_b["t"]))
    m_subj_b = per_subject_classify(p_te_b, p_va_b, yva, yte, subj_te, subj_va, classes)
    out["centered_only"] = {"solo": m_b, "threshold": {**m_thr_b, **bt_b}, "per_subject": m_subj_b}
    print(f"  Solo: κ={m_b['kappa_q']:.4f}  +thresh: κ={m_thr_b['kappa_q']:.4f}  per-subj: κ={m_subj_b['kappa_q']:.4f}", flush=True)

    # Variant C: concat[raw, subject-centered]
    print("\n[C] Bag LR on [raw ⊕ centered] (2x dim)...", flush=True)
    Xtr_cat = np.concatenate([Xtr, Xtr_c], axis=1)
    Xva_cat = np.concatenate([Xva, Xva_c], axis=1)
    Xte_cat = np.concatenate([Xte, Xte_c], axis=1)
    t0 = time.time()
    p_te_c, p_va_c = bag_lr(Xtr_cat, ytr, Xva_cat, yva, Xte_cat, K=20)
    print(f"  Done ({time.time()-t0:.0f}s)", flush=True)
    m_c = metrics(yte, p_te_c.argmax(1))
    e_va_c = (p_va_c * classes[None, :]).sum(1)
    e_te_c = (p_te_c * classes[None, :]).sum(1)
    bt_c = tune_thresh(e_va_c, yva)
    m_thr_c = metrics(yte, apply_t(e_te_c, bt_c["t"]))
    m_subj_c = per_subject_classify(p_te_c, p_va_c, yva, yte, subj_te, subj_va, classes)
    out["raw_plus_centered"] = {"solo": m_c, "threshold": {**m_thr_c, **bt_c}, "per_subject": m_subj_c}
    print(f"  Solo: κ={m_c['kappa_q']:.4f}  +thresh: κ={m_thr_c['kappa_q']:.4f}  per-subj: κ={m_subj_c['kappa_q']:.4f}", flush=True)

    # Variant D: concat[raw, subject-mean]
    print("\n[D] Bag LR on [raw ⊕ subject-mean]...", flush=True)
    Xtr_m = subject_mean_broadcast(Xtr, subj_tr)
    Xva_m = subject_mean_broadcast(Xva, subj_va)
    Xte_m = subject_mean_broadcast(Xte, subj_te)
    Xtr_cd = np.concatenate([Xtr, Xtr_m], axis=1)
    Xva_cd = np.concatenate([Xva, Xva_m], axis=1)
    Xte_cd = np.concatenate([Xte, Xte_m], axis=1)
    t0 = time.time()
    p_te_d, p_va_d = bag_lr(Xtr_cd, ytr, Xva_cd, yva, Xte_cd, K=20)
    print(f"  Done ({time.time()-t0:.0f}s)", flush=True)
    m_d = metrics(yte, p_te_d.argmax(1))
    e_va_d = (p_va_d * classes[None, :]).sum(1)
    e_te_d = (p_te_d * classes[None, :]).sum(1)
    bt_d = tune_thresh(e_va_d, yva)
    m_thr_d = metrics(yte, apply_t(e_te_d, bt_d["t"]))
    m_subj_d = per_subject_classify(p_te_d, p_va_d, yva, yte, subj_te, subj_va, classes)
    out["raw_plus_subj_mean"] = {"solo": m_d, "threshold": {**m_thr_d, **bt_d}, "per_subject": m_subj_d}
    print(f"  Solo: κ={m_d['kappa_q']:.4f}  +thresh: κ={m_thr_d['kappa_q']:.4f}  per-subj: κ={m_subj_d['kappa_q']:.4f}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
