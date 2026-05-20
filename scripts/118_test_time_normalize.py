"""
MOONSHOT 29: Test-time feature normalization.

Hypothesis: train/test domain shift causes the κ ≈ 0.24 ceiling. By normalizing
test features using test stats (mean, std), we may push them closer to train
distribution.

Variants:
  A) Standardize both train and test with train stats (baseline)
  B) Standardize train with train stats, test with test stats
  C) Standardize train+test together (uses test stats during training)
  D) Subtract test mean from test only (no scaling)
  E) Apply PCA fit on train, transform test (baseline)
  F) Apply PCA fit on test, project train+test (test-distribution PCA)

Output:
  results/sota/moonshot_test_time_norm.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_test_time_norm.json")
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
        for k in range(K):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            clf = fit_lr(Xtr[idx], ytr[idx], Xva, yva)
            bt.append(clf.predict_proba(Xte)); bv.append(clf.predict_proba(Xva))
            if (k+1) % 10 == 0:
                print(f"    seed={s} k={k+1}/{K}", flush=True)
        all_te.append(np.stack(bt).mean(0)); all_va.append(np.stack(bv).mean(0))
    return np.stack(all_te).mean(0), np.stack(all_va).mean(0)


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


def per_subject(e_te, subj_te, yte, t):
    yhat = np.zeros_like(yte)
    for s in np.unique(subj_te):
        mask = subj_te == s
        sm = e_te[mask].mean()
        cp = 0
        if sm > t[0]: cp = 1
        if sm > t[1]: cp = 2
        if sm > t[2]: cp = 3
        yhat[mask] = cp
    return metrics(yte, yhat)


def main():
    print("Loading...", flush=True)
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    subj = d["subject_id"]
    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    Xtr, ytr = X[tr], eng[tr]; Xva, yva = X[va], eng[va]; Xte, yte = X[te], eng[te]
    subj_te = subj[te]

    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    out = {}

    # Variant: standardize each split with its own stats
    print("\n[B] Standardize each split independently (test uses test stats)", flush=True)
    sc_tr = StandardScaler().fit(Xtr)
    sc_va = StandardScaler().fit(Xva)
    sc_te = StandardScaler().fit(Xte)
    Xtr_s = sc_tr.transform(Xtr).astype(np.float32)
    Xva_s = sc_va.transform(Xva).astype(np.float32)
    Xte_s = sc_te.transform(Xte).astype(np.float32)
    t0 = time.time()
    p_te, p_va = bag_lr(Xtr_s, ytr, Xva_s, yva, Xte_s, K=20)
    print(f"  Bag done ({time.time()-t0:.0f}s)", flush=True)
    e_va = (p_va * classes[None, :]).sum(1); e_te = (p_te * classes[None, :]).sum(1)
    bt = tune_thresh(e_va, yva)
    m_thr = metrics(yte, (lambda e, t: np.array([0 + (e > t[0]) + (e > t[1]) + (e > t[2])]).flatten())(e_te, bt["t"]) if False else (lambda e, t: np.where(e > t[2], 3, np.where(e > t[1], 2, np.where(e > t[0], 1, 0))).astype(int))(e_te, bt["t"]))
    m_subj = per_subject(e_te, subj_te, yte, bt["t"])
    out["per_split_stats"] = {"threshold": m_thr, "per_subject": m_subj}
    print(f"  Per-split: +thresh κ={m_thr['kappa_q']:.4f}  per-subj κ={m_subj['kappa_q']:.4f}", flush=True)

    # Variant: standardize using ALL data (train+val+test)
    print("\n[C] Standardize using all data combined", flush=True)
    sc_all = StandardScaler().fit(X)
    Xtr_a = sc_all.transform(Xtr).astype(np.float32)
    Xva_a = sc_all.transform(Xva).astype(np.float32)
    Xte_a = sc_all.transform(Xte).astype(np.float32)
    t0 = time.time()
    p_te, p_va = bag_lr(Xtr_a, ytr, Xva_a, yva, Xte_a, K=20)
    print(f"  Bag done ({time.time()-t0:.0f}s)", flush=True)
    e_va = (p_va * classes[None, :]).sum(1); e_te = (p_te * classes[None, :]).sum(1)
    bt = tune_thresh(e_va, yva)
    yhat = np.where(e_te > bt["t"][2], 3, np.where(e_te > bt["t"][1], 2, np.where(e_te > bt["t"][0], 1, 0))).astype(int)
    m_thr = metrics(yte, yhat)
    m_subj = per_subject(e_te, subj_te, yte, bt["t"])
    out["all_data_stats"] = {"threshold": m_thr, "per_subject": m_subj}
    print(f"  All-data: +thresh κ={m_thr['kappa_q']:.4f}  per-subj κ={m_subj['kappa_q']:.4f}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
