"""
Multi-frame concat bagged LR + threshold tuning on SigLIP-L.

Prior recipe: t=5 single frame → (N, 1024) → bagged LR → κ=0.247
This recipe: t=2,5,8 concatenated → (N, 3072) → bagged LR → κ=?

Memory says 3-frame MEAN gave only +0.01 over single-frame. Concat
preserves per-frame information rather than averaging it; if the
per-frame variation carries engagement signal beyond the mean, this
will lift κ above the 0.247 ceiling.

Also tries: 3-frame MEAN (parity check) and PER-FRAME SOLO bags.
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, accuracy_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MULTI = os.path.join(BASE, "features", "daisee_siglip_l_multiframe_features.npz")
OUT = os.path.join(BASE, "results", "sota", "multiframe_concat_bag.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)


def boot_ci(yt, yp, n=1000):
    nt = len(yt); out = []
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def fit_lr(Xtr, ytr, Xva, yva, Cs=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0)):
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=4000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": float(v), "clf": clf}
    return best["clf"]


def bag_lr(Xtr, ytr, Xva, yva, Xte, K=20, seeds=(0, 7, 42, 2025, 1024)):
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


def evaluate(Xtr, ytr, Xva, yva, Xte, yte, name):
    t0 = time.time()
    p_te, p_va = bag_lr(Xtr, ytr, Xva, yva, Xte, K=20)
    classes = np.arange(4, dtype=np.float32)
    e_va = (p_va * classes[None]).sum(1); e_te = (p_te * classes[None]).sum(1)
    yp_arg = p_te.argmax(1)
    bt = tune(e_va, yva)
    yp_thr = apply_thr(e_te, bt['t'])
    kq_arg = float(cohen_kappa_score(yte, yp_arg, weights="quadratic"))
    kq_thr = float(cohen_kappa_score(yte, yp_thr, weights="quadratic"))
    ci_arg = boot_ci(yte, yp_arg); ci_thr = boot_ci(yte, yp_thr)
    acc_thr = float(accuracy_score(yte, yp_thr))
    dt = time.time() - t0
    print(f"[{name}] arg κ={kq_arg:.4f} {ci_arg}  thr κ={kq_thr:.4f} {ci_thr} acc={acc_thr:.3f} t={bt['t']}  [{dt:.0f}s]", flush=True)
    return {"name": name, "p_te": p_te, "p_va": p_va,
            "argmax": {"kq": kq_arg, "ci": ci_arg},
            "threshold": {"kq": kq_thr, "ci": ci_thr, "t": bt['t'], "acc": acc_thr, "val_kq": bt['v']}}


def main():
    print("Loading multi-frame features...", flush=True)
    d = np.load(MULTI, allow_pickle=True)
    feat = d['feat'].astype(np.float32)   # (N, 3, 1024)
    split = d['split']; eng = d['engagement'].astype(np.int64)
    tr = (split == "Train"); va = (split == "Validation"); te = (split == "Test")

    feat_mean = feat.mean(axis=1)   # (N, 1024)
    feat_cat = feat.reshape(feat.shape[0], -1)  # (N, 3072)
    print(f"mean shape={feat_mean.shape}  cat shape={feat_cat.shape}", flush=True)

    ytr = eng[tr]; yva = eng[va]; yte = eng[te]
    results = {}

    print("\n=== A: 3-frame MEAN bagged + threshold (parity check) ===", flush=True)
    A = evaluate(feat_mean[tr], ytr, feat_mean[va], yva, feat_mean[te], yte, "mean3frame")
    results["A_mean3"] = {k: v for k, v in A.items() if k not in ("p_te", "p_va")}

    print("\n=== B: 3-frame CONCAT (3072-d) bagged + threshold ===", flush=True)
    B = evaluate(feat_cat[tr], ytr, feat_cat[va], yva, feat_cat[te], yte, "cat3frame")
    results["B_cat3"] = {k: v for k, v in B.items() if k not in ("p_te", "p_va")}

    print("\n=== C: t=5 single-frame bagged (parity with prior SOTA) ===", flush=True)
    C = evaluate(feat[tr, 1], ytr, feat[va, 1], yva, feat[te, 1], yte, "single_t5")
    results["C_single"] = {k: v for k, v in C.items() if k not in ("p_te", "p_va")}

    # Fuse A and B
    print("\n=== D: A+B mean-prob fusion + threshold ===", flush=True)
    p_va_d = 0.5 * A["p_va"] + 0.5 * B["p_va"]
    p_te_d = 0.5 * A["p_te"] + 0.5 * B["p_te"]
    classes = np.arange(4, dtype=np.float32)
    e_va = (p_va_d * classes[None]).sum(1); e_te = (p_te_d * classes[None]).sum(1)
    bt = tune(e_va, yva)
    yp = apply_thr(e_te, bt['t'])
    kq = float(cohen_kappa_score(yte, yp, weights="quadratic"))
    ci = boot_ci(yte, yp)
    acc = float(accuracy_score(yte, yp))
    print(f"[D fuse A+B] thr κ={kq:.4f} {ci} acc={acc:.3f} t={bt['t']}", flush=True)
    results["D_fuse_AB"] = {"argmax": {"kq": float(cohen_kappa_score(yte, p_te_d.argmax(1), weights='quadratic'))},
                           "threshold": {"kq": kq, "ci": ci, "t": bt['t'], "acc": acc}}

    print("\n=== Summary (sorted by threshold κ_q) ===", flush=True)
    rows = []
    for k, v in results.items():
        kq = v.get("threshold", {}).get("kq", -1)
        rows.append((k, kq, v.get("threshold", {}).get("ci")))
    for k, kq, ci in sorted(rows, key=lambda x: -x[1]):
        print(f"  {kq:.4f} {ci}   {k}", flush=True)

    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
