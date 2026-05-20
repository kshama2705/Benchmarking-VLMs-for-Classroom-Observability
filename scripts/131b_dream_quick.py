"""
DREAM quick — single-config sanity run to get the headline number fast.

Uses K=5 anchors (sweet spot prior), p_train protocol, sub modulation,
and one LR per C-value (not bagged) to land a result in <1 minute.

If sub works at all, the bagged + threshold-tuned version should follow.
"""

import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
ANCHORS = os.path.join(BASE, "features", "dream_anchors.npz")
OUT = os.path.join(BASE, "results", "dream", "dream_quick.json")
RNG = np.random.default_rng(42)


def boot_ci(yt, yp, n=1000):
    nt = len(yt); out = []
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def fit_lr_one(Xtr, ytr, Xva, yva, Cs=(0.001, 0.01, 0.1, 1.0, 10.0)):
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                                 solver="lbfgs", random_state=42, n_jobs=1)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": float(v), "clf": clf}
    return best


def tune_thresholds(e_va, y_va, step=0.04):
    grid = np.arange(0.0, 3.01, step)
    best = None
    for t1 in grid:
        for t2 in grid[grid > t1]:
            for t3 in grid[grid > t2]:
                yp = np.zeros_like(e_va, dtype=int)
                yp[e_va > t1] = 1; yp[e_va > t2] = 2; yp[e_va > t3] = 3
                v = cohen_kappa_score(y_va, yp, weights='quadratic')
                if best is None or v > best['v']:
                    best = {'t': (float(t1), float(t2), float(t3)), 'v': float(v)}
    return best


def apply_thresholds(e, t):
    yp = np.zeros_like(e, dtype=int)
    yp[e > t[0]] = 1; yp[e > t[1]] = 2; yp[e > t[2]] = 3
    return yp


def main():
    print("Loading...", flush=True)
    d = np.load(SIGLIP, allow_pickle=True)
    feat = d['feat'].astype(np.float32)
    split = d['split']; subject = d['subject_id']; eng = d['engagement'].astype(np.int64)
    a = np.load(ANCHORS, allow_pickle=True)
    subjects = a['subject_ids']

    tr_mask = (split == 'Train'); va_mask = (split == 'Validation'); te_mask = (split == 'Test')
    Xtr_raw = feat[tr_mask]; ytr = eng[tr_mask]
    Xva_raw = feat[va_mask]; yva = eng[va_mask]
    Xte_raw = feat[te_mask]; yte = eng[te_mask]

    classes = np.array([0, 1, 2, 3], dtype=np.float32)

    results = {}

    def run(Xtr, Xva, Xte, name):
        t0 = time.time()
        b = fit_lr_one(Xtr, ytr, Xva, yva)
        clf = b["clf"]
        p_te = clf.predict_proba(Xte); p_va = clf.predict_proba(Xva)
        e_va = (p_va * classes[None]).sum(1); e_te = (p_te * classes[None]).sum(1)
        yp_arg = p_te.argmax(1)
        bt = tune_thresholds(e_va, yva)
        yp_thr = apply_thresholds(e_te, bt['t'])
        kq_arg = float(cohen_kappa_score(yte, yp_arg, weights="quadratic"))
        kq_thr = float(cohen_kappa_score(yte, yp_thr, weights="quadratic"))
        ci_arg = boot_ci(yte, yp_arg, n=500)
        ci_thr = boot_ci(yte, yp_thr, n=500)
        dt = time.time() - t0
        print(f"[{name}] C={b['C']}  val={b['v']:.4f}  argmax κ={kq_arg:.4f} {ci_arg}  thr κ={kq_thr:.4f} {ci_thr} t={bt['t']}  [{dt:.1f}s]", flush=True)
        return {"C": b["C"], "val_kq": b["v"],
                "argmax": {"kq": kq_arg, "ci": ci_arg, "acc": float(accuracy_score(yte, yp_arg))},
                "threshold": {"kq": kq_thr, "ci": ci_thr, "t": bt['t'], "val_kq": bt['v'], "acc": float(accuracy_score(yte, yp_thr))}}

    # Baseline
    print("\n=== BASELINE (raw) ===", flush=True)
    results["baseline_raw"] = run(Xtr_raw, Xva_raw, Xte_raw, "baseline_raw")

    # DREAM-sub at K=5, p_train
    for K in [1, 3, 5, 10]:
        for proto in ["p_train", "p_zero"]:
            idx_key = f"{proto}_anchor_idx_k{K}"
            if idx_key not in a:
                continue
            anchor_idx = a[idx_key]
            # Build per-subject anchor mean
            anchor_means = {}
            for si, s in enumerate(subjects):
                idx = anchor_idx[si]
                idx = idx[idx >= 0]
                anchor_means[s] = feat[idx].mean(0) if len(idx) > 0 else np.zeros(feat.shape[1], np.float32)
            a_full = np.stack([anchor_means[s] for s in subject], axis=0)
            a_tr = a_full[tr_mask]; a_va = a_full[va_mask]; a_te = a_full[te_mask]

            Xtr = Xtr_raw - a_tr; Xva = Xva_raw - a_va; Xte = Xte_raw - a_te
            results[f"sub_{proto}_K{K}"] = run(Xtr, Xva, Xte, f"sub_{proto}_K{K}")

            Xtr = np.concatenate([Xtr_raw, Xtr_raw - a_tr], axis=1)
            Xva = np.concatenate([Xva_raw, Xva_raw - a_va], axis=1)
            Xte = np.concatenate([Xte_raw, Xte_raw - a_te], axis=1)
            results[f"cat_{proto}_K{K}"] = run(Xtr, Xva, Xte, f"cat_{proto}_K{K}")

    print("\n=== Sorted by threshold κ_q ===", flush=True)
    items = [(k, v["threshold"]["kq"], v["threshold"]["ci"]) for k, v in results.items()]
    for k, kq, ci in sorted(items, key=lambda x: -x[1]):
        print(f"  {kq:.4f} {ci}   {k}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
