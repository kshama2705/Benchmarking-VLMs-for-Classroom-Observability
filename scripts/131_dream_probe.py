"""
DREAM step 2 — anchor-modulated engagement probe.

For each clip i with subject s, build:
    sub  : phi(x_i) = f(x_i) - a_s          (1024-d)
    cat  : phi(x_i) = [f(x_i); f(x_i) - a_s] (2048-d)
    raw  : phi(x_i) = f(x_i)                (1024-d, baseline)
where f is frozen SigLIP-L CLS and a_s is the mean SigLIP-L feature over
subject s's K anchor frames.

Train a class-balanced LR probe on TRAIN (subject-disjoint), tune C on VAL κ_q,
then apply ordinal threshold tuning on VAL E[y] scores and evaluate on TEST.

This is the headline DREAM result. Compares against:
  raw / baseline → solo LR-unif + threshold ≈ 0.238 (prior SOTA)
  sub  → DREAM-subtract
  cat  → DREAM-concat (preserves raw signal alongside residual)

We run K=5 anchor budget by default (sweet-spot from prior literature).
"""

import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
ANCHORS = os.path.join(BASE, "features", "dream_anchors.npz")
OUT = os.path.join(BASE, "results", "dream", "dream_main.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)


def boot_ci(yt, yp, n=1000):
    nt = len(yt); out = []
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def metrics(yt, yp):
    return {
        "kappa_q": float(cohen_kappa_score(yt, yp, weights="quadratic")),
        "kappa_q_ci": boot_ci(yt, yp),
        "accuracy": float(accuracy_score(yt, yp)),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "pred_dist": {int(k): int((yp == k).sum()) for k in range(4)},
    }


def fit_lr(Xtr, ytr, Xva, yva, Cs=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0)):
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": float(v), "clf": clf}
    return best["clf"]


def bag_lr_uniform(Xtr, ytr, Xva, yva, Xte, K=20, seeds=(0, 7, 42, 2025, 1024)):
    """Same uniform-bootstrap LR bag used to achieve the prior SOTA."""
    all_te, all_va = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        bt, bv = [], []
        for _ in range(K):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            clf = fit_lr(Xtr[idx], ytr[idx], Xva, yva)
            bt.append(clf.predict_proba(Xte))
            bv.append(clf.predict_proba(Xva))
        all_te.append(np.stack(bt).mean(0))
        all_va.append(np.stack(bv).mean(0))
    return np.stack(all_te).mean(0), np.stack(all_va).mean(0)


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


def build_anchor_features(feat, subject, anchor_idx_per_subject, subjects):
    """For each clip, look up its subject's anchor mean."""
    # anchor_idx_per_subject: (n_subjects, K) ints into feat
    # subjects: (n_subjects,) array aligning with above
    anchor_means = {}
    K = anchor_idx_per_subject.shape[1]
    for si, s in enumerate(subjects):
        idx = anchor_idx_per_subject[si]
        idx = idx[idx >= 0]
        if len(idx) == 0:
            anchor_means[s] = np.zeros(feat.shape[1], dtype=np.float32)
        else:
            anchor_means[s] = feat[idx].mean(0)
    a_per_clip = np.stack([anchor_means[s] for s in subject], axis=0)
    return a_per_clip


def run_probe(Xtr, ytr, Xva, yva, Xte, yte, name):
    t0 = time.time()
    p_te, p_va = bag_lr_uniform(Xtr, ytr, Xva, yva, Xte, K=20)
    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    e_va = (p_va * classes[None]).sum(1)
    e_te = (p_te * classes[None]).sum(1)
    # baseline argmax
    yp_arg = p_te.argmax(1)
    arg_m = metrics(yte, yp_arg)
    # threshold-tuned
    b = tune_thresholds(e_va, yva)
    yp_thr = apply_thresholds(e_te, b['t'])
    thr_m = metrics(yte, yp_thr)
    dt = time.time() - t0
    print(f"[{name}] argmax κ={arg_m['kappa_q']:.4f}  thr κ={thr_m['kappa_q']:.4f} "
          f"(t={b['t']}, val={b['v']:.4f})  [{dt:.0f}s]", flush=True)
    return {"name": name, "argmax": arg_m, "threshold": thr_m,
            "thresholds": b['t'], "val_kq_at_thr": b['v'],
            "time_s": dt}


def main():
    print("Loading SigLIP-L features and anchors...", flush=True)
    d = np.load(SIGLIP, allow_pickle=True)
    feat = d['feat'].astype(np.float32)
    split = d['split']
    subject = d['subject_id']
    eng = d['engagement'].astype(np.int64)
    print(f"feat shape: {feat.shape}", flush=True)

    a = np.load(ANCHORS, allow_pickle=True)
    subjects = a['subject_ids']
    print(f"Anchors loaded for {len(subjects)} subjects", flush=True)

    tr_mask = (split == 'Train'); va_mask = (split == 'Validation'); te_mask = (split == 'Test')

    Xtr_raw = feat[tr_mask]; ytr = eng[tr_mask]
    Xva_raw = feat[va_mask]; yva = eng[va_mask]
    Xte_raw = feat[te_mask]; yte = eng[te_mask]
    sub_tr = subject[tr_mask]; sub_va = subject[va_mask]; sub_te = subject[te_mask]
    print(f"Splits: tr={len(ytr)} va={len(yva)} te={len(yte)}", flush=True)

    results = {}

    # ============ Baseline (raw features) ============
    print("\n--- BASELINE: raw SigLIP-L features ---", flush=True)
    results["baseline_raw"] = run_probe(Xtr_raw, ytr, Xva_raw, yva, Xte_raw, yte, "baseline_raw")

    # ============ Sweep over K and protocol ============
    for protocol in ["p_train", "p_zero"]:
        for K in [1, 3, 5, 10, 20]:
            key_idx = f"{protocol}_anchor_idx_k{K}"
            if key_idx not in a:
                continue
            anchor_idx = a[key_idx]

            # Build per-clip anchor means (in raw feature space)
            a_full = build_anchor_features(feat, subject, anchor_idx, subjects)
            a_tr = a_full[tr_mask]; a_va = a_full[va_mask]; a_te = a_full[te_mask]

            # SUB: subtract
            Xtr = Xtr_raw - a_tr; Xva = Xva_raw - a_va; Xte = Xte_raw - a_te
            tag = f"dream_sub_{protocol}_K{K}"
            results[tag] = run_probe(Xtr, ytr, Xva, yva, Xte, yte, tag)

            # CAT: concat (raw, residual)
            Xtr = np.concatenate([Xtr_raw, Xtr_raw - a_tr], axis=1)
            Xva = np.concatenate([Xva_raw, Xva_raw - a_va], axis=1)
            Xte = np.concatenate([Xte_raw, Xte_raw - a_te], axis=1)
            tag = f"dream_cat_{protocol}_K{K}"
            results[tag] = run_probe(Xtr, ytr, Xva, yva, Xte, yte, tag)

    print(f"\n=== Summary (threshold-tuned κ_q) ===", flush=True)
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]["threshold"]["kappa_q"]):
        kq = v["threshold"]["kappa_q"]; ci = v["threshold"]["kappa_q_ci"]
        print(f"  {kq:.4f} [{ci[0]:.3f}, {ci[1]:.3f}]   {k}", flush=True)

    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
