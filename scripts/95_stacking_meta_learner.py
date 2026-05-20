"""
MOONSHOT 6: Stacking meta-learner over all cached probability outputs.

Use cached probs from all prior moonshots as input features (32-d: 8 sources × 4 classes),
train a simple LR / GBC meta-learner on val to predict engagement, evaluate on test.

Sources:
  1. LR-unif (SigLIP-L)
  2. LR-RSB45 (SigLIP-L)
  3. CLIP-L LR-unif
  4. MLP bag (SigLIP-L)
  5. Subject-bag (SigLIP-L)
  6. L0-vs-rest binary detector (1-d: p(L0))

Plus engineered features:
  7. E[y] per source × 6 = 6-d
  8. Argmax entropy per source = 6-d

Train meta-learner LR on val (with class weights), evaluate on test.

Output:
  results/sota/moonshot_stacking.json
"""
import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_stacking.json")
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


def load_or_skip(path, key_te, key_va, name):
    if not os.path.exists(path):
        print(f"  SKIP {name}: missing {path}", flush=True)
        return None, None
    d = np.load(path)
    if key_te not in d.files:
        print(f"  SKIP {name}: missing key {key_te}", flush=True)
        return None, None
    return d[key_te], d[key_va]


def main():
    print("Loading labels...", flush=True)
    d = np.load(SIGLIP, allow_pickle=True)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    yva = eng[sp == "Validation"]; yte = eng[sp == "Test"]

    print("\nLoading cached probability sources...", flush=True)
    sources = []
    candidates = [
        ("_bag_cache.npz", "p_te_unif", "p_va_unif", "LR-unif"),
        ("_bag_cache.npz", "p_te_rsb", "p_va_rsb", "LR-RSB45"),
        ("_clipl_bag_cache.npz", "p_te", "p_va", "CLIP-L"),
        ("_mlp_bag_cache.npz", "p_te", "p_va", "MLP"),
        ("_subj_bag_cache.npz", "p_te", "p_va", "Subj-bag"),
        ("_l0_bag_cache.npz", "p_te_l0", "p_va_l0", "L0-binary"),
    ]
    for fname, kt, kv, name in candidates:
        path = os.path.join(BASE, "results", "sota", fname)
        pt, pv = load_or_skip(path, kt, kv, name)
        if pt is not None:
            sources.append({"name": name, "p_te": pt, "p_va": pv})
            print(f"  loaded {name}: te shape={pt.shape}", flush=True)

    # Build feature matrices
    feats_va = []; feats_te = []; src_names = []
    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    for s in sources:
        if s["p_va"].ndim == 1:
            # Binary L0 detector
            feats_va.append(s["p_va"][:, None])
            feats_te.append(s["p_te"][:, None])
            src_names.append(s["name"] + "_p")
        else:
            # Multi-class: full probs (4-d)
            feats_va.append(s["p_va"])
            feats_te.append(s["p_te"])
            for c in range(4):
                src_names.append(f"{s['name']}_p{c}")
            # E[y] (1-d engineered)
            e_va = (s["p_va"] * classes[None, :]).sum(1)
            e_te = (s["p_te"] * classes[None, :]).sum(1)
            feats_va.append(e_va[:, None]); feats_te.append(e_te[:, None])
            src_names.append(f"{s['name']}_Ey")
            # Entropy (1-d)
            ent_va = -((s["p_va"] + 1e-12) * np.log(s["p_va"] + 1e-12)).sum(1)
            ent_te = -((s["p_te"] + 1e-12) * np.log(s["p_te"] + 1e-12)).sum(1)
            feats_va.append(ent_va[:, None]); feats_te.append(ent_te[:, None])
            src_names.append(f"{s['name']}_ent")
    Xva_meta = np.concatenate(feats_va, axis=1)
    Xte_meta = np.concatenate(feats_te, axis=1)
    print(f"\nMeta features: val {Xva_meta.shape}, test {Xte_meta.shape}", flush=True)

    out = {"n_sources": len(sources), "source_names": [s["name"] for s in sources], "meta_feat_dim": Xva_meta.shape[1]}

    # ============ Meta LR ============
    print("\n[A] Meta LR (sweep C, val-tune)...", flush=True)
    best_lr = None
    for C in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=20000,
                                 solver="lbfgs", random_state=42)
        # Train on val (since meta features are val/test only)
        # Use 5-fold CV ON VAL to pick best C, then refit on full val
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        kqs = []
        for cal, hold in skf.split(Xva_meta, yva):
            clf.fit(Xva_meta[cal], yva[cal])
            v = cohen_kappa_score(yva[hold], clf.predict(Xva_meta[hold]), weights="quadratic")
            kqs.append(v)
        mean_kq = float(np.mean(kqs))
        if best_lr is None or mean_kq > best_lr["v"]:
            best_lr = {"C": C, "v": mean_kq}
        print(f"    C={C}: 5-fold val κ={mean_kq:.4f}", flush=True)
    # Final fit on full val
    clf_final = LogisticRegression(C=best_lr["C"], class_weight="balanced", max_iter=20000,
                                    solver="lbfgs", random_state=42)
    clf_final.fit(Xva_meta, yva)
    yp_te = clf_final.predict(Xte_meta)
    m_lr = metrics(yte, yp_te)
    out["meta_LR"] = {**m_lr, "C": best_lr["C"], "val_kq_cv": best_lr["v"]}
    print(f"  Meta LR: κ_q={m_lr['kappa_q']:.4f} {m_lr['kappa_q_ci']}  C={best_lr['C']}", flush=True)
    # +threshold via E[y] on test probs
    p_te = clf_final.predict_proba(Xte_meta)
    p_va = clf_final.predict_proba(Xva_meta)
    e_va_m = (p_va * classes[None, :]).sum(1)
    e_te_m = (p_te * classes[None, :]).sum(1)
    bt = tune_thresh(e_va_m, yva)
    m_thr = metrics(yte, apply_t(e_te_m, bt["t"]))
    out["meta_LR_threshold"] = {**m_thr, **bt}
    print(f"  Meta LR + threshold: κ_q={m_thr['kappa_q']:.4f} {m_thr['kappa_q_ci']}", flush=True)

    # ============ Meta GBC ============
    print("\n[B] Meta GBC...", flush=True)
    gbc = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=6,
                                         class_weight="balanced", random_state=42)
    gbc.fit(Xva_meta, yva)
    yp_te_gbc = gbc.predict(Xte_meta)
    m_gbc = metrics(yte, yp_te_gbc)
    out["meta_GBC"] = m_gbc
    print(f"  Meta GBC: κ_q={m_gbc['kappa_q']:.4f} {m_gbc['kappa_q_ci']}", flush=True)
    p_te_gbc = gbc.predict_proba(Xte_meta)
    p_va_gbc = gbc.predict_proba(Xva_meta)
    e_va_g = (p_va_gbc * classes[None, :]).sum(1)
    e_te_g = (p_te_gbc * classes[None, :]).sum(1)
    btg = tune_thresh(e_va_g, yva)
    m_gbc_thr = metrics(yte, apply_t(e_te_g, btg["t"]))
    out["meta_GBC_threshold"] = {**m_gbc_thr, **btg}
    print(f"  Meta GBC + threshold: κ_q={m_gbc_thr['kappa_q']:.4f} {m_gbc_thr['kappa_q_ci']}", flush=True)

    # ============ Meta ensemble: LR + GBC + best fold ============
    print("\n[C] Meta-ensemble: average meta-LR + meta-GBC probs...", flush=True)
    p_te_avg = (p_te + p_te_gbc) / 2
    p_va_avg = (p_va + p_va_gbc) / 2
    e_va_a = (p_va_avg * classes[None, :]).sum(1)
    e_te_a = (p_te_avg * classes[None, :]).sum(1)
    bta = tune_thresh(e_va_a, yva)
    m_avg = metrics(yte, apply_t(e_te_a, bta["t"]))
    out["meta_avg_threshold"] = {**m_avg, **bta}
    print(f"  Meta avg + threshold: κ_q={m_avg['kappa_q']:.4f} {m_avg['kappa_q_ci']}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
