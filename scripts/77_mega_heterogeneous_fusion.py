"""
Mega heterogeneous fusion: combine multiple bagged model classes + encoders +
feature types via late fusion.

Components (each separately bagged 5 seeds × bags):
  1. LR on SigLIP-L full features (vanilla bagging)
  2. LR on SigLIP-L with random 40% feature subsets (RSB)
  3. GBC on SigLIP-L
  4. LR on SigLIP-SO400M features
  5. LR on EMBER 73-d explicit signals

Late-fuse 5-way with val-tuned weights (Dirichlet sampling + grid).

Output:
  results/sota/mega_heterogeneous_fusion.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP_L = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
SIGLIP_SO400M = os.path.join(BASE, "features", "daisee_siglip_so400m_features.npz")
SIGNALS = os.path.join(BASE, "features", "daisee_face_signals.npz")
OUT = os.path.join(BASE, "results", "sota", "mega_heterogeneous_fusion.json")
LOG = os.path.join(BASE, "results", "sota", "mega_hetero_status.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)
OUTER_SEEDS = [0, 7, 42, 2025, 1024]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def boot_kq(yt, yp, n=1000):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def fit_lr(Xtr, ytr, Xva, yva):
    best = None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": float(v), "clf": clf}
    return best["clf"], best["v"], best["C"]


def metrics(yt, yp):
    return {
        "kappa_q": float(cohen_kappa_score(yt, yp, weights="quadratic")),
        "kappa_q_ci": boot_kq(yt, yp),
        "accuracy": float(accuracy_score(yt, yp)),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "pred_dist": {int(k): int((yp == k).sum()) for k in range(4)},
    }


def align(target_ids, source_ids, source_feats):
    s_idx = {c: i for i, c in enumerate(source_ids)}
    out = np.zeros((len(target_ids), source_feats.shape[1]), dtype=np.float32)
    for j, c in enumerate(target_ids):
        if c in s_idx:
            out[j] = source_feats[s_idx[c]]
    return out


def bag_lr(Xtr, ytr, Xva, yva, Xte, K=30, seeds=OUTER_SEEDS, subspace_frac=1.0):
    """Bagged LR. Returns (test_probs averaged, val_probs averaged)."""
    n_tr = len(ytr); D = Xtr.shape[1]
    all_te = []; all_va = []
    for outer_seed in seeds:
        rng = np.random.default_rng(outer_seed)
        bag_te = []; bag_va = []
        for _ in range(K):
            row_idx = rng.integers(0, n_tr, size=n_tr)
            if subspace_frac < 1.0:
                n_features = int(D * subspace_frac)
                col_idx = rng.choice(D, size=n_features, replace=False)
                Xtr_sub = Xtr[row_idx][:, col_idx]
                Xva_sub = Xva[:, col_idx]
                Xte_sub = Xte[:, col_idx]
            else:
                Xtr_sub = Xtr[row_idx]; Xva_sub = Xva; Xte_sub = Xte
            clf, _, _ = fit_lr(Xtr_sub, ytr[row_idx], Xva_sub, yva)
            bag_te.append(clf.predict_proba(Xte_sub))
            bag_va.append(clf.predict_proba(Xva_sub))
        all_te.append(np.stack(bag_te).mean(axis=0))
        all_va.append(np.stack(bag_va).mean(axis=0))
    return np.stack(all_te).mean(axis=0), np.stack(all_va).mean(axis=0)


def bag_gbc(Xtr, ytr, Xva, yva, Xte, K=20, seeds=OUTER_SEEDS, max_iter=300, lr=0.05, max_depth=7):
    n_tr = len(ytr)
    all_te = []; all_va = []
    for outer_seed in seeds:
        rng = np.random.default_rng(outer_seed)
        bag_te = []; bag_va = []
        for k in range(K):
            idx = rng.integers(0, n_tr, size=n_tr)
            clf = HistGradientBoostingClassifier(
                max_iter=max_iter, learning_rate=lr, max_depth=max_depth,
                class_weight="balanced",
                random_state=int(outer_seed * 1000 + k))
            clf.fit(Xtr[idx], ytr[idx])
            bag_te.append(clf.predict_proba(Xte))
            bag_va.append(clf.predict_proba(Xva))
        all_te.append(np.stack(bag_te).mean(axis=0))
        all_va.append(np.stack(bag_va).mean(axis=0))
    return np.stack(all_te).mean(axis=0), np.stack(all_va).mean(axis=0)


def search_dirichlet(probs_va_list, y_va, n_iter=20000, seed=42):
    """Dirichlet random-weight search over the list of probability matrices."""
    rng = np.random.default_rng(seed)
    n = len(probs_va_list)
    best = None
    for _ in range(n_iter):
        ws = rng.dirichlet(np.ones(n) * 1.2)
        p = sum(probs_va_list[j] * ws[j] for j in range(n))
        v = cohen_kappa_score(y_va, p.argmax(axis=1), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"ws": ws.tolist(), "v": float(v)}
    return best


def main():
    open(LOG, "w").close()
    log("Loading...")
    d = np.load(SIGLIP_L, allow_pickle=True)
    clip_ids = d["clip_id"]; splits = d["split"]; eng = d["engagement"].astype(np.int64)
    sig_feats = d["feat"].astype(np.float32)

    d_so = np.load(SIGLIP_SO400M, allow_pickle=True)
    so_feats = align(clip_ids, d_so["clip_id"], d_so["feat"].astype(np.float32))

    d_sf = np.load(SIGNALS, allow_pickle=True)
    explicit_raw = np.nan_to_num(np.concatenate([
        d_sf["blendshapes"], d_sf["head_pose"], d_sf["eye_gaze"], d_sf["landmark_summary"]], axis=1))
    explicit = align(clip_ids, d_sf["clip_id"], explicit_raw)

    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    sc = StandardScaler().fit(explicit[tr])
    explicit_z = sc.transform(explicit).astype(np.float32)

    Xtr_sig, Xva_sig, Xte_sig = sig_feats[tr], sig_feats[va], sig_feats[te]
    Xtr_so, Xva_so, Xte_so = so_feats[tr], so_feats[va], so_feats[te]
    Xtr_exp, Xva_exp, Xte_exp = explicit_z[tr], explicit_z[va], explicit_z[te]
    ytr, yva, yte = eng[tr], eng[va], eng[te]

    log(f"Train={tr.sum()} Val={va.sum()} Test={te.sum()}")

    components = {}

    # 1. LR bagged on SigLIP-L (vanilla)
    log("\n[1/5] LR bagged on SigLIP-L (vanilla, K=30)...")
    t0 = time.time()
    p_te, p_va = bag_lr(Xtr_sig, ytr, Xva_sig, yva, Xte_sig, K=30, subspace_frac=1.0)
    components["LR_SigLIP_L"] = (p_va, p_te)
    log(f"  done in {time.time()-t0:.0f}s. solo κ_q = {cohen_kappa_score(yte, p_te.argmax(axis=1), weights='quadratic'):.3f}")

    # 2. RSB LR on SigLIP-L (frac=0.45)
    log("\n[2/5] LR bagged on SigLIP-L (RSB, frac=0.45, K=30)...")
    t0 = time.time()
    p_te, p_va = bag_lr(Xtr_sig, ytr, Xva_sig, yva, Xte_sig, K=30, subspace_frac=0.45)
    components["LR_SigLIP_L_RSB45"] = (p_va, p_te)
    log(f"  done in {time.time()-t0:.0f}s. solo κ_q = {cohen_kappa_score(yte, p_te.argmax(axis=1), weights='quadratic'):.3f}")

    # 3. GBC bagged on SigLIP-L
    log("\n[3/5] GBC bagged on SigLIP-L (K=20, max_iter=300, lr=0.05, depth=7)...")
    t0 = time.time()
    p_te, p_va = bag_gbc(Xtr_sig, ytr, Xva_sig, yva, Xte_sig, K=20)
    components["GBC_SigLIP_L"] = (p_va, p_te)
    log(f"  done in {time.time()-t0:.0f}s. solo κ_q = {cohen_kappa_score(yte, p_te.argmax(axis=1), weights='quadratic'):.3f}")

    # 4. LR bagged on SO400M
    log("\n[4/5] LR bagged on SO400M (K=30)...")
    t0 = time.time()
    p_te, p_va = bag_lr(Xtr_so, ytr, Xva_so, yva, Xte_so, K=30, subspace_frac=1.0)
    components["LR_SO400M"] = (p_va, p_te)
    log(f"  done in {time.time()-t0:.0f}s. solo κ_q = {cohen_kappa_score(yte, p_te.argmax(axis=1), weights='quadratic'):.3f}")

    # 5. LR bagged on EMBER explicit
    log("\n[5/5] LR bagged on EMBER explicit (K=30)...")
    t0 = time.time()
    p_te, p_va = bag_lr(Xtr_exp, ytr, Xva_exp, yva, Xte_exp, K=30, subspace_frac=1.0)
    components["LR_explicit"] = (p_va, p_te)
    log(f"  done in {time.time()-t0:.0f}s. solo κ_q = {cohen_kappa_score(yte, p_te.argmax(axis=1), weights='quadratic'):.3f}")

    # Late fusion
    log("\n=== Late fusion ===")
    out = {}
    names = list(components.keys())

    # Solo results
    for name, (pva, pte) in components.items():
        out[f"solo_{name}"] = metrics(yte, pte.argmax(axis=1))

    # All-5 fusion
    probs_va_list = [components[n][0] for n in names]
    probs_te_list = [components[n][1] for n in names]
    best = search_dirichlet(probs_va_list, yva, n_iter=20000)
    ws = np.array(best["ws"])
    p_fuse = sum(probs_te_list[j] * ws[j] for j in range(len(names)))
    yhat = p_fuse.argmax(axis=1)
    m = metrics(yte, yhat)
    out["fusion_5way"] = {**m, "weights": dict(zip(names, [float(w) for w in ws])), "val_kq": best["v"]}
    log(f"  5-way: κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  val={best['v']:.3f}")
    log(f"  ws: {dict(zip(names, [f'{w:.3f}' for w in ws]))}")

    # Pairwise (just for diagnostic)
    log("\n=== Pair fusions (LR_SigLIP_L + each) ===")
    base = "LR_SigLIP_L"
    for other in names:
        if other == base: continue
        pva_b, pte_b = components[base]
        pva_o, pte_o = components[other]
        best_w = None
        for w in np.linspace(0.0, 1.0, 41):
            pv = w * pva_o + (1 - w) * pva_b
            v = cohen_kappa_score(yva, pv.argmax(axis=1), weights="quadratic")
            if best_w is None or v > best_w["v"]:
                best_w = {"w": float(w), "v": float(v)}
        wb = best_w["w"]
        p_pair = wb * pte_o + (1 - wb) * pte_b
        m = metrics(yte, p_pair.argmax(axis=1))
        out[f"pair_{base}+{other}"] = {**m, "w": wb, "val_kq": best_w["v"]}
        log(f"  {base}+{other:25} (w_other={wb:.2f})  κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  val={best_w['v']:.3f}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {OUT}")
    log("MEGA_HETERO_DONE")


if __name__ == "__main__":
    main()
