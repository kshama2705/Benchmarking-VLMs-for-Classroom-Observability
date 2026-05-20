"""
MOONSHOT 28: Per-subject classifier using mean + std features.

For each subject, compute:
  - mean E[y]
  - std E[y]
  - n_clips

Train a simple LR on val subjects (19 subjects) to predict their modal class.
Apply to test subjects.

Output:
  results/sota/moonshot_subject_meanstd.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_subject_meanstd.json")
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


def main():
    print("Loading...", flush=True)
    d = np.load(SIGLIP, allow_pickle=True)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    subj = d["subject_id"]
    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    ytr = eng[tr]; yva = eng[va]; yte = eng[te]
    subj_tr = subj[tr]; subj_va = subj[va]; subj_te = subj[te]

    c = np.load(CACHE)
    p_te = c["p_te_unif"]; p_va = c["p_va_unif"]
    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    e_va = (p_va * classes[None, :]).sum(1)
    e_te = (p_te * classes[None, :]).sum(1)

    # Build per-subject features: [mean, std, n_clips, max-min range, modal_class probability]
    def subj_features(e_arr, subj_arr):
        unique_s = np.unique(subj_arr)
        feats = {}
        for s in unique_s:
            mask = subj_arr == s
            e_s = e_arr[mask]
            feats[s] = {
                "mean": float(e_s.mean()),
                "std": float(e_s.std()),
                "median": float(np.median(e_s)),
                "min": float(e_s.min()),
                "max": float(e_s.max()),
                "q25": float(np.quantile(e_s, 0.25)),
                "q75": float(np.quantile(e_s, 0.75)),
                "n": int(mask.sum()),
            }
        return feats

    va_feats = subj_features(e_va, subj_va)
    te_feats = subj_features(e_te, subj_te)
    # Targets: per-subject majority class
    va_y = {s: int(np.round(yva[subj_va == s].mean())) for s in np.unique(subj_va)}
    te_y_per_subj = {s: yte[subj_te == s] for s in np.unique(subj_te)}
    feat_keys = ["mean", "std", "median", "min", "max", "q25", "q75", "n"]

    Xva = np.array([[va_feats[s][k] for k in feat_keys] for s in va_feats])
    yva_subj = np.array([va_y[s] for s in va_feats])
    Xte = np.array([[te_feats[s][k] for k in feat_keys] for s in te_feats])
    te_subj_list = list(te_feats.keys())

    print(f"Val: {Xva.shape} subjects, {Xva.shape[1]} features", flush=True)
    print(f"Test: {Xte.shape} subjects", flush=True)
    print(f"Val subject class distribution: {np.bincount(yva_subj, minlength=4).tolist()}", flush=True)

    out = {}

    # Baseline: use mean only with global thresholds
    bt = None
    print("\n[Baseline] Use global clip-level thresholds applied to subject mean...", flush=True)
    grid = np.arange(0, 3.01, 0.04)
    best = None
    for t1 in grid:
        for t2 in grid[grid > t1]:
            for t3 in grid[grid > t2]:
                yp = np.zeros_like(yva)
                yp[e_va > t1] = 1; yp[e_va > t2] = 2; yp[e_va > t3] = 3
                v = cohen_kappa_score(yva, yp, weights='quadratic')
                if best is None or v > best['v']:
                    best = {'t': (float(t1), float(t2), float(t3)), 'v': float(v)}
    bt = best
    yhat_base = np.zeros_like(yte)
    for s in te_subj_list:
        mask = subj_te == s
        sm = te_feats[s]["mean"]
        cp = 0
        if sm > bt["t"][0]: cp = 1
        if sm > bt["t"][1]: cp = 2
        if sm > bt["t"][2]: cp = 3
        yhat_base[mask] = cp
    m_base = metrics(yte, yhat_base)
    out["baseline_mean_only"] = m_base
    print(f"  κ_q={m_base['kappa_q']:.4f} {m_base['kappa_q_ci']}", flush=True)

    # ============ Variant A: Ridge regression on subject features → continuous pred → threshold ============
    print("\n[A] Ridge regression on subject features → continuous pred → threshold tune", flush=True)
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(Xva, yva_subj.astype(np.float32))
        pred_te = reg.predict(Xte)
        # Tune thresholds on val (predicted subject mean → subject class)
        pred_va = reg.predict(Xva)
        best_t = None
        for t1 in grid:
            for t2 in grid[grid > t1]:
                for t3 in grid[grid > t2]:
                    yp = np.zeros_like(yva_subj)
                    yp[pred_va > t1] = 1; yp[pred_va > t2] = 2; yp[pred_va > t3] = 3
                    v = cohen_kappa_score(yva_subj, yp, weights="quadratic")
                    if best_t is None or v > best_t["v"]:
                        best_t = {"t": (float(t1), float(t2), float(t3)), "v": float(v)}
        # Apply to test
        yhat = np.zeros_like(yte)
        for i, s in enumerate(te_subj_list):
            mask = subj_te == s
            sm = pred_te[i]
            cp = 0
            if sm > best_t["t"][0]: cp = 1
            if sm > best_t["t"][1]: cp = 2
            if sm > best_t["t"][2]: cp = 3
            yhat[mask] = cp
        m = metrics(yte, yhat)
        out[f"ridge_alpha{alpha}"] = {**m, **best_t}
        print(f"  α={alpha}: κ_q={m['kappa_q']:.4f}  thresh val κ={best_t['v']:.3f}", flush=True)

    # ============ Variant B: Just use mean feature with subject-level threshold tune ============
    print("\n[B] Subject-mean with subject-level threshold tune (using val subject means)", flush=True)
    va_means = np.array([va_feats[s]["mean"] for s in va_feats])
    best_t = None
    for t1 in grid:
        for t2 in grid[grid > t1]:
            for t3 in grid[grid > t2]:
                yp = np.zeros_like(yva_subj)
                yp[va_means > t1] = 1; yp[va_means > t2] = 2; yp[va_means > t3] = 3
                v = cohen_kappa_score(yva_subj, yp, weights="quadratic")
                if best_t is None or v > best_t["v"]:
                    best_t = {"t": (float(t1), float(t2), float(t3)), "v": float(v)}
    print(f"  Best subject-level thresholds: {best_t['t']}  val κ={best_t['v']:.3f}", flush=True)
    yhat_b = np.zeros_like(yte)
    for s in te_subj_list:
        mask = subj_te == s
        sm = te_feats[s]["mean"]
        cp = 0
        if sm > best_t["t"][0]: cp = 1
        if sm > best_t["t"][1]: cp = 2
        if sm > best_t["t"][2]: cp = 3
        yhat_b[mask] = cp
    m_b = metrics(yte, yhat_b)
    out["mean_subj_thresh"] = {**m_b, **best_t}
    print(f"  κ_q={m_b['kappa_q']:.4f} {m_b['kappa_q_ci']}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
