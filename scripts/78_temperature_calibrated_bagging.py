"""
Temperature-calibrated bagging.

Standard bagging averages raw probe probabilities, which can be overconfident.
Apply temperature scaling per probe (tuned on val to maximize κ) before averaging.

Also test L1-penalty LR as an alternative bag-member.

Output:
  results/sota/temperature_bagging.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from scipy.special import softmax

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
OUT = os.path.join(BASE, "results", "sota", "temperature_bagging.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)


def boot_kq(yt, yp, n=1000):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def fit_lr(Xtr, ytr, Xva, yva, penalty="l2"):
    best = None
    Cs = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    solver = "lbfgs" if penalty == "l2" else "saga"
    for C in Cs:
        clf = LogisticRegression(C=C, penalty=penalty, class_weight="balanced",
                                 max_iter=5000 if penalty == "l1" else 10000,
                                 solver=solver, random_state=42)
        try:
            clf.fit(Xtr, ytr)
        except Exception:
            continue
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


def temperature_scale(logits, T):
    """Apply softmax with temperature."""
    return softmax(logits / T, axis=-1)


def find_temperature(logits_va, y_va, T_grid=None):
    if T_grid is None:
        T_grid = [0.5, 0.7, 1.0, 1.3, 1.7, 2.0, 3.0, 5.0]
    best = None
    for T in T_grid:
        p = temperature_scale(logits_va, T)
        v = cohen_kappa_score(y_va, p.argmax(axis=1), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"T": float(T), "v": float(v)}
    return best["T"]


def bag_probes_with_temperature(Xtr, ytr, Xva, yva, Xte, K=30, seeds=[0, 7, 42, 2025, 1024],
                                  penalty="l2", temperature=True):
    """Bag LR probes; optionally apply per-probe temperature scaling on val before averaging."""
    n_tr = len(ytr)
    all_te = []; all_va = []
    for outer_seed in seeds:
        rng = np.random.default_rng(outer_seed)
        bag_te = []; bag_va = []
        for k in range(K):
            idx = rng.integers(0, n_tr, size=n_tr)
            clf, _, _ = fit_lr(Xtr[idx], ytr[idx], Xva, yva, penalty=penalty)
            logits_va = clf.decision_function(Xva)
            logits_te = clf.decision_function(Xte)
            # decision_function for multiclass returns (N, n_classes)
            if temperature:
                T = find_temperature(logits_va, yva)
                p_va = temperature_scale(logits_va, T)
                p_te = temperature_scale(logits_te, T)
            else:
                # Default scaling (T=1) — same as predict_proba
                p_va = clf.predict_proba(Xva)
                p_te = clf.predict_proba(Xte)
            bag_te.append(p_te); bag_va.append(p_va)
        all_te.append(np.stack(bag_te).mean(axis=0))
        all_va.append(np.stack(bag_va).mean(axis=0))
    return np.stack(all_te).mean(axis=0), np.stack(all_va).mean(axis=0)


def main():
    print("Loading SigLIP-L features...")
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    splits = d["split"]; eng = d["engagement"].astype(np.int64)
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    Xtr, ytr = X[tr], eng[tr]; Xva, yva = X[va], eng[va]; Xte, yte = X[te], eng[te]

    out = {}

    print("\n[1/4] L2 LR bagged (no temperature)...")
    t0 = time.time()
    p_te, p_va = bag_probes_with_temperature(Xtr, ytr, Xva, yva, Xte, K=30, penalty="l2", temperature=False)
    yhat = p_te.argmax(axis=1)
    m = metrics(yte, yhat)
    out["L2_no_temp"] = m
    print(f"  κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  ({time.time()-t0:.0f}s)")

    print("\n[2/4] L2 LR bagged WITH temperature...")
    t0 = time.time()
    p_te_T, p_va_T = bag_probes_with_temperature(Xtr, ytr, Xva, yva, Xte, K=30, penalty="l2", temperature=True)
    yhat = p_te_T.argmax(axis=1)
    m = metrics(yte, yhat)
    out["L2_with_temp"] = m
    print(f"  κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  ({time.time()-t0:.0f}s)")

    print("\n[3/4] no-temp + with-temp fusion...")
    # Fuse the two L2 bags (calibrated vs uncalibrated)
    best_w = None
    for w in np.linspace(0.0, 1.0, 41):
        pv = w * p_va_T + (1 - w) * p_va
        v = cohen_kappa_score(yva, pv.argmax(axis=1), weights="quadratic")
        if best_w is None or v > best_w["v"]:
            best_w = {"w": float(w), "v": float(v)}
    wb = best_w["w"]
    p_fuse = wb * p_te_T + (1 - wb) * p_te
    m = metrics(yte, p_fuse.argmax(axis=1))
    out["L2_temp_notemp_fusion"] = {**m, "w_temp": wb, "val_kq": best_w["v"]}
    print(f"  (w_temp={wb:.2f})  κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  val={best_w['v']:.3f}")

    print("\n[4/4] (L1 stage SKIPPED — too slow)")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
