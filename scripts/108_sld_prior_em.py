"""
MOONSHOT 19: SLD (Saerens-Latinne-Decaestecker) EM prior adjustment.

Iterative EM algorithm to re-estimate test-set class prior under prior shift:
  1. Initial p(y) = train prior
  2. E-step: posteriors p(y|x_test) ∝ p_train(y|x) * p_test(y)/p_train(y)
  3. M-step: p_test(y) = (1/N_test) Σ p(y|x_test)
  4. Iterate to convergence.

Apply this to cached LR-unif probs and threshold-tune the result.

Output:
  results/sota/moonshot_sld.json
"""
import os, json, time
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_sld.json")
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


def sld_em(p_test, train_prior, max_iter=50, eps=1e-7):
    """
    p_test: (N, K) train-trained model's predictions on test
    train_prior: (K,) prior used during training (class proportions)
    Returns: (N, K) adjusted posteriors, K test prior
    """
    K = p_test.shape[1]
    # likelihood ratio: p_train(y=k|x) / p_train(y=k)
    L = p_test / (train_prior[None, :] + eps)
    # Initialize test prior with train prior (or uniform)
    test_prior = train_prior.copy()
    for it in range(max_iter):
        # E-step
        numer = test_prior[None, :] * L
        denom = numer.sum(axis=1, keepdims=True) + eps
        q = numer / denom
        # M-step
        new_prior = q.mean(axis=0)
        if np.linalg.norm(new_prior - test_prior) < 1e-6:
            print(f"  Converged at iter {it}", flush=True)
            break
        test_prior = new_prior
    return q, test_prior


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
    print("Loading cached LR-unif bag + labels...", flush=True)
    c = np.load(CACHE)
    p_te = c["p_te_unif"]; p_va = c["p_va_unif"]
    d = np.load(SIGLIP, allow_pickle=True)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    ytr = eng[sp == "Train"]; yva = eng[sp == "Validation"]; yte = eng[sp == "Test"]

    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    train_prior = np.bincount(ytr, minlength=4).astype(np.float32) / len(ytr)
    val_prior = np.bincount(yva, minlength=4).astype(np.float32) / len(yva)
    test_prior_true = np.bincount(yte, minlength=4).astype(np.float32) / len(yte)
    print(f"  Train prior: {train_prior}", flush=True)
    print(f"  Val prior:   {val_prior}", flush=True)
    print(f"  Test prior:  {test_prior_true}", flush=True)

    out = {}

    # Baseline (no SLD)
    print("\n[A] Baseline (no SLD)...", flush=True)
    m_base = metrics(yte, p_te.argmax(1))
    e_te_base = (p_te * classes[None, :]).sum(1)
    e_va = (p_va * classes[None, :]).sum(1)
    bt = tune_thresh(e_va, yva)
    m_thr_base = metrics(yte, apply_t(e_te_base, bt["t"]))
    out["baseline"] = m_base
    out["baseline_threshold"] = {**m_thr_base, **bt}
    print(f"  Baseline: κ_q={m_base['kappa_q']:.4f}  +thresh: κ_q={m_thr_base['kappa_q']:.4f}", flush=True)

    # SLD with train prior as initialization
    print("\n[B] SLD (init=train prior)...", flush=True)
    p_te_sld, est_prior = sld_em(p_te, train_prior)
    print(f"  Estimated test prior: {est_prior}", flush=True)
    m_sld = metrics(yte, p_te_sld.argmax(1))
    e_te_sld = (p_te_sld * classes[None, :]).sum(1)
    # Tune threshold on val (compute SLD-adjusted val too)
    p_va_sld, _ = sld_em(p_va, train_prior)
    e_va_sld = (p_va_sld * classes[None, :]).sum(1)
    bt_sld = tune_thresh(e_va_sld, yva)
    m_thr_sld = metrics(yte, apply_t(e_te_sld, bt_sld["t"]))
    out["sld_argmax"] = m_sld
    out["sld_threshold"] = {**m_thr_sld, **bt_sld}
    out["estimated_test_prior"] = est_prior.tolist()
    print(f"  SLD argmax: κ_q={m_sld['kappa_q']:.4f}", flush=True)
    print(f"  SLD + thresh: κ_q={m_thr_sld['kappa_q']:.4f} {m_thr_sld['kappa_q_ci']}", flush=True)

    # SLD with VAL prior as initialization (use val to estimate test prior)
    print("\n[C] SLD (init=val prior)...", flush=True)
    p_te_sld2, est_prior2 = sld_em(p_te, val_prior)
    print(f"  Estimated test prior: {est_prior2}", flush=True)
    m_sld2 = metrics(yte, p_te_sld2.argmax(1))
    e_te_sld2 = (p_te_sld2 * classes[None, :]).sum(1)
    bt_sld2 = tune_thresh(e_va_sld, yva)
    m_thr_sld2 = metrics(yte, apply_t(e_te_sld2, bt_sld2["t"]))
    out["sld_valinit_threshold"] = {**m_thr_sld2, **bt_sld2}
    print(f"  SLD valinit + thresh: κ_q={m_thr_sld2['kappa_q']:.4f} {m_thr_sld2['kappa_q_ci']}", flush=True)

    # Oracle: use TRUE test prior (for reference)
    print("\n[D] Oracle (true test prior)...", flush=True)
    # Apply Bayesian adjustment with true prior
    L = p_te / (train_prior[None, :] + 1e-7)
    p_te_oracle = test_prior_true[None, :] * L
    p_te_oracle /= p_te_oracle.sum(axis=1, keepdims=True)
    m_oracle = metrics(yte, p_te_oracle.argmax(1))
    out["oracle_argmax"] = m_oracle
    e_te_oracle = (p_te_oracle * classes[None, :]).sum(1)
    bt_o = tune_thresh(e_va, yva)
    m_thr_o = metrics(yte, apply_t(e_te_oracle, bt_o["t"]))
    out["oracle_threshold"] = {**m_thr_o, **bt_o}
    print(f"  Oracle argmax: κ_q={m_oracle['kappa_q']:.4f}", flush=True)
    print(f"  Oracle + thresh: κ_q={m_thr_o['kappa_q']:.4f} {m_thr_o['kappa_q_ci']}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
