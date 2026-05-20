"""
MOONSHOT 20: Per-subject test-time calibration.

Memory insight: within-subject Spearman ≈ 0.05 → model can't rank a subject's
own clips, but DOES capture subject-level baselines. So we should classify
at the subject level, not per-clip.

Approach: for each test subject:
  1. Compute mean E[y] across all their test clips
  2. Apply global thresholds to that mean → subject baseline class
  3. Assign ALL their clips to that subject baseline class (or shift per-clip
     predictions by their distance to subject mean)

Variants:
  A) Hard: all clips of subject S → predicted subject baseline class
  B) Soft: per-clip E[y] shifted by (subject_mean - global_mean) bias
  C) Per-subject thresholds: tune thresholds per subject based on their mean

Output:
  results/sota/moonshot_per_subject.json
"""
import os, json, time
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_per_subject.json")
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


def main():
    print("Loading...", flush=True)
    d = np.load(SIGLIP, allow_pickle=True)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    subj = d["subject_id"]
    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    yva = eng[va]; yte = eng[te]
    subj_va = subj[va]; subj_te = subj[te]

    c = np.load(CACHE)
    p_te = c["p_te_unif"]; p_va = c["p_va_unif"]
    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    e_va = (p_va * classes[None, :]).sum(1)
    e_te = (p_te * classes[None, :]).sum(1)

    # Baseline: global thresholds
    bt = tune_thresh(e_va, yva)
    m_base = metrics(yte, apply_t(e_te, bt["t"]))
    print(f"\nBaseline (global thresh): κ_q={m_base['kappa_q']:.4f} {m_base['kappa_q_ci']}", flush=True)

    out = {"baseline_global_threshold": {**m_base, **bt}}

    # Subject-mean stats
    unique_te_subj = np.unique(subj_te)
    print(f"\nTest subjects: {len(unique_te_subj)}", flush=True)
    subj_mean_te = {}
    subj_n_clips = {}
    for s in unique_te_subj:
        mask = subj_te == s
        subj_mean_te[s] = e_te[mask].mean()
        subj_n_clips[s] = mask.sum()
    n_clips_arr = np.array(list(subj_n_clips.values()))
    print(f"  Clips per subject: min={n_clips_arr.min()}, max={n_clips_arr.max()}, mean={n_clips_arr.mean():.1f}", flush=True)

    # ============ Variant A: Subject baseline assignment ============
    print("\n[A] Hard: all clips → subject mean → global thresholds applied to mean", flush=True)
    yhat_a = np.zeros_like(yte)
    for s in unique_te_subj:
        mask = subj_te == s
        subj_e = subj_mean_te[s]
        # Apply global thresholds to subject mean
        c_pred = 0
        if subj_e > bt["t"][0]: c_pred = 1
        if subj_e > bt["t"][1]: c_pred = 2
        if subj_e > bt["t"][2]: c_pred = 3
        yhat_a[mask] = c_pred
    m_a = metrics(yte, yhat_a)
    out["hard_subject_baseline"] = m_a
    print(f"  κ_q={m_a['kappa_q']:.4f} {m_a['kappa_q_ci']}", flush=True)

    # ============ Variant B: Subject-bias shift ============
    print("\n[B] Soft: subject mean shift bias added to per-clip E[y]", flush=True)
    # Estimate "global mean" from val (true labels) — better estimate
    e_va_mean = e_va.mean()
    # Per-subject mean → bias = subject_mean - global_mean
    # Shifted per-clip e: subtract bias (or add — try both)
    # Actually: if subject's mean is much higher than global mean, the subject's
    # clips are SYSTEMATICALLY over-predicted (because subject baseline is high
    # while clips vary little). Subtract per-subject bias to recalibrate.
    e_te_shifted = e_te.copy()
    for s in unique_te_subj:
        mask = subj_te == s
        bias = subj_mean_te[s] - e_va_mean
        e_te_shifted[mask] = e_te[mask] - bias  # subtract bias from each clip
    # Apply global thresholds
    yhat_b = apply_t(e_te_shifted, bt["t"])
    m_b = metrics(yte, yhat_b)
    out["soft_subject_bias_subtract"] = m_b
    print(f"  Subtract bias: κ_q={m_b['kappa_q']:.4f} {m_b['kappa_q_ci']}", flush=True)

    # Variant B': add bias
    e_te_shifted2 = e_te.copy()
    for s in unique_te_subj:
        mask = subj_te == s
        bias = subj_mean_te[s] - e_va_mean
        e_te_shifted2[mask] = e_te[mask] + bias
    yhat_b2 = apply_t(e_te_shifted2, bt["t"])
    m_b2 = metrics(yte, yhat_b2)
    out["soft_subject_bias_add"] = m_b2
    print(f"  Add bias: κ_q={m_b2['kappa_q']:.4f} {m_b2['kappa_q_ci']}", flush=True)

    # ============ Variant C: Per-subject thresholds ============
    # For each subject, shift thresholds by (subject_mean - global_mean)
    print("\n[C] Per-subject threshold shift", flush=True)
    yhat_c = np.zeros_like(yte)
    for s in unique_te_subj:
        mask = subj_te == s
        shift = subj_mean_te[s] - e_va_mean
        # Shift thresholds upward if subject mean > global; downward otherwise
        t_subj = [bt["t"][0] + shift, bt["t"][1] + shift, bt["t"][2] + shift]
        yhat_c[mask] = apply_t(e_te[mask], t_subj)
    m_c = metrics(yte, yhat_c)
    out["per_subject_threshold_shift"] = m_c
    print(f"  κ_q={m_c['kappa_q']:.4f} {m_c['kappa_q_ci']}", flush=True)

    # ============ Variant D: Per-subject re-tuned thresholds (using val analogously) ============
    # Estimate global mean from val (avg of clip-level E[y])
    # For each subject's clips: subtract (subject_mean - val_mean), then apply global thresholds
    # Same as Variant B (subtract bias). Confirmed.

    # ============ Variant E: Mix hard subject baseline + global threshold ============
    # Weighted average: alpha * subject_class + (1-alpha) * clip_class
    # For now, just test alpha=0.0 (all clip), 0.5 (mix), 1.0 (all subject)
    print("\n[E] Combine subject-baseline + per-clip predictions", flush=True)
    yhat_base = apply_t(e_te, bt["t"])
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # Per-clip prediction: yhat_base. Subject prediction: yhat_a.
        # alpha * yhat_a + (1-alpha) * yhat_base — round
        mixed = alpha * yhat_a + (1 - alpha) * yhat_base
        yhat_mix = np.round(mixed).astype(int)
        m_mix = metrics(yte, yhat_mix)
        out[f"mix_subject_{alpha}"] = m_mix
        print(f"  alpha={alpha}: κ_q={m_mix['kappa_q']:.4f} {m_mix['kappa_q_ci']}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
