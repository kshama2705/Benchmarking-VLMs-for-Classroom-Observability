"""
TFIC — Test-time Few-shot Identity Calibration.

Premise: identity dominates frozen features. If we can subtract a subject-
specific "baseline embedding" before probing, the residual should be cleaner
engagement signal.

Setup:
  - At test time, we assume k labeled clips per subject (k ∈ {1, 3, 5, 10}).
  - For each test subject, compute mean feature over the k labeled clips and
    subtract from all test clips of that subject.
  - Train probe on Train, with the same mean-subtraction applied per subject
    (using all train clips of each subject).
  - Compare to the baseline probe with no mean-subtraction.

The "k labeled clips" can be:
  - Random k clips per subject (simulates random calibration)
  - The k engagement-neutral clips (oracle, upper bound)

Output:
  results/ember/tfic_results.json
"""
import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CLIPL = os.path.join(BASE, "features", "daisee_clip_l_14_features.npz")
DINO = os.path.join(BASE, "features", "dinov2_vitb14_features.npz")
OUT = os.path.join(BASE, "results", "ember", "tfic_results.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)


def boot_kq(yt, yp, n=1000):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def subtract_subject_means(X, subj_int, calib_idx_per_subj):
    """Subtract per-subject mean (from calib_idx_per_subj) from each row."""
    out = X.copy().astype(np.float32)
    for sid, idxs in calib_idx_per_subj.items():
        if len(idxs) == 0:
            continue
        mu = X[idxs].mean(axis=0)
        rows = (subj_int == sid)
        out[rows] = X[rows] - mu
    return out


def get_calibration_indices(subj_int, k, seed):
    """For each subject, pick k random indices."""
    rng = np.random.default_rng(seed)
    out = {}
    for sid in np.unique(subj_int):
        rows = np.where(subj_int == sid)[0]
        if len(rows) == 0:
            continue
        chosen = rng.choice(rows, size=min(k, len(rows)), replace=False)
        out[int(sid)] = chosen.tolist()
    return out


def fit_lr_probe(Xtr, ytr, Xva, yva):
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


def run_for_encoder(label, path):
    print(f"\n=== {label} ===")
    d = np.load(path, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    splits = d["split"]; eng = d["engagement"].astype(np.int64); subj = d["subject_id"]
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"

    # Map subjects to ints (separately per split since splits are disjoint)
    def make_subj_ints(subj_arr):
        u = sorted(set(subj_arr.tolist()))
        m = {s: i for i, s in enumerate(u)}
        return np.array([m[s] for s in subj_arr], dtype=np.int64), u

    subj_tr_int, _ = make_subj_ints(subj[tr])
    subj_va_int, _ = make_subj_ints(subj[va])
    subj_te_int, _ = make_subj_ints(subj[te])

    Xtr = X[tr]; ytr = eng[tr]
    Xva = X[va]; yva = eng[va]
    Xte = X[te]; yte = eng[te]

    out = {}

    # Baseline (no calibration)
    clf, val_v, C = fit_lr_probe(Xtr, ytr, Xva, yva)
    yhat = clf.predict(Xte)
    out["baseline"] = {**metrics(yte, yhat), "val_kq": val_v, "C": C}
    print(f"  baseline                κ_q={out['baseline']['kappa_q']:.3f} {out['baseline']['kappa_q_ci']}")

    # TFIC: for each k, subtract per-subject means and re-train
    for k in [1, 3, 5, 10, 20]:
        # For train: use the FULL train clips per subject as calibration (oracle for train)
        calib_tr = {sid: np.where(subj_tr_int == sid)[0].tolist() for sid in np.unique(subj_tr_int)}
        # For val: k random clips per subject
        calib_va = get_calibration_indices(subj_va_int, k, seed=42)
        # For test: k random clips per subject
        calib_te = get_calibration_indices(subj_te_int, k, seed=42)

        Xtr_cal = subtract_subject_means(Xtr, subj_tr_int, calib_tr)
        Xva_cal = subtract_subject_means(Xva, subj_va_int, calib_va)
        Xte_cal = subtract_subject_means(Xte, subj_te_int, calib_te)

        clf, val_v, C = fit_lr_probe(Xtr_cal, ytr, Xva_cal, yva)
        yhat = clf.predict(Xte_cal)
        tag = f"TFIC_k{k}"
        out[tag] = {**metrics(yte, yhat), "val_kq": val_v, "C": C, "k": k}
        print(f"  {tag:20}  κ_q={out[tag]['kappa_q']:.3f} {out[tag]['kappa_q_ci']}  val={val_v:.3f}")

    return out


def main():
    results = {}
    for label, path in [("SigLIP_L", SIGLIP), ("CLIP_L", CLIPL), ("DINOv2", DINO)]:
        if os.path.exists(path):
            results[label] = run_for_encoder(label, path)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
