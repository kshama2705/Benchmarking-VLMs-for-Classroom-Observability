"""
N3: IDEP v2 — LEACE concept erasure (Belrose et al. NeurIPS 2023).

LEACE finds the minimum-norm linear transformation that makes a target concept
(here: subject identity) linearly unpredictable from the representation, while
least-disturbing the rest of the representation.

Compare:
  - Baseline (no erasure)
  - Linear residualization (from N2)
  - LEACE (this script)

Output:
  results/idep/idep_leace_results.json
"""
import os, json
import numpy as np
import torch
from concept_erasure import LeaceEraser, LeaceFitter
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_squared_error

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "idep")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)

FEATS_CLIP = os.path.join(BASE, "features", "clip_vitb32_features.npz")
FEATS_DINO = os.path.join(BASE, "features", "dinov2_vitb14_features.npz")


def metrics(yt, yp):
    return {
        "n": int(len(yt)),
        "accuracy": float(accuracy_score(yt, yp)),
        "kappa_quadratic": float(cohen_kappa_score(yt, yp, weights="quadratic")),
        "kappa_linear": float(cohen_kappa_score(yt, yp, weights="linear")),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "mse": float(mean_squared_error(yt, yp)),
        "pred_dist": {int(k): int((yp == k).sum()) for k in range(4)},
    }


def boot_kappa(yt, yp, n=1000):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def engagement_probe(Xtr, ytr, Xva, yva, Xte, yte):
    best = None
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": float(v), "clf": clf}
    yhat_lr = best["clf"].predict(Xte)
    m_lr = metrics(yte, yhat_lr)
    m_lr["kappa_q_ci95"] = boot_kappa(yte, yhat_lr)
    m_lr["best_C"] = best["C"]

    best = None
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(Xtr, ytr.astype(np.float32))
        yhat = np.clip(np.round(reg.predict(Xva)), 0, 3).astype(int)
        v = cohen_kappa_score(yva, yhat, weights="quadratic")
        if best is None or v > best["v"]:
            best = {"a": alpha, "v": float(v), "reg": reg}
    yhat_rd = np.clip(np.round(best["reg"].predict(Xte)), 0, 3).astype(int)
    m_rd = metrics(yte, yhat_rd)
    m_rd["kappa_q_ci95"] = boot_kappa(yte, yhat_rd)
    m_rd["best_alpha"] = best["a"]

    return {"logreg": m_lr, "ridge_ordinal": m_rd}


def post_erasure_id_acc(X_train_erased, subj_train_idx):
    """Sanity check: how recoverable is identity AFTER LEACE?"""
    rng = np.random.default_rng(42)
    # Stratified 80/20 within-train subject-level
    train_idx, test_idx = [], []
    n_subj = subj_train_idx.max() + 1
    for sid in range(n_subj):
        ci = np.where(subj_train_idx == sid)[0]
        rng.shuffle(ci)
        n_t = int(0.8 * len(ci))
        train_idx.extend(ci[:n_t]); test_idx.extend(ci[n_t:])
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)
    clf = LogisticRegression(C=10.0, max_iter=2000, solver="lbfgs", random_state=42)
    clf.fit(X_train_erased[train_idx], subj_train_idx[train_idx])
    return float(clf.score(X_train_erased[test_idx], subj_train_idx[test_idx]))


def run_for_features(label, feats_path):
    print(f"\n========== {label} ==========")
    data = np.load(feats_path, allow_pickle=True)
    splits = data["split"]
    feats = data["feat"].astype(np.float32)
    eng = data["engagement"]
    subj = data["subject_id"]
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    print(f"Train={tr.sum()} Val={va.sum()} Test={te.sum()}")

    # Map subjects to int for LEACE
    unique_subj = sorted(set(subj[tr].tolist()))
    sid_map = {s: i for i, s in enumerate(unique_subj)}
    subj_train_int = np.array([sid_map[s] for s in subj[tr]])

    # === Baseline ===
    print("\n--- Baseline (no erasure) ---")
    base = engagement_probe(feats[tr], eng[tr], feats[va], eng[va], feats[te], eng[te])
    lr, rd = base["logreg"], base["ridge_ordinal"]
    print(f"  LR   : κ_q={lr['kappa_quadratic']:.3f} [{lr['kappa_q_ci95'][0]:.3f},{lr['kappa_q_ci95'][1]:.3f}]")
    print(f"  Ridge: κ_q={rd['kappa_quadratic']:.3f} [{rd['kappa_q_ci95'][0]:.3f},{rd['kappa_q_ci95'][1]:.3f}]")

    # === LEACE ===
    print("\n--- LEACE concept erasure ---")
    Xtr_t = torch.from_numpy(feats[tr]).float()
    n_subj_int = int(subj_train_int.max() + 1)
    # One-hot the subject IDs so LEACE treats it as a 69-dim categorical concept
    z_one_hot = torch.zeros(len(subj_train_int), n_subj_int, dtype=torch.float32)
    z_one_hot[torch.arange(len(subj_train_int)), torch.from_numpy(subj_train_int).long()] = 1.0
    fitter = LeaceFitter.fit(Xtr_t, z_one_hot)
    eraser = fitter.eraser
    # eraser is callable: applies the erasure to a tensor
    def erase(X):
        return eraser(torch.from_numpy(X).float()).numpy()

    feats_erased = erase(feats)
    # Sanity: check ID recoverability after erasure
    id_after = post_erasure_id_acc(feats_erased[tr], subj_train_int)
    print(f"  Subject-ID accuracy after LEACE: {id_after:.3f}")

    leace_res = engagement_probe(feats_erased[tr], eng[tr],
                                  feats_erased[va], eng[va],
                                  feats_erased[te], eng[te])
    lr2, rd2 = leace_res["logreg"], leace_res["ridge_ordinal"]
    print(f"  LR   : κ_q={lr2['kappa_quadratic']:.3f} [{lr2['kappa_q_ci95'][0]:.3f},{lr2['kappa_q_ci95'][1]:.3f}]  "
          f"acc={lr2['accuracy']:.3f}  pred_dist={lr2['pred_dist']}")
    print(f"  Ridge: κ_q={rd2['kappa_quadratic']:.3f} [{rd2['kappa_q_ci95'][0]:.3f},{rd2['kappa_q_ci95'][1]:.3f}]  "
          f"acc={rd2['accuracy']:.3f}  pred_dist={rd2['pred_dist']}")

    return {
        "baseline": base,
        "leace": leace_res,
        "subject_id_after_leace": id_after,
        "n_subjects_train": len(unique_subj),
    }


def main():
    out = {}
    for label, path in [("CLIP_ViT-B32", FEATS_CLIP), ("DINOv2_ViT-B14", FEATS_DINO)]:
        if os.path.exists(path):
            out[label] = run_for_features(label, path)

    out_path = os.path.join(OUT, "idep_leace_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
