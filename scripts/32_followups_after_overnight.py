"""
Follow-up experiments using overnight features. All use cached .npz files,
no re-encoding.

Stages:
  F1. Subject-ID probe on SigLIP-L and CLIP-L
  F2. Bootstrap CIs on tier-2 linear probes (proper test-set bootstrap)
  F3. Entanglement curve (k-sweep) on SigLIP-L
  F4. SIEP-contrastive multi-seed on SigLIP-L

Output:
  results/overnight/followups.json
"""
import os, json, time, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_squared_error

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "overnight")
STATUS = os.path.join(OUT, "followups_status.txt")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))


def status(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def boot_kq(yt, yp, n=1000):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    splits = d["split"]; feats = d["feat"].astype(np.float32)
    eng = d["engagement"]; subj = d["subject_id"]
    return splits, feats, eng, subj


def f1_subject_id_probe(label, path):
    """N1-style subject-ID probe (within-Train clip-level 80/20 split)."""
    status(f"F1: subject-ID probe on {label}")
    splits, feats, eng, subj = load_npz(path)
    tr = splits == "Train"
    Xtr = feats[tr]; subj_tr = subj[tr]
    unique = sorted(set(subj_tr.tolist()))
    sid_map = {s: i for i, s in enumerate(unique)}
    y = np.array([sid_map[s] for s in subj_tr])
    rng = np.random.default_rng(42)
    train_idx, test_idx = [], []
    for sid in range(len(unique)):
        ci = np.where(y == sid)[0]; rng.shuffle(ci)
        n_t = int(0.8 * len(ci))
        train_idx.extend(ci[:n_t]); test_idx.extend(ci[n_t:])
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)
    clf = LogisticRegression(C=10.0, max_iter=5000, solver="lbfgs", random_state=42, n_jobs=-1)
    clf.fit(Xtr[train_idx], y[train_idx])
    train_acc = clf.score(Xtr[train_idx], y[train_idx])
    test_acc = clf.score(Xtr[test_idx], y[test_idx])
    status(f"  {label}: n_subj={len(unique)}  train_acc={train_acc:.3f}  held-out_acc={test_acc:.3f}")
    return {"n_subjects": len(unique), "train_acc": float(train_acc), "test_acc": float(test_acc)}


def f2_lp_with_ci(label, path):
    """Linear probes with proper bootstrap CIs."""
    status(f"F2: linear probe + bootstrap CI on {label}")
    splits, feats, eng, subj = load_npz(path)
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    Xtr, ytr = feats[tr], eng[tr]; Xva, yva = feats[va], eng[va]; Xte, yte = feats[te], eng[te]

    # LogReg
    best = None
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                                 solver="lbfgs", random_state=42, n_jobs=-1)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": float(v), "clf": clf}
    yhat = best["clf"].predict(Xte)
    m_lr = {
        "best_C": best["C"],
        "kappa_q": float(cohen_kappa_score(yte, yhat, weights="quadratic")),
        "kappa_q_ci95": boot_kq(yte, yhat),
        "accuracy": float(accuracy_score(yte, yhat)),
        "f1_macro": float(f1_score(yte, yhat, average="macro", zero_division=0)),
        "pred_dist": {int(k): int((yhat == k).sum()) for k in range(4)},
    }
    # Ridge ordinal
    best = None
    for alpha in [1.0, 10.0, 100.0, 1000.0]:
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(Xtr, ytr.astype(np.float32))
        v = cohen_kappa_score(yva, np.clip(np.round(reg.predict(Xva)), 0, 3).astype(int), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"a": alpha, "v": float(v), "reg": reg}
    yhat = np.clip(np.round(best["reg"].predict(Xte)), 0, 3).astype(int)
    m_rd = {
        "best_alpha": best["a"],
        "kappa_q": float(cohen_kappa_score(yte, yhat, weights="quadratic")),
        "kappa_q_ci95": boot_kq(yte, yhat),
        "accuracy": float(accuracy_score(yte, yhat)),
        "f1_macro": float(f1_score(yte, yhat, average="macro", zero_division=0)),
        "pred_dist": {int(k): int((yhat == k).sum()) for k in range(4)},
    }
    status(f"  {label}: LR κ_q={m_lr['kappa_q']:.3f} {m_lr['kappa_q_ci95']}  "
           f"Ridge κ_q={m_rd['kappa_q']:.3f} {m_rd['kappa_q_ci95']}")
    return {"logreg": m_lr, "ridge_ordinal": m_rd}


def f3_entanglement_curve(label, path, ks=(0, 1, 2, 3, 5, 10, 20, 50, 69)):
    """Sweep k for top-k identity removal; report id_acc and engagement κ at each k."""
    status(f"F3: entanglement curve on {label}")
    splits, feats, eng, subj = load_npz(path)
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    unique = sorted(set(subj[tr].tolist()))
    sid_map = {s: i for i, s in enumerate(unique)}
    y_subj = np.array([sid_map[s] for s in subj[tr]])
    id_clf = LogisticRegression(C=10.0, max_iter=5000, solver="lbfgs", random_state=42, n_jobs=-1)
    id_clf.fit(feats[tr], y_subj)
    W = id_clf.coef_  # (n_subj, D)
    U_full, S_full, Vt_full = np.linalg.svd(W.T, full_matrices=False)

    rows = []
    for k in ks:
        if k == 0:
            P = np.eye(W.shape[1])
        else:
            U_k = U_full[:, :min(k, U_full.shape[1])]
            P = np.eye(W.shape[1]) - U_k @ U_k.T
        Ftr = feats[tr] @ P; Fva = feats[va] @ P; Fte = feats[te] @ P
        # ID acc on within-train 80/20
        rng = np.random.default_rng(42)
        train_idx, test_idx = [], []
        for sid in range(len(unique)):
            ci = np.where(y_subj == sid)[0]; rng.shuffle(ci)
            n_t = int(0.8 * len(ci))
            train_idx.extend(ci[:n_t]); test_idx.extend(ci[n_t:])
        clf2 = LogisticRegression(C=10.0, max_iter=2000, solver="lbfgs", random_state=42, n_jobs=-1)
        clf2.fit(Ftr[np.array(train_idx)], y_subj[np.array(train_idx)])
        id_acc = float(clf2.score(Ftr[np.array(test_idx)], y_subj[np.array(test_idx)]))
        # Engagement LR
        best = None
        for C in [0.01, 0.1, 1.0, 10.0]:
            clf3 = LogisticRegression(C=C, class_weight="balanced", max_iter=3000,
                                      solver="lbfgs", random_state=42, n_jobs=-1)
            clf3.fit(Ftr, eng[tr])
            v = cohen_kappa_score(eng[va], clf3.predict(Fva), weights="quadratic")
            if best is None or v > best["v"]:
                best = {"C": C, "v": v, "clf": clf3}
        yhat = best["clf"].predict(Fte)
        kq = float(cohen_kappa_score(eng[te], yhat, weights="quadratic"))
        ci = boot_kq(eng[te], yhat)
        rows.append({"k": k, "id_acc": id_acc, "kq": kq, "ci": ci})
        status(f"  k={k:3d}  id_acc={id_acc:.3f}  κ_q={kq:.3f} [{ci[0]:.3f},{ci[1]:.3f}]")
    return rows


def main():
    out = {}
    encoders = [
        ("SigLIP-L", os.path.join(BASE, "features", "daisee_siglip_l_features.npz")),
        ("CLIP-L/14", os.path.join(BASE, "features", "daisee_clip_l_14_features.npz")),
    ]

    out["subject_id_probe"] = {}
    out["linear_probe_with_ci"] = {}
    for label, path in encoders:
        if not os.path.exists(path):
            status(f"Missing features file: {path}")
            continue
        try:
            out["subject_id_probe"][label] = f1_subject_id_probe(label, path)
        except Exception as e:
            status(f"F1 {label} FAILED: {e}")
            status(traceback.format_exc())
        try:
            out["linear_probe_with_ci"][label] = f2_lp_with_ci(label, path)
        except Exception as e:
            status(f"F2 {label} FAILED: {e}")
            status(traceback.format_exc())

    # F3 only on SigLIP-L (the headline encoder)
    try:
        out["entanglement_curve_siglip_l"] = f3_entanglement_curve(
            "SigLIP-L", encoders[0][1])
    except Exception as e:
        status(f"F3 FAILED: {e}")
        status(traceback.format_exc())

    out_path = os.path.join(OUT, "followups.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    status(f"Saved: {out_path}")
    status("FOLLOWUPS_DONE")


if __name__ == "__main__":
    main()
