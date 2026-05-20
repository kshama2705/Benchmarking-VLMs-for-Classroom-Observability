"""
DREAM diagnostics — within-subject ranking + identity-erasure check.

For each (modulation, K, protocol) variant we test:
  (a) Subject-ID recovery on residual features:  if frozen features encode
      identity at 99.5%, do DREAM-subtract features still do? If DREAM is
      doing anything, we expect ID accuracy to drop. The more it drops at
      a given engagement κ, the better.
  (b) Within-subject Spearman ρ between predicted E[y] and true engagement.
      Baseline: 0.05 (frozen features only predict subject priors).
      Target:  > 0.15 (within-subject ranking emerges).

Reads: features/daisee_siglip_l_features.npz, features/dream_anchors.npz
Writes: results/dream/dream_diagnostics.json
"""
import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
ANCHORS = os.path.join(BASE, "features", "dream_anchors.npz")
OUT = os.path.join(BASE, "results", "dream", "dream_diagnostics.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def within_subject_split(subject, rng_seed=42):
    """Within-subject 80/20 split for subject-ID probe."""
    rng = np.random.default_rng(rng_seed)
    tr = np.zeros(len(subject), dtype=bool)
    for s in np.unique(subject):
        idx = np.where(subject == s)[0]
        rng.shuffle(idx)
        cut = max(1, int(0.8 * len(idx)))
        tr[idx[:cut]] = True
    te = ~tr
    return tr, te


def subj_id_acc(X, subj):
    """In-subject classifier acc (80/20 within subject)."""
    tr, te = within_subject_split(subj)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                             multi_class="multinomial", n_jobs=1)
    # Encode subject as int labels
    uniq = np.unique(subj)
    s2i = {s: i for i, s in enumerate(uniq)}
    y = np.array([s2i[s] for s in subj])
    clf.fit(X[tr], y[tr])
    return float(accuracy_score(y[te], clf.predict(X[te])))


def within_subject_spearman(yhat_score, ytrue, subj):
    """Spearman ρ averaged across subjects (only subjects with >=3 clips)."""
    rs = []
    n_used = 0
    for s in np.unique(subj):
        mask = (subj == s)
        if mask.sum() < 3 or len(np.unique(ytrue[mask])) < 2:
            continue
        r, _ = spearmanr(yhat_score[mask], ytrue[mask])
        if r is not None and not np.isnan(r):
            rs.append(r)
            n_used += 1
    return float(np.mean(rs)) if rs else float('nan'), n_used


def main():
    d = np.load(SIGLIP, allow_pickle=True)
    feat = d['feat'].astype(np.float32)
    split = d['split']; subject = d['subject_id']; eng = d['engagement'].astype(np.int64)
    a = np.load(ANCHORS, allow_pickle=True)
    subjects = a['subject_ids']

    tr_mask = (split == 'Train'); va_mask = (split == 'Validation'); te_mask = (split == 'Test')
    Xtr_raw = feat[tr_mask]; ytr = eng[tr_mask]
    Xva_raw = feat[va_mask]; yva = eng[va_mask]
    Xte_raw = feat[te_mask]; yte = eng[te_mask]
    s_te = subject[te_mask]

    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    results = {}

    def diagnose(Xtr, Xva, Xte, name):
        # Engagement probe (LR with C-sweep)
        best = None
        for C in (0.001, 0.01, 0.1, 1.0, 10.0):
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                                     solver="lbfgs", random_state=42)
            clf.fit(Xtr, ytr)
            v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
            if best is None or v > best["v"]:
                best = {"C": C, "v": float(v), "clf": clf}
        clf = best["clf"]
        p_te = clf.predict_proba(Xte); e_te = (p_te * classes[None]).sum(1)
        kq = float(cohen_kappa_score(yte, p_te.argmax(1), weights="quadratic"))

        # Identity probe (on TRAIN features only — to avoid leak)
        id_acc = subj_id_acc(Xtr, subject[tr_mask])

        # Within-subject Spearman on test set
        ws_r, ws_n = within_subject_spearman(e_te, yte, s_te)

        print(f"[{name}] kq={kq:.4f}  id_acc={id_acc:.3f}  ws_spearman={ws_r:.4f} (n={ws_n})", flush=True)
        return {"engagement_kq_argmax": kq, "subject_id_acc": id_acc,
                "within_subject_spearman_mean": ws_r, "n_subjects_used": ws_n,
                "best_C": best["C"], "val_kq": best["v"]}

    print("=== BASELINE (raw) ===", flush=True)
    results["baseline_raw"] = diagnose(Xtr_raw, Xva_raw, Xte_raw, "baseline_raw")

    for K in [1, 3, 5, 10]:
        for proto in ["p_train"]:
            idx_key = f"{proto}_anchor_idx_k{K}"
            if idx_key not in a:
                continue
            anchor_idx = a[idx_key]
            anchor_means = {}
            for si, s in enumerate(subjects):
                idx = anchor_idx[si]
                idx = idx[idx >= 0]
                anchor_means[s] = feat[idx].mean(0) if len(idx) > 0 else np.zeros(feat.shape[1], np.float32)
            a_full = np.stack([anchor_means[s] for s in subject], axis=0)
            a_tr = a_full[tr_mask]; a_va = a_full[va_mask]; a_te = a_full[te_mask]

            Xtr = Xtr_raw - a_tr; Xva = Xva_raw - a_va; Xte = Xte_raw - a_te
            results[f"sub_{proto}_K{K}"] = diagnose(Xtr, Xva, Xte, f"sub_{proto}_K{K}")

    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
