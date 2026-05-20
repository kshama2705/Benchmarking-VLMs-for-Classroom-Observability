"""
EMBER bagging — bootstrap-aggregated ensemble.

For K seeds:
  - Bootstrap-resample Train (with replacement, same size)
  - Fit SigLIP-L LR probe and explicit LR probe on the bootstrap sample
  - Tune late-fusion weight on Val
  - Predict test probabilities
Average K test prediction sets → argmax.

Reduces variance from probe instability + val-weight noise.

Output:
  results/sota/ember_bagging.json
"""
import os, csv, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
SIGNALS = os.path.join(BASE, "features", "daisee_face_signals.npz")
TEMPORAL = os.path.join(BASE, "features", "daisee_face_signals_temporal.npz")
POSE = os.path.join(BASE, "features", "daisee_pose_signals.npz")
OUT = os.path.join(BASE, "results", "sota", "ember_bagging.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)


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


def main():
    print("Loading features...")
    d_sig = np.load(SIGLIP, allow_pickle=True)
    clip_ids = d_sig["clip_id"]; splits = d_sig["split"]
    eng = d_sig["engagement"].astype(np.int64)
    sig_feats = d_sig["feat"].astype(np.float32)

    d_sf = np.load(SIGNALS, allow_pickle=True)
    sf_explicit_raw = np.nan_to_num(np.concatenate([
        d_sf["blendshapes"], d_sf["head_pose"], d_sf["eye_gaze"], d_sf["landmark_summary"]], axis=1))
    explicit = align(clip_ids, d_sf["clip_id"], sf_explicit_raw)

    d_tmp = np.load(TEMPORAL, allow_pickle=True)
    temporal_raw = np.nan_to_num(np.concatenate([
        d_tmp["blendshapes_mean"], d_tmp["blendshapes_std"], d_tmp["blendshapes_delta"],
        d_tmp["head_pose_mean"], d_tmp["head_pose_std"],
        d_tmp["eye_gaze_mean"], d_tmp["eye_gaze_std"],
        d_tmp["landmark_summary_mean"]], axis=1))
    temporal = align(clip_ids, d_tmp["clip_id"], temporal_raw)

    d_pose = np.load(POSE, allow_pickle=True)
    pose_raw = align(clip_ids, d_pose["clip_id"], np.nan_to_num(d_pose["pose_features"]))

    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    yte = eng[te]

    # Z-score explicit/pose/temporal using Train
    sc_e = StandardScaler().fit(explicit[tr]); explicit_z = sc_e.transform(explicit).astype(np.float32)
    sc_t = StandardScaler().fit(temporal[tr]); temporal_z = sc_t.transform(temporal).astype(np.float32)
    sc_p = StandardScaler().fit(pose_raw[tr]); pose_z = sc_p.transform(pose_raw).astype(np.float32)

    feat_groups = {
        "SigLIP_L": sig_feats,
        "single_explicit": explicit_z,
        "temporal_explicit": temporal_z,
        "body_pose": pose_z,
    }

    # Baseline (no bagging)
    print("\n=== Baseline (no bagging) ===")
    out = {}
    clf_sig, val_v, C = fit_lr(sig_feats[tr], eng[tr], sig_feats[va], eng[va])
    yhat = clf_sig.predict(sig_feats[te])
    m_base = metrics(yte, yhat)
    print(f"  SigLIP_L solo  κ_q={m_base['kappa_q']:.3f} {m_base['kappa_q_ci']}")
    p_sig_te = clf_sig.predict_proba(sig_feats[te])

    # EMBER baseline (single shot with single_explicit, w=0.35 fixed)
    clf_e, _, _ = fit_lr(explicit_z[tr], eng[tr], explicit_z[va], eng[va])
    p_e_va = clf_e.predict_proba(explicit_z[va])
    p_e_te = clf_e.predict_proba(explicit_z[te])
    # Tune w on val
    p_sig_va = clf_sig.predict_proba(sig_feats[va])
    best_w = None
    for w in np.linspace(0.0, 1.0, 41):
        pv = w * p_e_va + (1 - w) * p_sig_va
        v = cohen_kappa_score(eng[va], pv.argmax(axis=1), weights="quadratic")
        if best_w is None or v > best_w["v"]:
            best_w = {"w": float(w), "v": float(v)}
    wb = best_w["w"]
    p_te_baseline = wb * p_e_te + (1 - wb) * p_sig_te
    yhat_base = p_te_baseline.argmax(axis=1)
    m_ember = metrics(yte, yhat_base)
    print(f"  EMBER baseline (w={wb:.2f})  κ_q={m_ember['kappa_q']:.3f} {m_ember['kappa_q_ci']}  val={best_w['v']:.3f}")
    out["baseline_SigLIP_L"] = m_base
    out["baseline_EMBER"] = {**m_ember, "w": wb}

    # Bagging — bootstrap train K times
    print("\n=== EMBER bagging (K seeds, bootstrap resample of Train) ===")
    K_VALUES = [5, 10, 20]
    train_idx_full = np.where(tr)[0]
    n_tr = len(train_idx_full)
    sig_tr = sig_feats[tr]; eng_tr = eng[tr]
    exp_tr = explicit_z[tr]
    tmp_tr = temporal_z[tr]
    pose_tr_z = pose_z[tr]

    p_sig_va_all = []  # K predictions
    p_sig_te_all = []
    p_e_va_all = []
    p_e_te_all = []
    p_tmp_va_all = []
    p_tmp_te_all = []
    p_pose_va_all = []
    p_pose_te_all = []
    K = max(K_VALUES)
    rng = np.random.default_rng(0)
    for k in range(K):
        idx = rng.integers(0, n_tr, size=n_tr)
        # SigLIP-L probe
        clf_s, _, _ = fit_lr(sig_tr[idx], eng_tr[idx], sig_feats[va], eng[va])
        # explicit probe
        clf_e, _, _ = fit_lr(exp_tr[idx], eng_tr[idx], explicit_z[va], eng[va])
        # temporal probe
        clf_t, _, _ = fit_lr(tmp_tr[idx], eng_tr[idx], temporal_z[va], eng[va])
        # pose probe
        clf_p, _, _ = fit_lr(pose_tr_z[idx], eng_tr[idx], pose_z[va], eng[va])
        p_sig_va_all.append(clf_s.predict_proba(sig_feats[va]))
        p_sig_te_all.append(clf_s.predict_proba(sig_feats[te]))
        p_e_va_all.append(clf_e.predict_proba(explicit_z[va]))
        p_e_te_all.append(clf_e.predict_proba(explicit_z[te]))
        p_tmp_va_all.append(clf_t.predict_proba(temporal_z[va]))
        p_tmp_te_all.append(clf_t.predict_proba(temporal_z[te]))
        p_pose_va_all.append(clf_p.predict_proba(pose_z[va]))
        p_pose_te_all.append(clf_p.predict_proba(pose_z[te]))
        if (k + 1) % 5 == 0:
            print(f"  bag {k+1}/{K} done")

    p_sig_va_stack = np.stack(p_sig_va_all)
    p_sig_te_stack = np.stack(p_sig_te_all)
    p_e_va_stack = np.stack(p_e_va_all)
    p_e_te_stack = np.stack(p_e_te_all)

    for K_use in K_VALUES:
        # Average first K_use predictions
        p_sig_va_mean = p_sig_va_stack[:K_use].mean(axis=0)
        p_sig_te_mean = p_sig_te_stack[:K_use].mean(axis=0)
        p_e_va_mean = p_e_va_stack[:K_use].mean(axis=0)
        p_e_te_mean = p_e_te_stack[:K_use].mean(axis=0)

        # Bagged SigLIP_L solo
        yhat_solo = p_sig_te_mean.argmax(axis=1)
        m_solo = metrics(yte, yhat_solo)
        print(f"  K={K_use}  bagged_SigLIP_L  κ_q={m_solo['kappa_q']:.3f} {m_solo['kappa_q_ci']}")
        out[f"K{K_use}_bagged_SigLIP_L"] = m_solo

        # EMBER bagged with scalar w
        best_w = None
        for w in np.linspace(0.0, 1.0, 41):
            pv = w * p_e_va_mean + (1 - w) * p_sig_va_mean
            v = cohen_kappa_score(eng[va], pv.argmax(axis=1), weights="quadratic")
            if best_w is None or v > best_w["v"]:
                best_w = {"w": float(w), "v": float(v)}
        wb = best_w["w"]
        p_te = wb * p_e_te_mean + (1 - wb) * p_sig_te_mean
        yhat = p_te.argmax(axis=1)
        m = metrics(yte, yhat)
        print(f"  K={K_use}  bagged_EMBER (w={wb:.2f})  κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  val={best_w['v']:.3f}")
        out[f"K{K_use}_bagged_EMBER"] = {**m, "w": wb}

        # Multi-probe bagged ensemble (sig + e + tmp + pose)
        p_tmp_va_mean = np.stack(p_tmp_va_all[:K_use]).mean(axis=0)
        p_tmp_te_mean = np.stack(p_tmp_te_all[:K_use]).mean(axis=0)
        p_pose_va_mean = np.stack(p_pose_va_all[:K_use]).mean(axis=0)
        p_pose_te_mean = np.stack(p_pose_te_all[:K_use]).mean(axis=0)

        # Sweep 4-weight simplex (random search)
        probs_va_list = [p_sig_va_mean, p_e_va_mean, p_tmp_va_mean, p_pose_va_mean]
        probs_te_list = [p_sig_te_mean, p_e_te_mean, p_tmp_te_mean, p_pose_te_mean]
        best_m = None
        rng2 = np.random.default_rng(42)
        for _ in range(3000):
            ws = rng2.dirichlet(np.ones(4) * 1.5)
            pv = sum(probs_va_list[j] * ws[j] for j in range(4))
            v = cohen_kappa_score(eng[va], pv.argmax(axis=1), weights="quadratic")
            if best_m is None or v > best_m["v"]:
                best_m = {"ws": ws.tolist(), "v": float(v)}
        ws_v = np.array(best_m["ws"])
        p_te = sum(probs_te_list[j] * ws_v[j] for j in range(4))
        yhat = p_te.argmax(axis=1)
        m = metrics(yte, yhat)
        out[f"K{K_use}_bagged_multi"] = {**m, "weights": best_m["ws"], "val_kq": best_m["v"]}
        print(f"  K={K_use}  bagged_multi (4-probe)   κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  "
              f"ws={[f'{w:.2f}' for w in best_m['ws']]}  val={best_m['v']:.3f}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
