"""
EMBER-Full — late-fusion ensemble with all signals:
  - Face (52 blendshapes mean+std+delta = 156, 6 gaze mean+std = 12, 3 pose mean+std = 6, 12 landmark summary = 186)
  - Body pose (16-d posture/shoulder/torso features)
  - Frozen VLM CLS (SigLIP-L, CLIP-L, DINOv2)

We train independent linear probes on each signal group, then combine
their class probabilities with class-aware late fusion learned on val.

Output:
  results/ember/ember_full_results.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPORAL = os.path.join(BASE, "features", "daisee_face_signals_temporal.npz")
POSE = os.path.join(BASE, "features", "daisee_pose_signals.npz")
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CLIPL = os.path.join(BASE, "features", "daisee_clip_l_14_features.npz")
DINO = os.path.join(BASE, "features", "dinov2_vitb14_features.npz")
OUT = os.path.join(BASE, "results", "ember", "ember_full_results.json")
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


def search_weights(probs_va_list, y_va, n_iter=2000, seed=42):
    """Class-aware weight search. probs_va_list: list of (N, 4)."""
    rng = np.random.default_rng(seed)
    n = len(probs_va_list)
    best = None
    for _ in range(n_iter):
        ws = rng.dirichlet(np.ones(n) * 1.5, size=4)  # (4, n)
        p = sum(probs_va_list[j] * ws[None, :, j] for j in range(n))
        v = cohen_kappa_score(y_va, p.argmax(axis=1), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"ws": ws.tolist(), "v": float(v)}
    return best


def search_scalar_weights(probs_va_list, y_va, granularity=21):
    """Simpler scalar-per-probe search via simplex grid."""
    n = len(probs_va_list)
    best = None
    # For 2-3 probes, exhaustive grid
    if n == 2:
        for w in np.linspace(0.0, 1.0, granularity):
            p = w * probs_va_list[0] + (1 - w) * probs_va_list[1]
            v = cohen_kappa_score(y_va, p.argmax(axis=1), weights="quadratic")
            if best is None or v > best["v"]:
                best = {"ws": [float(w), float(1 - w)], "v": float(v)}
    elif n == 3:
        for w1 in np.linspace(0, 1, granularity):
            for w2 in np.linspace(0, 1 - w1, granularity):
                w3 = 1 - w1 - w2
                p = w1 * probs_va_list[0] + w2 * probs_va_list[1] + w3 * probs_va_list[2]
                v = cohen_kappa_score(y_va, p.argmax(axis=1), weights="quadratic")
                if best is None or v > best["v"]:
                    best = {"ws": [float(w1), float(w2), float(w3)], "v": float(v)}
    return best


def fuse_scalar(probs_list, ws):
    return sum(probs_list[j] * ws[j] for j in range(len(probs_list)))


def fuse_caw(probs_list, ws_per_class):
    """ws_per_class is (4, n_probes)."""
    out = np.zeros_like(probs_list[0])
    for j, p in enumerate(probs_list):
        out += p * np.array(ws_per_class)[None, :, j]
    return out


def main():
    print("Loading...")
    tmp = np.load(TEMPORAL, allow_pickle=True)
    clip_ids = tmp["clip_id"]; splits = tmp["split"]
    eng = tmp["engagement"].astype(np.int64)
    face_temporal = np.nan_to_num(np.concatenate([
        tmp["blendshapes_mean"], tmp["blendshapes_std"], tmp["blendshapes_delta"],
        tmp["head_pose_mean"], tmp["head_pose_std"],
        tmp["eye_gaze_mean"], tmp["eye_gaze_std"],
        tmp["landmark_summary_mean"]], axis=1))
    print(f"  face_temporal: {face_temporal.shape}")

    pose_d = np.load(POSE, allow_pickle=True)
    pose_feats = align(clip_ids, pose_d["clip_id"], np.nan_to_num(pose_d["pose_features"]))
    print(f"  body_pose: {pose_feats.shape}")

    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    sc_f = StandardScaler().fit(face_temporal[tr])
    face_temporal_z = sc_f.transform(face_temporal).astype(np.float32)
    sc_p = StandardScaler().fit(pose_feats[tr])
    pose_z = sc_p.transform(pose_feats).astype(np.float32)

    feats = {}
    for label, path in [("SigLIP_L", SIGLIP), ("CLIP_L", CLIPL), ("DINOv2", DINO)]:
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            key = "feat" if "feat" in d.files else "cls_feat"
            feats[label] = align(clip_ids, d["clip_id"], d[key].astype(np.float32))

    print(f"\nFitting base probes...")
    probes = {}
    for name, X in [("face_temporal", face_temporal_z),
                    ("body_pose", pose_z)]:
        clf, val_v, C = fit_lr(X[tr], eng[tr], X[va], eng[va])
        probes[name] = {"va": clf.predict_proba(X[va]),
                        "te": clf.predict_proba(X[te]),
                        "val_kq": val_v, "C": C, "dim": X.shape[1]}
        print(f"  {name:15} val_kq={val_v:.3f} C={C}")
    for label, X in feats.items():
        clf, val_v, C = fit_lr(X[tr], eng[tr], X[va], eng[va])
        probes[label] = {"va": clf.predict_proba(X[va]),
                         "te": clf.predict_proba(X[te]),
                         "val_kq": val_v, "C": C, "dim": X.shape[1]}
        print(f"  {label:15} val_kq={val_v:.3f} C={C}")

    out = {}

    # Solo metrics for each probe
    print("\n=== Solo probes ===")
    for name, P in probes.items():
        yhat = P["te"].argmax(axis=1)
        m = metrics(eng[te], yhat)
        out[f"solo_{name}"] = m
        print(f"  solo_{name:15} κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  val={P['val_kq']:.3f}")

    # ======== Pair fusions: SigLIP_L + each explicit signal ========
    print("\n=== Pair fusions ===")
    cls_label = "SigLIP_L"
    for ex in ["face_temporal", "body_pose"]:
        # Scalar
        best = search_scalar_weights([probes[cls_label]["va"], probes[ex]["va"]], eng[va])
        ws = best["ws"]
        p_te = fuse_scalar([probes[cls_label]["te"], probes[ex]["te"]], ws)
        yhat = p_te.argmax(axis=1)
        tag = f"{cls_label}+{ex}_scalar"
        out[tag] = {**metrics(eng[te], yhat), "weights": ws, "val_kq": best["v"]}
        print(f"  {tag:45} ws={ws}  κ_q={out[tag]['kappa_q']:.3f} {out[tag]['kappa_q_ci']}")

    # ======== Triple: SigLIP_L + face + pose (scalar) ========
    print("\n=== Triple fusions ===")
    probes_3_va = [probes["SigLIP_L"]["va"], probes["face_temporal"]["va"], probes["body_pose"]["va"]]
    probes_3_te = [probes["SigLIP_L"]["te"], probes["face_temporal"]["te"], probes["body_pose"]["te"]]

    best_s = search_scalar_weights(probes_3_va, eng[va], granularity=21)
    p_te = fuse_scalar(probes_3_te, best_s["ws"])
    yhat = p_te.argmax(axis=1)
    out["EMBER_triple_scalar"] = {**metrics(eng[te], yhat),
                                   "weights": best_s["ws"], "val_kq": best_s["v"]}
    print(f"  EMBER_triple_scalar    ws={best_s['ws']}  κ_q={out['EMBER_triple_scalar']['kappa_q']:.3f} {out['EMBER_triple_scalar']['kappa_q_ci']}  val={best_s['v']:.3f}")

    # Class-aware (lighter search budget to avoid overfit)
    for n_iter in [500, 2000]:
        best_c = search_weights(probes_3_va, eng[va], n_iter=n_iter)
        p_te = fuse_caw(probes_3_te, best_c["ws"])
        yhat = p_te.argmax(axis=1)
        tag = f"EMBER_triple_classaware_{n_iter}"
        out[tag] = {**metrics(eng[te], yhat), "weights": best_c["ws"], "val_kq": best_c["v"]}
        print(f"  {tag:30} κ_q={out[tag]['kappa_q']:.3f} {out[tag]['kappa_q_ci']}  val={best_c['v']:.3f}")

    # ======== Quad: SigLIP_L + CLIP_L + face + pose ========
    if "CLIP_L" in probes:
        print("\n=== Quad fusions (SigLIP_L + CLIP_L + face + pose) ===")
        probes_4_va = [probes["SigLIP_L"]["va"], probes["CLIP_L"]["va"],
                       probes["face_temporal"]["va"], probes["body_pose"]["va"]]
        probes_4_te = [probes["SigLIP_L"]["te"], probes["CLIP_L"]["te"],
                       probes["face_temporal"]["te"], probes["body_pose"]["te"]]
        for n_iter in [500, 2000]:
            best_c = search_weights(probes_4_va, eng[va], n_iter=n_iter)
            p_te = fuse_caw(probes_4_te, best_c["ws"])
            yhat = p_te.argmax(axis=1)
            tag = f"EMBER_quad_classaware_{n_iter}"
            out[tag] = {**metrics(eng[te], yhat), "weights": best_c["ws"], "val_kq": best_c["v"]}
            print(f"  {tag:30} κ_q={out[tag]['kappa_q']:.3f} {out[tag]['kappa_q_ci']}  val={best_c['v']:.3f}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
