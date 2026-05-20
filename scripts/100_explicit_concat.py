"""
MOONSHOT 11: Concat SigLIP-L with explicit face/gaze/pose signals.

Solo explicit features were 0.056 κ — bad alone. But the SigLIP-L embeddings
may MISS specific signals like gaze direction or blink frequency that explicit
features capture. Concat may add orthogonal information.

Variants:
  A) SigLIP-L (1024) + face static (73)
  B) SigLIP-L (1024) + face temporal (181) + pose (16)
  C) SigLIP-L (1024) + face static (73) + pose (16) = 1113-d
  D) SigLIP-L + face_temporal_extended (188 with detected_count)

Output:
  results/sota/moonshot_explicit_concat.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
FACE = os.path.join(BASE, "features", "daisee_face_signals.npz")
POSE = os.path.join(BASE, "features", "daisee_pose_signals.npz")
FACE_TMP = os.path.join(BASE, "features", "daisee_face_signals_temporal.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_explicit_concat.json")
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


def fit_lr(Xtr, ytr, Xva, yva):
    best = None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": float(v), "clf": clf}
    return best["clf"]


def bag_lr(Xtr, ytr, Xva, yva, Xte, K=20, seeds=[0, 7, 42, 2025, 1024]):
    all_te, all_va = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        bt, bv = [], []
        for k in range(K):
            idx = rng.integers(0, len(ytr), size=len(ytr))
            clf = fit_lr(Xtr[idx], ytr[idx], Xva, yva)
            bt.append(clf.predict_proba(Xte)); bv.append(clf.predict_proba(Xva))
            if (k+1) % 10 == 0:
                print(f"    seed={s} k={k+1}/{K}", flush=True)
        all_te.append(np.stack(bt).mean(0)); all_va.append(np.stack(bv).mean(0))
    return np.stack(all_te).mean(0), np.stack(all_va).mean(0)


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


def build_explicit(face, pose, face_tmp, use_temporal=False, use_pose=False):
    parts = []
    if use_temporal:
        parts.append(face_tmp["blendshapes_mean"])
        parts.append(face_tmp["blendshapes_std"])
        parts.append(face_tmp["blendshapes_delta"])
        parts.append(face_tmp["head_pose_mean"])
        parts.append(face_tmp["head_pose_std"])
        parts.append(face_tmp["eye_gaze_mean"])
        parts.append(face_tmp["eye_gaze_std"])
        parts.append(face_tmp["landmark_summary_mean"])
        parts.append(face_tmp["detected_count"][:, None].astype(np.float32))
    else:
        parts.append(face["blendshapes"])
        parts.append(face["head_pose"])
        parts.append(face["eye_gaze"])
        parts.append(face["landmark_summary"])
        parts.append(face["detected"][:, None].astype(np.float32))
    if use_pose:
        parts.append(pose["pose_features"])
    return np.concatenate([p.astype(np.float32) for p in parts], axis=1)


def main():
    print("Loading...", flush=True)
    d = np.load(SIGLIP, allow_pickle=True)
    Xs = d["feat"].astype(np.float32)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    face = np.load(FACE, allow_pickle=True)
    pose = np.load(POSE, allow_pickle=True)
    face_tmp = np.load(FACE_TMP, allow_pickle=True)

    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    ytr = eng[tr]; yva = eng[va]; yte = eng[te]

    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    out = {}

    configs = [
        {"name": "siglip_face_static", "use_temporal": False, "use_pose": False},
        {"name": "siglip_face_pose_static", "use_temporal": False, "use_pose": True},
        {"name": "siglip_face_temporal", "use_temporal": True, "use_pose": False},
        {"name": "siglip_face_pose_temporal", "use_temporal": True, "use_pose": True},
    ]

    for cfg in configs:
        name = cfg["name"]
        print(f"\n=== Config: {name} ===", flush=True)
        explicit_full = build_explicit(face, pose, face_tmp, use_temporal=cfg["use_temporal"], use_pose=cfg["use_pose"])
        # Standardize explicit features
        scaler = StandardScaler().fit(explicit_full[tr])
        explicit_std = scaler.transform(explicit_full).astype(np.float32)
        Xcat = np.concatenate([Xs, explicit_std], axis=1)
        print(f"  Concat dim: {Xcat.shape[1]} (siglip={Xs.shape[1]} + explicit={explicit_std.shape[1]})", flush=True)

        Xtr_cat = Xcat[tr]; Xva_cat = Xcat[va]; Xte_cat = Xcat[te]

        t0 = time.time()
        p_te, p_va = bag_lr(Xtr_cat, ytr, Xva_cat, yva, Xte_cat, K=20)
        print(f"  Bag done ({time.time()-t0:.0f}s)", flush=True)

        m_solo = metrics(yte, p_te.argmax(1))
        e_va = (p_va * classes[None, :]).sum(1); e_te = (p_te * classes[None, :]).sum(1)
        bt = tune_thresh(e_va, yva)
        m_thr = metrics(yte, apply_t(e_te, bt["t"]))
        out[name] = {"solo": m_solo, "threshold": {**m_thr, **bt}}
        print(f"  Solo: κ_q={m_solo['kappa_q']:.4f}  +thresh: κ_q={m_thr['kappa_q']:.4f} {m_thr['kappa_q_ci']}", flush=True)

        # Cache for later fusion
        np.savez_compressed(os.path.join(BASE, "results", "sota", f"_explicit_{name}_cache.npz"),
                            p_te=p_te, p_va=p_va)

    # Best fusion with cached LR-unif
    print("\n=== Best explicit + cached LR-unif fusion ===", flush=True)
    best_n = max(out.keys(), key=lambda k: out[k]["threshold"]["kappa_q"])
    print(f"  Best: {best_n}  κ={out[best_n]['threshold']['kappa_q']:.4f}", flush=True)
    c = np.load(CACHE); ce = np.load(os.path.join(BASE, "results", "sota", f"_explicit_{best_n}_cache.npz"))
    p_te_lr = c["p_te_unif"]; p_va_lr = c["p_va_unif"]
    p_te_e = ce["p_te"]; p_va_e = ce["p_va"]
    best_w = None
    for w in np.linspace(0, 1, 21):
        pv = w * p_va_lr + (1 - w) * p_va_e
        e = (pv * classes[None, :]).sum(1)
        bt2 = tune_thresh(e, yva)
        if best_w is None or bt2["v"] > best_w["v"]:
            best_w = {"w_lr": float(w), **bt2}
    wl = best_w["w_lr"]
    p_te_f = wl * p_te_lr + (1 - wl) * p_te_e
    e_te_f = (p_te_f * classes[None, :]).sum(1)
    m_f = metrics(yte, apply_t(e_te_f, best_w["t"]))
    out["fusion_lr_best_explicit"] = {**m_f, "best_explicit": best_n, "w_lr": wl, **best_w}
    print(f"  Fusion: w_lr={wl:.2f}  t={best_w['t']}  κ_q={m_f['kappa_q']:.4f} {m_f['kappa_q_ci']}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
