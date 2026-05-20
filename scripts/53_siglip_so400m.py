"""
Encode DAiSEE frames with SigLIP ViT-SO400M-14 (400M parameter, biggest open_clip
SigLIP variant) at 384x384 input. Then run subject-disjoint linear probe and
late-fuse with EMBER explicit signals.

Output:
  features/daisee_siglip_so400m_features.npz
  results/sota/siglip_so400m_probe.json
"""
import os, csv, json, time
import numpy as np
import torch
import open_clip
from PIL import Image
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(BASE, "frames_full", "manifest.csv")
SIGNALS = os.path.join(BASE, "features", "daisee_face_signals.npz")
TEMPORAL = os.path.join(BASE, "features", "daisee_face_signals_temporal.npz")
POSE = os.path.join(BASE, "features", "daisee_pose_signals.npz")
FEAT_OUT = os.path.join(BASE, "features", "daisee_siglip_so400m_features.npz")
RES_OUT = os.path.join(BASE, "results", "sota", "siglip_so400m_probe.json")
LOG = os.path.join(BASE, "results", "sota", "siglip_so400m_status.txt")
os.makedirs(os.path.dirname(RES_OUT), exist_ok=True)
RNG = np.random.default_rng(42)
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def encode():
    if os.path.exists(FEAT_OUT):
        log(f"features exist at {FEAT_OUT}, skipping encoding")
        return
    rows = []
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "exists") and os.path.exists(r["frame_path"]):
                rows.append(r)
    log(f"Loading SigLIP-SO400M-14 ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-SO400M-14-SigLIP", pretrained="webli")
    model = model.to(DEVICE).eval()

    # Test forward shape
    sample = preprocess(Image.open(rows[0]["frame_path"]).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = model.encode_image(sample)
    log(f"  output feature dim: {tuple(f.shape)}")
    feat_dim = f.shape[-1]
    BATCH = 8  # SO400M is memory-heavy

    feats = np.zeros((len(rows), feat_dim), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            imgs = torch.stack([preprocess(Image.open(r["frame_path"]).convert("RGB"))
                                for r in chunk]).to(DEVICE)
            ff = model.encode_image(imgs)
            ff = ff / ff.norm(dim=-1, keepdim=True)
            feats[i:i + len(chunk)] = ff.cpu().numpy()
            if (i // BATCH) % 50 == 0:
                dt = time.time() - t0
                rate = (i + len(chunk)) / dt
                eta = (len(rows) - i - len(chunk)) / rate
                log(f"  {i + len(chunk)}/{len(rows)}  rate={rate:.1f} fps  elapsed={dt:.0f}s  ETA={eta:.0f}s")

    np.savez_compressed(
        FEAT_OUT,
        clip_id=np.array([r["clip_id"] for r in rows]),
        split=np.array([r["split"] for r in rows]),
        subject_id=np.array([r["subject_id"] for r in rows]),
        engagement=np.array([int(r["engagement"]) for r in rows], dtype=np.int64),
        feat=feats,
    )
    log(f"Saved: {FEAT_OUT}  shape={feats.shape}")


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


def probe_and_fuse():
    log("Loading SigLIP-SO400M features for probe...")
    d = np.load(FEAT_OUT, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    eng = d["engagement"].astype(np.int64)
    splits = d["split"]; clip_ids = d["clip_id"]
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"

    out = {}

    # Baseline probe
    clf, val_v, C = fit_lr(X[tr], eng[tr], X[va], eng[va])
    yhat = clf.predict(X[te])
    out["solo_LR"] = {**metrics(eng[te], yhat), "val_kq": val_v, "C": C, "dim": X.shape[1]}
    log(f"  SO400M solo LR κ_q={out['solo_LR']['kappa_q']:.3f} {out['solo_LR']['kappa_q_ci']}  val={val_v:.3f}")

    # Ridge baseline
    best_rd = None
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(X[tr], eng[tr].astype(np.float32))
        v = cohen_kappa_score(eng[va], np.clip(np.round(reg.predict(X[va])), 0, 3).astype(int), weights="quadratic")
        if best_rd is None or v > best_rd["v"]:
            best_rd = {"a": alpha, "v": v, "reg": reg}
    yhat = np.clip(np.round(best_rd["reg"].predict(X[te])), 0, 3).astype(int)
    out["solo_Ridge"] = {**metrics(eng[te], yhat), "alpha": best_rd["a"]}
    log(f"  SO400M solo Ridge κ_q={out['solo_Ridge']['kappa_q']:.3f} {out['solo_Ridge']['kappa_q_ci']}")

    # Get probe probabilities for fusion
    p_cls_va = clf.predict_proba(X[va])
    p_cls_te = clf.predict_proba(X[te])

    # Fuse with EMBER explicit signals
    log("\nFusion with EMBER explicit signals...")
    if os.path.exists(SIGNALS):
        sf = np.load(SIGNALS, allow_pickle=True)
        sf_ids = sf["clip_id"]
        sf_exp = np.nan_to_num(np.concatenate([sf["blendshapes"], sf["head_pose"],
                                                sf["eye_gaze"], sf["landmark_summary"]], axis=1))
        exp_single = align(clip_ids, sf_ids, sf_exp)
        sc = StandardScaler().fit(exp_single[tr])
        exp_single = sc.transform(exp_single).astype(np.float32)
        clf_e, _, _ = fit_lr(exp_single[tr], eng[tr], exp_single[va], eng[va])
        p_e_va = clf_e.predict_proba(exp_single[va])
        p_e_te = clf_e.predict_proba(exp_single[te])

        # Scalar weight sweep
        best_w = None
        for w in np.linspace(0.0, 1.0, 41):
            p_va = w * p_e_va + (1 - w) * p_cls_va
            v = cohen_kappa_score(eng[va], p_va.argmax(axis=1), weights="quadratic")
            if best_w is None or v > best_w["v"]:
                best_w = {"w": float(w), "v": float(v)}
        wb = best_w["w"]
        yhat = (wb * p_e_te + (1 - wb) * p_cls_te).argmax(axis=1)
        out["SO400M+explicit_late_scalar"] = {**metrics(eng[te], yhat),
                                                "w": wb, "val_kq": best_w["v"]}
        log(f"  SO400M + single-frame explicit (w={wb:.2f}) κ_q={out['SO400M+explicit_late_scalar']['kappa_q']:.3f} {out['SO400M+explicit_late_scalar']['kappa_q_ci']}")

    if os.path.exists(TEMPORAL):
        tmp = np.load(TEMPORAL, allow_pickle=True)
        tmp_ids = tmp["clip_id"]
        tmp_exp = np.nan_to_num(np.concatenate([
            tmp["blendshapes_mean"], tmp["blendshapes_std"], tmp["blendshapes_delta"],
            tmp["head_pose_mean"], tmp["head_pose_std"],
            tmp["eye_gaze_mean"], tmp["eye_gaze_std"],
            tmp["landmark_summary_mean"]], axis=1))
        exp_temp = align(clip_ids, tmp_ids, tmp_exp)
        sc = StandardScaler().fit(exp_temp[tr])
        exp_temp = sc.transform(exp_temp).astype(np.float32)
        clf_t, _, _ = fit_lr(exp_temp[tr], eng[tr], exp_temp[va], eng[va])
        p_t_va = clf_t.predict_proba(exp_temp[va])
        p_t_te = clf_t.predict_proba(exp_temp[te])

        best_w = None
        for w in np.linspace(0.0, 1.0, 41):
            p_va = w * p_t_va + (1 - w) * p_cls_va
            v = cohen_kappa_score(eng[va], p_va.argmax(axis=1), weights="quadratic")
            if best_w is None or v > best_w["v"]:
                best_w = {"w": float(w), "v": float(v)}
        wb = best_w["w"]
        yhat = (wb * p_t_te + (1 - wb) * p_cls_te).argmax(axis=1)
        out["SO400M+temporal_explicit_scalar"] = {**metrics(eng[te], yhat),
                                                     "w": wb, "val_kq": best_w["v"]}
        log(f"  SO400M + temporal explicit (w={wb:.2f}) κ_q={out['SO400M+temporal_explicit_scalar']['kappa_q']:.3f} {out['SO400M+temporal_explicit_scalar']['kappa_q_ci']}")

    with open(RES_OUT, "w") as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {RES_OUT}")
    log("SIGLIP_SO400M_DONE")


def main():
    open(LOG, "w").close()
    encode()
    probe_and_fuse()


if __name__ == "__main__":
    main()
