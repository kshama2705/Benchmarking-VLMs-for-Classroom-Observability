"""
MOONSHOT 2: FER2013 emotion → DAiSEE engagement transfer.

Memory: FER2013 LR on SigLIP-L gave κ=0.554 acc=62.8% (7 emotion classes).
Same encoder hits κ=0.20 on DAiSEE engagement. Maybe FER2013-emotion features
(when projected via emotion classifier head) are more useful for engagement.

Plan:
  A) Train FER2013 LR on FER2013 features.
  B) Apply that LR to DAiSEE features → get 7-d emotion soft labels per DAiSEE clip.
  C) Concat emotion soft labels with raw SigLIP-L features.
  D) Train new LR bag on concat features → threshold tune.

Output:
  results/sota/moonshot_fer_transfer.json
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
FER_CLIP = os.path.join(BASE, "features", "fer2013_clip_features.npz")
FER_DINO = os.path.join(BASE, "features", "fer2013_dinov2_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_fer_transfer.json")
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


def main():
    print("Loading...", flush=True)
    d = np.load(SIGLIP, allow_pickle=True)
    Xs = d["feat"].astype(np.float32)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    Xtr_s, ytr = Xs[tr], eng[tr]; Xva_s, yva = Xs[va], eng[va]; Xte_s, yte = Xs[te], eng[te]

    # FER2013 only has CLIP-B/32 features (512-d), not SigLIP-L. Check.
    print("Checking FER2013 features...", flush=True)
    fer = np.load(FER_CLIP, allow_pickle=True)
    print(f"  FER2013 keys: {fer.files}, train shape: {fer['train_feats'].shape}", flush=True)
    # FER features are 512-d (CLIP-B/32) — different encoder than SigLIP-L.
    # We can't directly transfer projection to SigLIP-L features.
    # Strategy: train FER-emotion LR on FER's CLIP-B/32 features → 7-d soft emotion
    # Need: DAiSEE features in CLIP-B/32 space to apply that LR.
    daisee_clip = np.load(os.path.join(BASE, "features", "clip_vitb32_features.npz"), allow_pickle=True)
    Xd_clip = daisee_clip["feat"].astype(np.float32)
    Xtr_c = Xd_clip[tr]; Xva_c = Xd_clip[va]; Xte_c = Xd_clip[te]
    print(f"  DAiSEE CLIP-B/32 feat shape: {Xd_clip.shape}", flush=True)

    # ============ Step A: Train FER-emotion LR (on CLIP-B/32) ============
    Xfer_tr = fer["train_feats"].astype(np.float32)
    yfer_tr = fer["train_lbl"].astype(np.int64)
    Xfer_te = fer["test_feats"].astype(np.float32)
    yfer_te = fer["test_lbl"].astype(np.int64)
    print(f"  FER train: {Xfer_tr.shape}  test: {Xfer_te.shape}  label range: {yfer_tr.min()}-{yfer_tr.max()}", flush=True)

    print("\n[A] Training FER-emotion LR on CLIP-B/32...", flush=True)
    fer_clf = fit_lr(Xfer_tr, yfer_tr, Xfer_te, yfer_te)
    fer_acc = (fer_clf.predict(Xfer_te) == yfer_te).mean()
    fer_kq = cohen_kappa_score(yfer_te, fer_clf.predict(Xfer_te), weights="quadratic")
    print(f"  FER LR: test acc={fer_acc:.3f}, κ_q={fer_kq:.3f}", flush=True)

    # ============ Step B: Apply FER LR to DAiSEE CLIP-B/32 features ============
    print("\n[B] Extracting FER-emotion soft features for DAiSEE...", flush=True)
    pemo_tr = fer_clf.predict_proba(Xtr_c)
    pemo_va = fer_clf.predict_proba(Xva_c)
    pemo_te = fer_clf.predict_proba(Xte_c)
    print(f"  Emotion soft labels: tr {pemo_tr.shape}, va {pemo_va.shape}, te {pemo_te.shape}", flush=True)
    # Examine: what emotion does the model predict for engaged vs not engaged DAiSEE?
    # FER label order: 0:angry 1:disgust 2:fear 3:happy 4:sad 5:surprise 6:neutral
    print(f"\n  Mean emotion probs per engagement class (train):", flush=True)
    for e in range(4):
        mask = ytr == e
        if mask.sum() == 0: continue
        mean_e = pemo_tr[mask].mean(0)
        print(f"    eng={e} (n={mask.sum()}): {mean_e.tolist()}", flush=True)

    out = {"fer_val_acc": float(fer_acc), "fer_val_kq": float(fer_kq)}

    # ============ Step C: Solo bag on emotion features ============
    print("\n[C] LR bag on emotion-only features (7-d)...", flush=True)
    p_te_emo, p_va_emo = bag_lr(pemo_tr, ytr, pemo_va, yva, pemo_te, K=20)
    m_solo_emo = metrics(yte, p_te_emo.argmax(1))
    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    e_va_emo = (p_va_emo * classes[None, :]).sum(1)
    e_te_emo = (p_te_emo * classes[None, :]).sum(1)
    bt_emo = tune_thresh(e_va_emo, yva)
    m_thr_emo = metrics(yte, apply_t(e_te_emo, bt_emo["t"]))
    out["solo_emotion_bag"] = m_solo_emo
    out["emotion_threshold"] = {**m_thr_emo, **bt_emo}
    print(f"  Solo emotion: κ_q={m_solo_emo['kappa_q']:.4f}", flush=True)
    print(f"  +threshold: κ_q={m_thr_emo['kappa_q']:.4f} {m_thr_emo['kappa_q_ci']} t={bt_emo['t']}", flush=True)

    # ============ Step D: Concat SigLIP-L + emotion features, bag again ============
    print("\n[D] LR bag on SigLIP-L ⊕ emotion (1024+7=1031-d)...", flush=True)
    Xtr_cat = np.concatenate([Xtr_s, pemo_tr], axis=1)
    Xva_cat = np.concatenate([Xva_s, pemo_va], axis=1)
    Xte_cat = np.concatenate([Xte_s, pemo_te], axis=1)
    p_te_cat, p_va_cat = bag_lr(Xtr_cat, ytr, Xva_cat, yva, Xte_cat, K=20)
    m_solo_cat = metrics(yte, p_te_cat.argmax(1))
    e_va_cat = (p_va_cat * classes[None, :]).sum(1)
    e_te_cat = (p_te_cat * classes[None, :]).sum(1)
    bt_cat = tune_thresh(e_va_cat, yva)
    m_thr_cat = metrics(yte, apply_t(e_te_cat, bt_cat["t"]))
    out["solo_siglip_plus_emotion"] = m_solo_cat
    out["siglip_plus_emotion_threshold"] = {**m_thr_cat, **bt_cat}
    print(f"  Solo concat: κ_q={m_solo_cat['kappa_q']:.4f}", flush=True)
    print(f"  +threshold: κ_q={m_thr_cat['kappa_q']:.4f} {m_thr_cat['kappa_q_ci']}", flush=True)

    # ============ Step E: Fusion with cached LR-unif ============
    print("\n[E] Fusion with cached LR-unif...", flush=True)
    c = np.load(CACHE)
    p_te_lr = c["p_te_unif"]; p_va_lr = c["p_va_unif"]
    best_w = None
    for w in np.linspace(0, 1, 21):
        pv = w * p_va_lr + (1 - w) * p_va_cat
        e = (pv * classes[None, :]).sum(1)
        bt2 = tune_thresh(e, yva)
        if best_w is None or bt2["v"] > best_w["v"]:
            best_w = {"w_lr": float(w), **bt2}
    wl = best_w["w_lr"]
    p_te_f = wl * p_te_lr + (1 - wl) * p_te_cat
    e_te_f = (p_te_f * classes[None, :]).sum(1)
    m_f = metrics(yte, apply_t(e_te_f, best_w["t"]))
    out["fusion_lr_concat"] = {**m_f, "w_lr": wl, **best_w}
    print(f"  Fusion: w_lr={wl:.2f}  t={best_w['t']}  κ_q={m_f['kappa_q']:.4f} {m_f['kappa_q_ci']}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
