"""
Quick free experiment: concatenated face + bg + cls probe on DINOv2 patch features.
Tests if patches carry unique info beyond CLS, even if pooled simply.

Reads features/dinov2_patch_face_features.npz
Writes results/ppep/ppep_concat_results.json
"""
import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(BASE, "features", "dinov2_patch_face_features.npz")
OUT = os.path.join(BASE, "results", "ppep", "ppep_concat_results.json")
RNG = np.random.default_rng(42)


def boot_kq(yt, yp, n=1000):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def probe(Xtr, ytr, Xva, yva, Xte, yte, label):
    best = None
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                                 solver="lbfgs", random_state=42, n_jobs=-1)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": v, "clf": clf}
    yhat = best["clf"].predict(Xte)
    m_lr = {"kappa_q": float(cohen_kappa_score(yte, yhat, weights="quadratic")),
            "kappa_q_ci": boot_kq(yte, yhat),
            "accuracy": float(accuracy_score(yte, yhat)),
            "best_C": best["C"]}
    best = None
    for alpha in [1.0, 10.0, 100.0, 1000.0]:
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(Xtr, ytr.astype(np.float32))
        v = cohen_kappa_score(yva, np.clip(np.round(reg.predict(Xva)), 0, 3).astype(int), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"a": alpha, "v": v, "reg": reg}
    yhat = np.clip(np.round(best["reg"].predict(Xte)), 0, 3).astype(int)
    m_rd = {"kappa_q": float(cohen_kappa_score(yte, yhat, weights="quadratic")),
            "kappa_q_ci": boot_kq(yte, yhat),
            "accuracy": float(accuracy_score(yte, yhat)),
            "best_alpha": best["a"]}
    print(f"  {label:24} LR    κ_q={m_lr['kappa_q']:.3f} {m_lr['kappa_q_ci']}  acc={m_lr['accuracy']:.3f}")
    print(f"  {label:24} Ridge κ_q={m_rd['kappa_q']:.3f} {m_rd['kappa_q_ci']}  acc={m_rd['accuracy']:.3f}")
    return {"logreg": m_lr, "ridge_ordinal": m_rd}


def main():
    d = np.load(FEAT, allow_pickle=True)
    splits = d["split"]; eng = d["engagement"]
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    face, bg, cls, mean = (d["face_feat"].astype(np.float32),
                            d["bg_feat"].astype(np.float32),
                            d["cls_feat"].astype(np.float32),
                            d["mean_feat"].astype(np.float32))

    print("\nConcat experiments (DINOv2 patch pooling):")
    configs = {
        "cls_only": cls,
        "face_only": face,
        "bg_only": bg,
        "face+bg": np.concatenate([face, bg], axis=1),
        "face+cls": np.concatenate([face, cls], axis=1),
        "bg+cls": np.concatenate([bg, cls], axis=1),
        "face+bg+cls": np.concatenate([face, bg, cls], axis=1),
        "face+bg+cls+mean": np.concatenate([face, bg, cls, mean], axis=1),
    }
    out = {}
    for name, X in configs.items():
        out[name] = probe(X[tr], eng[tr], X[va], eng[va], X[te], eng[te], name)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
