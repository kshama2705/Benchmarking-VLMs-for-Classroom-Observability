"""
N6: Multi-seed validation of best SIEP configs.

Re-trains the headline configs across 5 seeds to confirm seed-stability:
  - CLIP: SIEP λ=0 (MLP-only, d=256, 2-layer)
  - DINOv2: SIEP λ=2 (DANN, d=256, 2-layer)
  - DINOv2: SIEP λ=0 (MLP-only, d=256, 2-layer)

For each config, reports per-seed κ_q + bootstrap CI on test, plus mean ± std
across seeds.

Output:
  results/siep/siep_multiseed.json
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_squared_error

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Inline-import the SIEP class & helpers from script 26
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location(
    "siep_mod",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "26_siep.py"))
siep_mod = module_from_spec(spec)
spec.loader.exec_module(siep_mod)
SIEP = siep_mod.SIEP
class_weights = siep_mod.class_weights
evaluate = siep_mod.evaluate
GradReverse = siep_mod.GradReverse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "results", "siep")
os.makedirs(OUT_DIR, exist_ok=True)
RNG = np.random.default_rng(42)
DEVICE = siep_mod.DEVICE

FEATS = {
    "CLIP_ViT-B32": os.path.join(BASE, "features", "clip_vitb32_features.npz"),
    "DINOv2_ViT-B14": os.path.join(BASE, "features", "dinov2_vitb14_features.npz"),
}

SEEDS = [42, 7, 123, 2025, 999]


def boot_kappa(yt, yp, n=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = rng.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def load_split(path):
    data = np.load(path, allow_pickle=True)
    splits = data["split"]; feats = data["feat"].astype(np.float32)
    eng = data["engagement"]; subj = data["subject_id"]
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    unique = sorted(set(subj[tr].tolist()))
    sid_map = {s: i for i, s in enumerate(unique)}
    str_arr = np.array([sid_map[s] for s in subj[tr]], dtype=np.int64)
    return (feats[tr], eng[tr].astype(np.int64), str_arr,
            feats[va], eng[va].astype(np.int64),
            feats[te], eng[te].astype(np.int64),
            len(unique))


def train_one(Xtr, ytr, str_, Xva, yva, n_subj, cfg, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d_in = Xtr.shape[1]
    model = SIEP(d_in, cfg["d_hidden"], 4, n_subj,
                 n_layers=cfg["n_layers"], dropout=cfg["dropout"]).to(DEVICE)
    eng_w = class_weights(ytr, 4)
    eng_loss_fn = nn.CrossEntropyLoss(weight=eng_w)
    subj_loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    Xtr_t = torch.from_numpy(Xtr).float().to(DEVICE)
    ytr_t = torch.from_numpy(ytr).long().to(DEVICE)
    str_t = torch.from_numpy(str_).long().to(DEVICE)
    n = len(Xtr); bs = cfg["batch_size"]
    best = {"v": -1e9, "state": None}
    for ep in range(cfg["epochs"]):
        progress = ep / max(cfg["epochs"] - 1, 1)
        lambd = cfg["lambda_max"] * (2.0 / (1.0 + np.exp(-10 * progress)) - 1.0)
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr_t[idx]; yb = ytr_t[idx]; sb = str_t[idx]
            eng_logits, subj_logits, _ = model(xb, lambd=float(lambd))
            le = eng_loss_fn(eng_logits, yb)
            ls = subj_loss_fn(subj_logits, sb)
            loss = le + ls
            opt.zero_grad(); loss.backward(); opt.step()
        # Validate
        yhat_va = evaluate(model, Xva, yva)
        v = cohen_kappa_score(yva, yhat_va, weights="quadratic")
        if v > best["v"]:
            best["v"] = float(v)
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model


def run_config(label, feats_path, cfg, seeds):
    Xtr, ytr, str_, Xva, yva, Xte, yte, n_subj = load_split(feats_path)
    print(f"\n  Encoder={label}  config={cfg}")
    rows = []
    for seed in seeds:
        model = train_one(Xtr, ytr, str_, Xva, yva, n_subj, cfg, seed)
        yhat = evaluate(model, Xte, yte)
        kq = float(cohen_kappa_score(yte, yhat, weights="quadratic"))
        acc = float(accuracy_score(yte, yhat))
        ci = boot_kappa(yte, yhat)
        rows.append({"seed": seed, "kappa_q": kq, "kappa_q_ci95": ci, "accuracy": acc})
        print(f"    seed={seed:5d}  κ_q={kq:.3f} [{ci[0]:.3f},{ci[1]:.3f}]  acc={acc:.3f}", flush=True)
    kappas = np.array([r["kappa_q"] for r in rows])
    summary = {
        "encoder": label,
        "config": cfg,
        "per_seed": rows,
        "mean_kq": float(kappas.mean()),
        "std_kq": float(kappas.std(ddof=1) if len(kappas) > 1 else 0.0),
        "min_kq": float(kappas.min()),
        "max_kq": float(kappas.max()),
    }
    print(f"    => mean={summary['mean_kq']:.3f} ± {summary['std_kq']:.3f}  "
          f"(min={summary['min_kq']:.3f}, max={summary['max_kq']:.3f})", flush=True)
    return summary


def main():
    runs = [
        ("DINOv2_ViT-B14", FEATS["DINOv2_ViT-B14"],
         {"d_hidden": 256, "n_layers": 2, "dropout": 0.2, "lr": 1e-3,
          "batch_size": 64, "epochs": 30, "lambda_max": 2.0}),
        ("DINOv2_ViT-B14", FEATS["DINOv2_ViT-B14"],
         {"d_hidden": 256, "n_layers": 2, "dropout": 0.2, "lr": 1e-3,
          "batch_size": 64, "epochs": 30, "lambda_max": 0.0}),
        ("CLIP_ViT-B32", FEATS["CLIP_ViT-B32"],
         {"d_hidden": 256, "n_layers": 2, "dropout": 0.2, "lr": 1e-3,
          "batch_size": 64, "epochs": 30, "lambda_max": 0.0}),
    ]
    summaries = []
    for label, path, cfg in runs:
        summaries.append(run_config(label, path, cfg, SEEDS))

    out_path = os.path.join(OUT_DIR, "siep_multiseed.json")
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
