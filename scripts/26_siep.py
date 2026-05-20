"""
N5: SIEP — Subject-Invariant Engagement Probe.

Non-linear MLP head on top of frozen CLIP/DINOv2 features.
Two parallel objectives:
  1. Engagement cross-entropy (class-balanced)
  2. Subject-invariance via gradient-reversal-layer adversarial subject classifier
     (Ganin et al., DANN, ICML 2015)

The encoder is a 2-3 layer MLP that produces a d_hidden representation. From that:
  - eng_head -> 4 logits, trained directly on engagement label
  - subj_head -> N_subj logits, trained via Gradient Reversal Layer (GRL)
    Encoder gets reversed gradients from subj_head, so it's pushed to make
    subjects unpredictable while preserving engagement predictability.

Subject-disjoint protocol:
  - Train: 5,358 clips / 70 subjects
  - Val:   1,429 clips / 22 subjects (subject-disjoint from Train)
  - Test:  1,784 clips / 21 subjects (subject-disjoint from both)

Subject loss is meaningful only on Train (subject IDs differ in val/test).
Engagement loss can be computed on all splits.

Output:
  results/siep/siep_results.json
  results/siep/siep_predictions_<encoder>_<config>.csv
"""

import os, json, csv, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_squared_error

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "results", "siep")
os.makedirs(OUT_DIR, exist_ok=True)
RNG = np.random.default_rng(42)
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else \
         torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

FEATS = {
    "CLIP_ViT-B32": os.path.join(BASE, "features", "clip_vitb32_features.npz"),
    "DINOv2_ViT-B14": os.path.join(BASE, "features", "dinov2_vitb14_features.npz"),
}


# ============ Gradient Reversal Layer ============
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.lambd, None


def grl(x, lambd):
    return GradReverse.apply(x, lambd)


# ============ Model ============
class SIEP(nn.Module):
    def __init__(self, d_in, d_hidden, n_eng, n_subj, n_layers=2, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_d = d_in if i == 0 else d_hidden
            layers += [nn.Linear(in_d, d_hidden), nn.ReLU(), nn.Dropout(dropout)]
        self.encoder = nn.Sequential(*layers)
        self.eng_head = nn.Linear(d_hidden, n_eng)
        self.subj_head = nn.Linear(d_hidden, n_subj)

    def forward(self, x, lambd=1.0):
        z = self.encoder(x)
        eng_logits = self.eng_head(z)
        subj_logits = self.subj_head(grl(z, lambd))
        return eng_logits, subj_logits, z


# ============ Training ============
def class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1)
    w = counts.sum() / (n_classes * counts)
    return torch.from_numpy(w).float().to(DEVICE)


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


def evaluate(model, X, y, batch=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).float().to(DEVICE)
            eng_logits, _, _ = model(xb, lambd=0.0)
            preds.append(eng_logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def train_siep(Xtr, ytr, str_, Xva, yva, n_eng, n_subj, cfg):
    """str_ = subject-id ints for train; subjects in val/test are different so we
    don't compute subject loss on them."""
    d_in = Xtr.shape[1]
    model = SIEP(d_in, cfg["d_hidden"], n_eng, n_subj,
                 n_layers=cfg["n_layers"], dropout=cfg["dropout"]).to(DEVICE)

    eng_w = class_weights(ytr, n_eng)
    eng_loss = nn.CrossEntropyLoss(weight=eng_w)
    subj_loss = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    Xtr_t = torch.from_numpy(Xtr).float().to(DEVICE)
    ytr_t = torch.from_numpy(ytr).long().to(DEVICE)
    str_t = torch.from_numpy(str_).long().to(DEVICE)
    n = len(Xtr)
    bs = cfg["batch_size"]
    n_epochs = cfg["epochs"]

    best = {"val_kq": -1e9, "epoch": -1, "state": None}

    for epoch in range(n_epochs):
        # GRL warmup: lambda ramps from 0 -> lambda_max over training
        # following Ganin et al. schedule
        progress = epoch / max(n_epochs - 1, 1)
        lambd = cfg["lambda_max"] * (2.0 / (1.0 + np.exp(-10 * progress)) - 1.0)

        model.train()
        perm = torch.randperm(n)
        running_e, running_s = 0.0, 0.0
        nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr_t[idx]; yb = ytr_t[idx]; sb = str_t[idx]
            eng_logits, subj_logits, _ = model(xb, lambd=float(lambd))
            le = eng_loss(eng_logits, yb)
            ls = subj_loss(subj_logits, sb)
            loss = le + ls  # GRL inside subj_logits handles sign for encoder
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_e += le.item(); running_s += ls.item(); nb += 1

        # Validate on engagement only (val subjects unseen to subj_head)
        yhat_va = evaluate(model, Xva, yva)
        val_kq = cohen_kappa_score(yva, yhat_va, weights="quadratic")
        if val_kq > best["val_kq"]:
            best.update({"val_kq": float(val_kq), "epoch": epoch,
                         "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}})
        if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
            print(f"    ep{epoch+1:3d}  λ={lambd:.3f}  eng_loss={running_e/nb:.3f}  "
                  f"subj_loss={running_s/nb:.3f}  val_κq={val_kq:.3f}  (best={best['val_kq']:.3f}@ep{best['epoch']+1})",
                  flush=True)

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, best["val_kq"]


def run_for_features(label, feats_path, configs):
    print(f"\n========== {label} ==========", flush=True)
    data = np.load(feats_path, allow_pickle=True)
    splits = data["split"]; feats = data["feat"].astype(np.float32)
    eng = data["engagement"]; subj = data["subject_id"]
    cid = data["clip_id"]
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"

    # Subject-int mapping (train subjects only)
    unique = sorted(set(subj[tr].tolist()))
    sid_map = {s: i for i, s in enumerate(unique)}
    str_arr = np.array([sid_map[s] for s in subj[tr]], dtype=np.int64)

    Xtr, ytr = feats[tr], eng[tr].astype(np.int64)
    Xva, yva = feats[va], eng[va].astype(np.int64)
    Xte, yte = feats[te], eng[te].astype(np.int64)
    cte = cid[te]

    print(f"Train={len(Xtr)} ({len(unique)} subj)  Val={len(Xva)}  Test={len(Xte)}  feat_dim={Xtr.shape[1]}",
          flush=True)

    results = []
    for cfg in configs:
        print(f"\n  Config: {cfg}", flush=True)
        torch.manual_seed(cfg.get("seed", 42))
        model, val_kq = train_siep(Xtr, ytr, str_arr, Xva, yva, 4, len(unique), cfg)
        yhat_te = evaluate(model, Xte, yte)
        m = metrics(yte, yhat_te)
        m["kappa_q_ci95"] = boot_kappa(yte, yhat_te)
        m["val_kappa_q"] = float(val_kq)
        m["config"] = cfg
        print(f"  -> Test: κ_q={m['kappa_quadratic']:.3f} [{m['kappa_q_ci95'][0]:.3f},"
              f"{m['kappa_q_ci95'][1]:.3f}]  acc={m['accuracy']:.3f}  pred_dist={m['pred_dist']}",
              flush=True)
        results.append(m)

        # Save predictions for the best config (last in sweep)
        out_csv = os.path.join(OUT_DIR, f"siep_preds_{label.replace('/', '_')}_lh{cfg['d_hidden']}_l{cfg['lambda_max']}.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip_id", "predicted", "ground_truth"])
            for c, p, y in zip(cte, yhat_te, yte):
                w.writerow([c, int(p), int(y)])
    return results


def main():
    # Hyperparameter sweep
    configs = [
        {"d_hidden": 256, "n_layers": 2, "dropout": 0.2, "lr": 1e-3, "batch_size": 64, "epochs": 30, "lambda_max": 0.0,  "seed": 42},   # baseline (no adversarial) — to compare to SIEP
        {"d_hidden": 256, "n_layers": 2, "dropout": 0.2, "lr": 1e-3, "batch_size": 64, "epochs": 30, "lambda_max": 0.5,  "seed": 42},
        {"d_hidden": 256, "n_layers": 2, "dropout": 0.2, "lr": 1e-3, "batch_size": 64, "epochs": 30, "lambda_max": 1.0,  "seed": 42},
        {"d_hidden": 256, "n_layers": 2, "dropout": 0.2, "lr": 1e-3, "batch_size": 64, "epochs": 30, "lambda_max": 2.0,  "seed": 42},
        {"d_hidden": 512, "n_layers": 2, "dropout": 0.3, "lr": 1e-3, "batch_size": 64, "epochs": 30, "lambda_max": 1.0,  "seed": 42},
        {"d_hidden": 128, "n_layers": 3, "dropout": 0.2, "lr": 1e-3, "batch_size": 64, "epochs": 30, "lambda_max": 1.0,  "seed": 42},
    ]

    out = {}
    for label, path in FEATS.items():
        if not os.path.exists(path):
            continue
        out[label] = run_for_features(label, path, configs)

    out_path = os.path.join(OUT_DIR, "siep_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
