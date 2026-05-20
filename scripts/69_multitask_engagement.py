"""
Multi-task learning: predict all 4 DAiSEE affect labels (Boredom, Engagement,
Confusion, Frustration) jointly with shared SigLIP-L features.

Architecture:
  - Shared MLP encoder (1024 -> 256 -> 128)
  - 4 task-specific linear heads (one per affect label, 4 classes each)
Train with summed cross-entropy across tasks.

Evaluate engagement only on test.
Bagged + multi-seed for robustness.

Output:
  results/sota/multitask_results.json
"""
import os, json, csv, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
DAISEE_LABELS = {
    "Train": os.path.join(BASE, "DAiSEE", "Labels", "TrainLabels.csv"),
    "Validation": os.path.join(BASE, "DAiSEE", "Labels", "ValidationLabels.csv"),
    "Test": os.path.join(BASE, "DAiSEE", "Labels", "TestLabels.csv"),
}
OUT = os.path.join(BASE, "results", "sota", "multitask_results.json")
LOG = os.path.join(BASE, "results", "sota", "multitask_status.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))
SEEDS = [42, 7, 2025, 1024, 31337]
TASKS = ["Boredom", "Engagement", "Confusion", "Frustration"]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


class MultiTaskHead(nn.Module):
    def __init__(self, d_in, d_hidden=256, dropout=0.3, n_tasks=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, 128), nn.ReLU(), nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([nn.Linear(128, 4) for _ in range(n_tasks)])

    def forward(self, x):
        h = self.encoder(x)
        return [head(h) for head in self.heads]


def load_all_labels():
    """Returns dict[split][clip_id] = [Boredom, Engagement, Confusion, Frustration]."""
    out = {}
    for split, path in DAISEE_LABELS.items():
        out[split] = {}
        with open(path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                cid = r["ClipID"].strip()
                # Note CSV header has trailing space on "Frustration "; handle both
                key = "Frustration " if "Frustration " in reader.fieldnames else "Frustration"
                out[split][cid] = [int(r["Boredom"]), int(r["Engagement"]),
                                   int(r["Confusion"]), int(r[key])]
    return out


def boot_kq(yt, yp, n=1000):
    out = []
    nt = len(yt)
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


def class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1)
    w = counts.sum() / (n_classes * counts)
    return torch.from_numpy(w).float().to(DEVICE)


def train_multitask(Xtr, Ytr, Xva, Yva, cfg, seed):
    """Ytr/Yva: (N, 4) integer labels for 4 tasks."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = MultiTaskHead(Xtr.shape[1], d_hidden=cfg["d_hidden"], dropout=cfg["dropout"]).to(DEVICE)
    # Class weights per task
    task_weights = []
    for t in range(4):
        task_weights.append(class_weights(Ytr[:, t], 4))
    losses = [nn.CrossEntropyLoss(weight=w) for w in task_weights]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    Xtr_t = torch.from_numpy(Xtr).float().to(DEVICE)
    Ytr_t = torch.from_numpy(Ytr).long().to(DEVICE)
    Xva_t = torch.from_numpy(Xva).float().to(DEVICE)
    Yva_t = torch.from_numpy(Yva).long().to(DEVICE)

    n = len(Ytr); bs = cfg["batch_size"]
    best = {"v_eng": -1e9, "state": None}
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr_t[idx]; yb = Ytr_t[idx]
            outs = model(xb)
            loss = sum(losses[t](outs[t], yb[:, t]) for t in range(4))
            opt.zero_grad(); loss.backward(); opt.step()
        # Validate engagement only
        model.eval()
        with torch.no_grad():
            outs_va = model(Xva_t)
            yhat_eng_va = outs_va[1].argmax(dim=1).cpu().numpy()
        v_eng = float(cohen_kappa_score(Yva[:, 1], yhat_eng_va, weights="quadratic"))
        if v_eng > best["v_eng"]:
            best["v_eng"] = v_eng
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best["state"] is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best["state"].items()})
    return model, best["v_eng"]


def main():
    open(LOG, "w").close()
    log("Loading SigLIP-L features and DAiSEE labels...")
    d = np.load(SIGLIP, allow_pickle=True)
    clip_ids = d["clip_id"]; splits = d["split"]
    X = d["feat"].astype(np.float32)

    labels = load_all_labels()
    Y = np.zeros((len(clip_ids), 4), dtype=np.int64)
    for i, cid in enumerate(clip_ids):
        s = splits[i]
        if cid in labels[s]:
            Y[i] = labels[s][cid]

    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    Xtr, Ytr = X[tr], Y[tr]
    Xva, Yva = X[va], Y[va]
    Xte, Yte = X[te], Y[te]
    log(f"  Train={tr.sum()} Val={va.sum()} Test={te.sum()}")
    log(f"  Label dist Engagement Train: {np.bincount(Ytr[:, 1], minlength=4).tolist()}")
    log(f"  Label dist Boredom Train:    {np.bincount(Ytr[:, 0], minlength=4).tolist()}")
    log(f"  Label dist Confusion Train:  {np.bincount(Ytr[:, 2], minlength=4).tolist()}")
    log(f"  Label dist Frustration Train:{np.bincount(Ytr[:, 3], minlength=4).tolist()}")

    cfg = {"d_hidden": 256, "dropout": 0.3, "lr": 5e-4, "batch_size": 128, "epochs": 30}
    log(f"Config: {cfg}")

    out = {"cfg": cfg, "per_seed": []}
    for seed in SEEDS:
        log(f"\n--- seed {seed} ---")
        model, val_v = train_multitask(Xtr, Ytr, Xva, Yva, cfg, seed)
        model.eval()
        Xte_t = torch.from_numpy(Xte).float().to(DEVICE)
        with torch.no_grad():
            outs_te = model(Xte_t)
            yhat_eng_te = outs_te[1].argmax(dim=1).cpu().numpy()
        m = metrics(Yte[:, 1], yhat_eng_te)
        out["per_seed"].append({"seed": seed, "val_eng_kq": val_v, **m})
        log(f"  val κ_q={val_v:.3f}  test κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}")

    kappas = np.array([r["kappa_q"] for r in out["per_seed"]])
    out["mean_kq"] = float(kappas.mean())
    out["std_kq"] = float(kappas.std(ddof=1) if len(kappas) > 1 else 0.0)
    log(f"\nMulti-task: mean={out['mean_kq']:.3f} ± {out['std_kq']:.3f}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {OUT}")
    log("MULTITASK_DONE")


if __name__ == "__main__":
    main()
