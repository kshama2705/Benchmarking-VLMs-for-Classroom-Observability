"""
End-to-end ResNet18 fine-tune on DAiSEE engagement (MPS-friendly).

ResNet18 (11.7M params, ImageNet pretrained) fully fine-tuned. Single
frame (t=5s) per clip; 224x224 input; ImageNet normalization.

Training:
- Class-balanced focal loss (gamma=2.0)
- AdamW: backbone LR 1e-4, head LR 1e-3
- Linear warmup 200 steps + cosine decay
- 15 epochs, batch 32 (no accumulation needed — small model)
- 3 seeds; ensemble averages softmax probabilities

This is ~8x faster than SigLIP-L on the same hardware. Expected per
seed: 1.5-2.5h on MPS. Total 5-8h for 3 seeds.

Output: results/sota/resnet18_e2e.json + per-epoch test κ_q logs.
"""
import os, json, time, csv, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import cohen_kappa_score, accuracy_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(BASE, "frames_full", "manifest.csv")
OUT = os.path.join(BASE, "results", "sota", "resnet18_e2e.json")
LOG = os.path.join(BASE, "results", "sota", "resnet18_e2e.log")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class DaiSEEFrames(Dataset):
    def __init__(self, rows, tfm):
        self.rows = rows; self.tfm = tfm

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["frame_path"]).convert("RGB")
        x = self.tfm(img)
        y = int(r["engagement"])
        return x, y


def focal_loss(logits, y, alpha, gamma=2.0):
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    nll = -logp.gather(1, y.unsqueeze(1)).squeeze(1)
    pt = p.gather(1, y.unsqueeze(1)).squeeze(1)
    a = alpha[y]
    return (a * (1 - pt).pow(gamma) * nll).mean()


def tune_thresholds(e, y, step=0.04):
    grid = np.arange(0.0, 3.01, step); best = None
    for t1 in grid:
        for t2 in grid[grid > t1]:
            for t3 in grid[grid > t2]:
                yp = np.zeros_like(e, dtype=int)
                yp[e > t1] = 1; yp[e > t2] = 2; yp[e > t3] = 3
                v = cohen_kappa_score(y, yp, weights='quadratic')
                if best is None or v > best['v']:
                    best = {'t': (float(t1), float(t2), float(t3)), 'v': float(v)}
    return best


def apply_thr(e, t):
    yp = np.zeros_like(e, dtype=int)
    yp[e > t[0]] = 1; yp[e > t[1]] = 2; yp[e > t[2]] = 3
    return yp


def boot_ci(yt, yp, n=1000, seed=42):
    rng = np.random.default_rng(seed); nt = len(yt); out = []
    for _ in range(n):
        idx = rng.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def load_manifest():
    rows = []
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "exists") and os.path.exists(r["frame_path"]):
                rows.append(r)
    return rows


def train_one(rows_train, rows_val, rows_test, seed=42, epochs=15, batch=32,
              lr_backbone=1e-4, lr_head=1e-3, gamma_focal=2.0):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = DEVICE

    log(f"[seed {seed}] Building ResNet18 (ImageNet pretrained)...")
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    head = nn.Sequential(
        nn.Linear(feat_dim, feat_dim),
        nn.GELU(), nn.Dropout(0.2),
        nn.Linear(feat_dim, 4),
    )
    backbone = backbone.to(device); head = head.to(device)

    cnt = np.zeros(4)
    for r in rows_train: cnt[int(r["engagement"])] += 1
    alpha = 1.0 / np.maximum(cnt, 1); alpha = alpha / alpha.sum() * 4
    alpha_t = torch.tensor(alpha, dtype=torch.float32, device=device)
    log(f"[seed {seed}] class counts: {cnt.tolist()}  alpha: {alpha.tolist()}")

    train_ds = DaiSEEFrames(rows_train, train_tfm)
    val_ds = DaiSEEFrames(rows_val, eval_tfm)
    test_ds = DaiSEEFrames(rows_test, eval_tfm)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2, pin_memory=False, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=batch*2, shuffle=False, num_workers=2, persistent_workers=True)
    test_dl = DataLoader(test_ds, batch_size=batch*2, shuffle=False, num_workers=2, persistent_workers=True)

    opt = torch.optim.AdamW([
        {"params": list(backbone.parameters()), "lr": lr_backbone},
        {"params": list(head.parameters()), "lr": lr_head},
    ], weight_decay=0.05)
    base = [lr_backbone, lr_head]
    n_steps = len(train_dl) * epochs
    warmup = max(200, n_steps // 20)

    def set_lr(step):
        s = step / max(1, warmup) if step < warmup else 0.5 * (1 + np.cos(np.pi * (step - warmup) / max(1, n_steps - warmup)))
        for pg, b in zip(opt.param_groups, base):
            pg["lr"] = b * s

    @torch.no_grad()
    def eval_split(dl):
        backbone.eval(); head.eval()
        probs = []; ys = []
        for x, y in dl:
            x = x.to(device)
            logits = head(backbone(x))
            p = F.softmax(logits, dim=1).cpu().numpy()
            probs.append(p); ys.append(y.numpy())
        return np.concatenate(probs), np.concatenate(ys)

    history = []; step = 0
    log(f"[seed {seed}] Training {epochs} epochs, {n_steps} optimizer steps...")
    for ep in range(epochs):
        backbone.train(); head.train()
        ep_t0 = time.time(); losses = []
        for i, (x, y) in enumerate(train_dl):
            x = x.to(device); y = y.to(device)
            set_lr(step); step += 1
            opt.zero_grad()
            logits = head(backbone(x))
            loss = focal_loss(logits, y, alpha_t, gamma=gamma_focal)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(head.parameters()), 1.0)
            opt.step()
            losses.append(loss.item())
            if step % 50 == 0:
                log(f"  [seed {seed}] ep{ep} step{step}/{n_steps} loss={np.mean(losses[-50:]):.3f} lr_bb={opt.param_groups[0]['lr']:.2e}")
        p_va, y_va = eval_split(val_dl); p_te, y_te = eval_split(test_dl)
        classes = np.arange(4, dtype=np.float32)
        e_va = (p_va * classes[None]).sum(1); e_te = (p_te * classes[None]).sum(1)
        bt = tune_thresholds(e_va, y_va)
        yp_arg = p_te.argmax(1); yp_thr = apply_thr(e_te, bt['t'])
        val_kq = float(cohen_kappa_score(y_va, p_va.argmax(1), weights="quadratic"))
        test_kq_arg = float(cohen_kappa_score(y_te, yp_arg, weights="quadratic"))
        test_kq_thr = float(cohen_kappa_score(y_te, yp_thr, weights="quadratic"))
        dt = time.time() - ep_t0
        history.append({"ep": ep, "val_kq_arg": val_kq, "val_kq_thr": float(bt['v']),
                       "test_kq_arg": test_kq_arg, "test_kq_thr": test_kq_thr,
                       "thresholds": bt['t'], "duration_min": dt/60,
                       "mean_loss": float(np.mean(losses))})
        log(f"[seed {seed} ep{ep}] loss={np.mean(losses):.3f} val_arg={val_kq:.4f} val_thr={bt['v']:.4f} test_arg={test_kq_arg:.4f} test_thr={test_kq_thr:.4f} dt={dt/60:.1f}min")

    # Return final-epoch test predictions for ensembling
    p_va, y_va = eval_split(val_dl); p_te, y_te = eval_split(test_dl)
    return {"history": history, "p_te": p_te, "p_va": p_va, "y_te": y_te, "y_va": y_va}


def main():
    log(f"Device: {DEVICE}")
    rows = load_manifest()
    by_split = {}
    for r in rows: by_split.setdefault(r["split"], []).append(r)
    log(f"Splits: " + ", ".join(f"{s}={len(v)}" for s, v in by_split.items()))

    seeds = [0, 42, 2025]
    all_p_te = []; all_p_va = []; y_te = y_va = None; seed_results = []
    for s in seeds:
        log(f"\n========== Seed {s} ==========")
        r = train_one(by_split["Train"], by_split["Validation"], by_split["Test"], seed=s)
        all_p_te.append(r["p_te"]); all_p_va.append(r["p_va"])
        y_te = r["y_te"]; y_va = r["y_va"]
        seed_results.append({"seed": s, "history": r["history"]})
        # Intermediate save
        partial = {"per_seed": seed_results, "ensemble": None}
        with open(OUT.replace(".json", "_partial.json"), 'w') as f:
            json.dump(partial, f, indent=2)

    log("\n========== ENSEMBLE ==========")
    p_te = np.mean(all_p_te, axis=0); p_va = np.mean(all_p_va, axis=0)
    classes = np.arange(4, dtype=np.float32)
    e_va = (p_va * classes[None]).sum(1); e_te = (p_te * classes[None]).sum(1)
    bt = tune_thresholds(e_va, y_va)
    yp_arg = p_te.argmax(1); yp_thr = apply_thr(e_te, bt['t'])
    kq_arg = float(cohen_kappa_score(y_te, yp_arg, weights="quadratic"))
    kq_thr = float(cohen_kappa_score(y_te, yp_thr, weights="quadratic"))
    acc_thr = float(accuracy_score(y_te, yp_thr))
    ci_arg = boot_ci(y_te, yp_arg); ci_thr = boot_ci(y_te, yp_thr)
    log(f"ENSEMBLE test_arg κ={kq_arg:.4f} {ci_arg}  test_thr κ={kq_thr:.4f} {ci_thr} acc={acc_thr:.3f} t={bt['t']}")

    out = {"per_seed": seed_results,
           "ensemble": {"argmax": {"kq": kq_arg, "ci": ci_arg},
                        "threshold": {"kq": kq_thr, "ci": ci_thr, "acc": acc_thr, "t": bt['t']}}}
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
