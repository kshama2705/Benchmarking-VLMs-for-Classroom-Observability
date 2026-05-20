"""
LoRA fine-tuning of SigLIP-L on DAiSEE engagement.

Architecture:
  - Frozen SigLIP-L visual encoder (24 transformer blocks)
  - LoRA adapters (rank=16) injected into last 2 blocks' qkv and proj layers
    (4 LoRA modules per block × 2 blocks = 8 LoRA pairs)
  - Linear classification head on top of pooled CLS-like feature

Trainable parameters:
  - LoRA: rank × (d_in + d_out) per matrix
    qkv: 16 × (1024 + 3072) = 65,536 params
    proj: 16 × (1024 + 1024) = 32,768 params
    per block: ~98K; 2 blocks: ~200K
  - Classifier head: 1024 × 4 + 4 = 4,100 params
  Total: ~200K trainable params (vs 350M frozen)

Training: class-balanced cross-entropy, AdamW, multi-seed.

Output:
  results/sota/lora_siglip_l_results.json
"""
import os, csv, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import open_clip
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(BASE, "frames_full", "manifest.csv")
OUT = os.path.join(BASE, "results", "sota", "lora_siglip_l_results.json")
LOG = os.path.join(BASE, "results", "sota", "lora_status.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))

SEEDS = [42, 7, 2025]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


# ============ LoRA Linear ============
class LoRALinear(nn.Module):
    """Wraps a Linear with LoRA adapter."""
    def __init__(self, base_linear, rank=16, alpha=32):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        # Frozen base
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        # LoRA
        self.lora_A = nn.Parameter(torch.randn(rank, self.in_features) * (1.0 / np.sqrt(self.in_features)))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.scale = alpha / rank
        self.rank = rank

    def forward(self, x):
        base_out = self.base(x)
        lora = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + lora * self.scale


def inject_lora(model, block_indices=(22, 23), rank=16, alpha=32):
    """Replace qkv and proj in attention of selected blocks with LoRA versions."""
    n_replaced = 0
    for idx in block_indices:
        block = model.visual.trunk.blocks[idx]
        # Replace qkv
        if hasattr(block.attn, "qkv") and isinstance(block.attn.qkv, nn.Linear):
            block.attn.qkv = LoRALinear(block.attn.qkv, rank=rank, alpha=alpha).to(DEVICE)
            n_replaced += 1
        # Replace proj
        if hasattr(block.attn, "proj") and isinstance(block.attn.proj, nn.Linear):
            block.attn.proj = LoRALinear(block.attn.proj, rank=rank, alpha=alpha).to(DEVICE)
            n_replaced += 1
    return n_replaced


# ============ Dataset ============
class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, rows, preprocess):
        self.rows = rows
        self.pp = preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["frame_path"]).convert("RGB")
        return self.pp(img), int(r["engagement"])


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


def train_one(rows_tr, rows_va, rows_te, preprocess, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    log(f"  seed={seed}")
    # Load fresh SigLIP-L
    model, _, _ = open_clip.create_model_and_transforms("ViT-L-16-SigLIP-256", pretrained="webli")
    model = model.to(DEVICE).eval()
    # Freeze all params
    for p in model.parameters():
        p.requires_grad = False
    # Inject LoRA in last N blocks
    n_lora = inject_lora(model, block_indices=cfg["lora_blocks"], rank=cfg["rank"], alpha=cfg["alpha"])
    # Classifier head
    head = nn.Linear(1024, 4).to(DEVICE)

    # Count trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in head.parameters())
    log(f"    {n_lora} LoRA modules, trainable params: {trainable:,}")

    # Optimizer (only LoRA + head)
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=1e-4)

    yt_tr = np.array([int(r["engagement"]) for r in rows_tr])
    yt_va = np.array([int(r["engagement"]) for r in rows_va])
    yt_te = np.array([int(r["engagement"]) for r in rows_te])
    eng_w = class_weights(yt_tr, 4)
    eng_loss_fn = nn.CrossEntropyLoss(weight=eng_w)

    train_ds = FrameDataset(rows_tr, preprocess)
    val_ds = FrameDataset(rows_va, preprocess)
    test_ds = FrameDataset(rows_te, preprocess)
    BATCH = cfg["batch_size"]
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=2)

    best = {"val_kq": -1e9, "epoch": -1, "state": None}
    for ep in range(cfg["epochs"]):
        # Train
        model.eval()  # base is eval; LoRA modules will still update via grad
        head.train()
        for p in params:
            p.requires_grad = True  # safety
        for xb, yb in train_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            feat = model.encode_image(xb)  # (B, 1024)
            # Normalize? Open_clip normalizes by default; but we want the unprojected one. encode_image typically returns the projected/normalized.
            # Actually open_clip returns projected features. For our task we use as-is.
            logits = head(feat)
            loss = eng_loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        # Validate
        with torch.no_grad():
            preds = []
            for xb, _ in val_loader:
                xb = xb.to(DEVICE)
                feat = model.encode_image(xb)
                preds.append(head(feat).argmax(dim=1).cpu().numpy())
            yhat_va = np.concatenate(preds)
            val_kq = float(cohen_kappa_score(yt_va, yhat_va, weights="quadratic"))
        if val_kq > best["val_kq"]:
            best["val_kq"] = val_kq
            best["epoch"] = ep
            best["state"] = ({k: v.detach().cpu().clone() for k, v in model.state_dict().items() if any(t in k for t in ['lora_A', 'lora_B'])},
                             {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
        log(f"    ep{ep:2d}  val κ_q={val_kq:.3f}  (best={best['val_kq']:.3f}@ep{best['epoch']})")

    # Load best
    if best["state"] is not None:
        lora_state, head_state = best["state"]
        model.load_state_dict({**model.state_dict(), **{k: v.to(DEVICE) for k, v in lora_state.items()}})
        head.load_state_dict({k: v.to(DEVICE) for k, v in head_state.items()})

    # Test
    model.eval(); head.eval()
    with torch.no_grad():
        preds = []
        for xb, _ in test_loader:
            xb = xb.to(DEVICE)
            feat = model.encode_image(xb)
            preds.append(head(feat).argmax(dim=1).cpu().numpy())
        yhat_te = np.concatenate(preds)
    m = metrics(yt_te, yhat_te)
    log(f"  seed={seed}  test κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}  acc={m['accuracy']:.3f}")
    return m, best["val_kq"]


def main():
    open(LOG, "w").close()
    rows = []
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "exists") and os.path.exists(r["frame_path"]):
                rows.append(r)
    rows_tr = [r for r in rows if r["split"] == "Train"]
    rows_va = [r for r in rows if r["split"] == "Validation"]
    rows_te = [r for r in rows if r["split"] == "Test"]
    log(f"Train={len(rows_tr)} Val={len(rows_va)} Test={len(rows_te)}")

    # Get preprocess by loading model once
    _, _, preprocess = open_clip.create_model_and_transforms("ViT-L-16-SigLIP-256", pretrained="webli")

    cfg = {
        "lora_blocks": (22, 23),  # last 2 of 24
        "rank": 16,
        "alpha": 32,
        "lr": 1e-4,
        "batch_size": 16,
        "epochs": 10,
    }
    log(f"Config: {cfg}")

    out = {"cfg": cfg, "per_seed": []}
    for seed in SEEDS:
        m, val_kq = train_one(rows_tr, rows_va, rows_te, preprocess, cfg, seed)
        out["per_seed"].append({"seed": seed, **m, "val_kq": val_kq})

    kappas = np.array([r["kappa_q"] for r in out["per_seed"]])
    out["mean_kq"] = float(kappas.mean())
    out["std_kq"] = float(kappas.std(ddof=1) if len(kappas) > 1 else 0.0)
    out["min_kq"] = float(kappas.min())
    out["max_kq"] = float(kappas.max())

    log(f"\nFINAL: mean κ_q = {out['mean_kq']:.3f} ± {out['std_kq']:.3f}  "
        f"(min={out['min_kq']:.3f}, max={out['max_kq']:.3f})")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {OUT}")
    log("LORA_DONE")


if __name__ == "__main__":
    main()
