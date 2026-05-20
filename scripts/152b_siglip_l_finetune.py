"""
Plan B — stable last-block fine-tune of SigLIP-L for engagement.

Single-frame (t=5s) input. Unfreezes only the LAST transformer block +
post_layernorm of the vision encoder + a new classification head.

Training:
- Low LR (3e-5 for encoder, 5e-4 for head)
- Linear warmup 100 steps, then cosine decay
- AdamW, weight_decay=0.01
- Class-balanced cross-entropy
- Batch=16, gradient accumulation = 4 (effective batch 64)
- 6 epochs, save best val κ_q

Inputs: frames at frames_full/<split>/<clip_id>.jpg (via manifest.csv).
Outputs: results/sota/siglip_l_finetune.json + saved best weights.
"""

import os, json, time, random, csv
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import cohen_kappa_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_BASE = os.path.join(BASE, "frames_full")
MANIFEST = os.path.join(FRAMES_BASE, "manifest.csv")
OUT_JSON = os.path.join(BASE, "results", "sota", "siglip_l_finetune.json")
OUT_LOG = os.path.join(BASE, "results", "sota", "siglip_l_finetune.log")
BEST_W = os.path.join(BASE, "models", "siglip_l_finetune_best.pt")
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
os.makedirs(os.path.dirname(BEST_W), exist_ok=True)

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT_LOG, "a") as f:
        f.write(line + "\n")


def load_manifest():
    rows = []
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "exists") and os.path.exists(r["frame_path"]):
                rows.append(r)
    return rows


class DaiSEEFrames(Dataset):
    def __init__(self, rows, processor):
        self.rows = rows
        self.proc = processor

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["frame_path"]).convert("RGB")
        inp = self.proc(images=img, return_tensors="pt")
        pv = inp["pixel_values"][0]   # (3, H, W)
        y = int(r["engagement"])
        return pv, y


def main(epochs=6, batch=16, accum=4, lr_head=5e-4, lr_enc=3e-5):
    log(f"Device: {DEVICE}")
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    rows = load_manifest()
    log(f"Total rows in manifest with extant frame: {len(rows)}")
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r)
    for s, lst in by_split.items():
        log(f"  {s}: {len(lst)}")

    log("Loading SigLIP-L...")
    from transformers import AutoModel, SiglipImageProcessor
    model_name = "google/siglip-large-patch16-256"
    proc = SiglipImageProcessor.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name)
    vision = backbone.vision_model

    # Freeze all
    for p in vision.parameters():
        p.requires_grad = False
    # Unfreeze last block + post_layernorm
    n_blocks = len(vision.encoder.layers)
    log(f"vision has {n_blocks} blocks; unfreezing last block + post_layernorm")
    for p in vision.encoder.layers[-1].parameters():
        p.requires_grad = True
    if hasattr(vision, 'post_layernorm') and vision.post_layernorm is not None:
        for p in vision.post_layernorm.parameters():
            p.requires_grad = True

    feat_dim = vision.config.hidden_size  # 1024 for L/16
    head = nn.Sequential(
        nn.LayerNorm(feat_dim),
        nn.Linear(feat_dim, feat_dim),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(feat_dim, 4),
    )

    vision = vision.to(DEVICE)
    head = head.to(DEVICE)

    n_trainable_enc = sum(p.numel() for p in vision.parameters() if p.requires_grad)
    n_trainable_head = sum(p.numel() for p in head.parameters())
    log(f"Trainable encoder params: {n_trainable_enc/1e6:.2f}M")
    log(f"Trainable head params:    {n_trainable_head/1e6:.2f}M")

    # Class-balanced weights from Train split
    counts = np.zeros(4, dtype=np.float32)
    for r in by_split["Train"]:
        counts[int(r["engagement"])] += 1
    w = (1.0 / np.maximum(counts, 1)); w = w / w.sum() * 4
    log(f"Class counts: {counts.tolist()}  weights: {w.tolist()}")
    ce = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEVICE, dtype=torch.float32))

    enc_params = [p for p in vision.parameters() if p.requires_grad]
    head_params = list(head.parameters())
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": lr_enc, "name": "enc"},
         {"params": head_params, "lr": lr_head, "name": "head"}],
        weight_decay=0.01,
    )
    base_lrs = [lr_enc, lr_head]

    train_ds = DaiSEEFrames(by_split["Train"], proc)
    val_ds = DaiSEEFrames(by_split["Validation"], proc)
    test_ds = DaiSEEFrames(by_split["Test"], proc)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=0)

    n_steps = max(1, (len(train_dl) // accum) * epochs)
    warmup = max(50, n_steps // 30)

    def lr_scale(step):
        if step < warmup: return step / max(1, warmup)
        prog = (step - warmup) / max(1, n_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * prog))

    def set_lrs(step):
        s = lr_scale(step)
        for pg, base in zip(opt.param_groups, base_lrs):
            pg["lr"] = base * s

    @torch.no_grad()
    def eval_split(dl, tag):
        vision.eval(); head.eval()
        all_logits = []; all_y = []
        for pv, y in dl:
            pv = pv.to(DEVICE)
            out = vision(pixel_values=pv)
            pooled = out.pooler_output  # (B, D)
            logits = head(pooled)
            all_logits.append(logits.cpu().numpy()); all_y.append(y.numpy())
        logits = np.concatenate(all_logits); y_arr = np.concatenate(all_y)
        yp = logits.argmax(1)
        kq = float(cohen_kappa_score(y_arr, yp, weights="quadratic"))
        return kq, logits, y_arr

    log("Starting training...")
    step = 0; best_val_kq = -1; best = None
    history = []

    for ep in range(epochs):
        vision.train(); head.train()
        opt.zero_grad()
        ep_t0 = time.time()
        for i, (pv, y) in enumerate(train_dl):
            pv = pv.to(DEVICE); y = y.to(DEVICE)
            out = vision(pixel_values=pv)
            pooled = out.pooler_output
            logits = head(pooled)
            loss = ce(logits, y) / accum
            loss.backward()
            if (i + 1) % accum == 0:
                set_lrs(step)
                opt.step()
                opt.zero_grad()
                step += 1
                if step % 20 == 0:
                    log(f"  ep{ep} step{step}/{n_steps} loss={loss.item()*accum:.4f} lr_enc={opt.param_groups[0]['lr']:.2e}")
        # End-of-epoch eval
        val_kq, _, _ = eval_split(val_dl, "val")
        test_kq, _, _ = eval_split(test_dl, "test")
        dt = time.time() - ep_t0
        history.append({"epoch": ep, "val_kq": val_kq, "test_kq": test_kq, "duration_s": dt})
        log(f"[ep {ep}] val_kq={val_kq:.4f}  test_kq={test_kq:.4f}  dt={dt/60:.1f}min")
        if val_kq > best_val_kq:
            best_val_kq = val_kq
            best = {"ep": ep, "val_kq": val_kq, "test_kq": test_kq}
            torch.save({"vision": vision.state_dict(), "head": head.state_dict(),
                        "config": {"lr_enc": lr_enc, "lr_head": lr_head, "batch": batch,
                                   "accum": accum, "epochs": epochs}}, BEST_W)
            log(f"  saved best checkpoint @ ep{ep}")

    log(f"FINAL best val={best['val_kq']:.4f}  test={best['test_kq']:.4f} @ ep{best['ep']}")
    with open(OUT_JSON, 'w') as f:
        json.dump({"best": best, "history": history,
                   "config": {"epochs": epochs, "batch": batch, "accum": accum,
                              "lr_head": lr_head, "lr_enc": lr_enc}}, f, indent=2)
    log(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
