"""
Plan B — Stable last-block fine-tune of SigLIP-L for engagement.

Unfreezes only the LAST transformer block + LayerNorm + classification head
of SigLIP-L. Trains with:
- Low LR (1e-5 for encoder, 3e-4 for head)
- Linear warmup over 200 steps
- Cosine decay over remaining
- Class-balanced cross-entropy
- AdamW, weight_decay=0.01
- Gradient accumulation: effective batch 64 (per-step batch 16)
- Multi-frame: 3-frame mean per clip (t=2,5,8)
- 5 epochs (long enough; LoRA prior attempt killed at ep1)

Per-frame inputs are precomputed at 224x224 (DAiSEE faces) and stored;
if not present, we encode on the fly.

If this lands κ_q > 0.30 it's the BMVC headline; if it plateaus near the
frozen ceiling, it's another data point confirming the structural story.

NOTE: requires raw frames, not just features. We use the existing frames
at frames_full/<split>/<subject>/<clip>/t5.jpg.
"""

import os, json, time, random
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import cohen_kappa_score

# Imported lazily so that the module can be inspected without HF deps
def _load_siglip():
    from transformers import AutoModel, AutoProcessor
    name = "google/siglip-large-patch16-256"
    model = AutoModel.from_pretrained(name)
    proc = AutoProcessor.from_pretrained(name)
    return model, proc

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = os.path.join(BASE, "frames_full")
SIGLIP_FEATURES = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
OUT = os.path.join(BASE, "results", "sota", "siglip_l_finetune.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))


class DaiSEEFrameDataset(Dataset):
    def __init__(self, clip_ids, subjects, labels, splits, proc):
        self.clip_ids = clip_ids
        self.subjects = subjects
        self.labels = labels
        self.splits = splits
        self.proc = proc

    def __len__(self): return len(self.clip_ids)

    def _frame_path(self, i, t='t5'):
        cid = self.clip_ids[i]; split = self.splits[i]; sub = self.subjects[i]
        return os.path.join(FRAMES, split, sub, cid, f"{t}.jpg")

    def __getitem__(self, i):
        # mean of 3 frames if available
        imgs = []
        for t in ('t2', 't5', 't8'):
            p = self._frame_path(i, t=t)
            if os.path.exists(p):
                imgs.append(Image.open(p).convert("RGB"))
        if len(imgs) == 0:
            # Fallback
            imgs = [Image.new("RGB", (224, 224), color=(128,128,128))]
        # processor returns dict of pixel_values
        inp = self.proc(images=imgs, return_tensors="pt")
        pv = inp["pixel_values"]   # (T, 3, H, W)
        return pv, int(self.labels[i])


def collate(batch):
    # batch: list of (pv, y) with variable T
    # pad to max T in batch
    maxT = max(p.shape[0] for p, _ in batch)
    out_p = []; out_y = []; out_m = []
    for pv, y in batch:
        T = pv.shape[0]
        pad = torch.zeros((maxT - T,) + pv.shape[1:]) if T < maxT else None
        if pad is not None:
            pv = torch.cat([pv, pad], dim=0)
        mask = torch.zeros(maxT); mask[:T] = 1.0
        out_p.append(pv); out_y.append(y); out_m.append(mask)
    return torch.stack(out_p), torch.tensor(out_y, dtype=torch.long), torch.stack(out_m)


def main(epochs=5, batch=16, accum=4, lr_head=3e-4, lr_enc=1e-5):
    print(f"Device: {DEVICE}", flush=True)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    d = np.load(SIGLIP_FEATURES, allow_pickle=True)
    clip_ids = d["clip_id"]; subjects = d["subject_id"]
    splits = d["split"]; labels = d["engagement"]

    tr_idx = np.where(splits == "Train")[0]
    va_idx = np.where(splits == "Validation")[0]
    te_idx = np.where(splits == "Test")[0]
    print(f"tr={len(tr_idx)}  va={len(va_idx)}  te={len(te_idx)}", flush=True)

    print("Loading SigLIP-L...", flush=True)
    model, proc = _load_siglip()
    vision = model.vision_model  # has embeddings + encoder + post_layernorm + head
    # Freeze everything
    for p in vision.parameters():
        p.requires_grad = False
    # Unfreeze last block + post_layernorm + head
    n_blocks = len(vision.encoder.layers)
    for p in vision.encoder.layers[n_blocks - 1].parameters():
        p.requires_grad = True
    for p in vision.post_layernorm.parameters():
        p.requires_grad = True
    if hasattr(vision, 'head') and vision.head is not None:
        for p in vision.head.parameters():
            p.requires_grad = True

    feat_dim = vision.config.hidden_size  # 1024 for SigLIP-L
    head = nn.Sequential(
        nn.LayerNorm(feat_dim),
        nn.Linear(feat_dim, feat_dim),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(feat_dim, 4),
    ).to(DEVICE)

    model = vision.to(DEVICE)
    # Class-balanced loss
    counts = np.bincount(labels[tr_idx].astype(int), minlength=4).astype(np.float32)
    w = (1.0 / np.maximum(counts, 1)); w = w / w.sum() * 4
    ce = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEVICE, dtype=torch.float32))

    enc_params = [p for p in model.parameters() if p.requires_grad]
    head_params = list(head.parameters())
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": lr_enc},
         {"params": head_params, "lr": lr_head}],
        weight_decay=0.01,
    )

    train_ds = DaiSEEFrameDataset(clip_ids[tr_idx], subjects[tr_idx], labels[tr_idx], splits[tr_idx], proc)
    val_ds = DaiSEEFrameDataset(clip_ids[va_idx], subjects[va_idx], labels[va_idx], splits[va_idx], proc)
    test_ds = DaiSEEFrameDataset(clip_ids[te_idx], subjects[te_idx], labels[te_idx], splits[te_idx], proc)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, collate_fn=collate, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch, shuffle=False, collate_fn=collate, num_workers=0)

    n_steps = len(train_dl) * epochs // accum
    warmup = max(1, n_steps // 20)

    def lr_scale(step):
        if step < warmup: return step / max(1, warmup)
        prog = (step - warmup) / max(1, n_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * prog))

    @torch.no_grad()
    def eval_split(dl):
        model.eval(); head.eval()
        preds = []; ys = []
        for pv, y, mask in dl:
            pv = pv.to(DEVICE); mask = mask.to(DEVICE)
            B, T, C, H, W = pv.shape
            out = model(pixel_values=pv.view(B*T, C, H, W)).pooler_output  # (B*T, D)
            out = out.view(B, T, -1)
            denom = mask.sum(1, keepdim=True).clamp(min=1)
            pooled = (out * mask.unsqueeze(-1)).sum(1) / denom
            logits = head(pooled)
            preds.append(logits.argmax(1).cpu().numpy())
            ys.append(y.numpy())
        yp = np.concatenate(preds); yt = np.concatenate(ys)
        return float(cohen_kappa_score(yt, yp, weights="quadratic")), yp, yt

    print("Training...", flush=True)
    step = 0; best_val_kq = -1; best_test = None
    history = []
    for ep in range(epochs):
        model.train(); head.train()
        opt.zero_grad()
        for i, (pv, y, mask) in enumerate(train_dl):
            pv = pv.to(DEVICE); y = y.to(DEVICE); mask = mask.to(DEVICE)
            B, T, C, H, W = pv.shape
            out = model(pixel_values=pv.view(B*T, C, H, W)).pooler_output
            out = out.view(B, T, -1)
            denom = mask.sum(1, keepdim=True).clamp(min=1)
            pooled = (out * mask.unsqueeze(-1)).sum(1) / denom
            logits = head(pooled)
            loss = ce(logits, y) / accum
            loss.backward()
            if (i + 1) % accum == 0:
                # apply scheduled LR
                for pg in opt.param_groups:
                    pg["lr"] *= lr_scale(step) / max(1e-9, lr_scale(max(0, step-1)))
                opt.step()
                opt.zero_grad()
                step += 1
        # Eval each epoch
        val_kq, _, _ = eval_split(val_dl)
        test_kq, yp_te, yt_te = eval_split(test_dl)
        history.append({"epoch": ep, "val_kq": val_kq, "test_kq": test_kq})
        print(f"[ep {ep}] val_kq={val_kq:.4f}  test_kq={test_kq:.4f}", flush=True)
        if val_kq > best_val_kq:
            best_val_kq = val_kq
            best_test = test_kq
    print(f"BEST val={best_val_kq:.4f}  test={best_test:.4f}", flush=True)
    out = {"best_val_kq": best_val_kq, "best_test_kq": best_test, "history": history,
            "config": {"epochs": epochs, "batch": batch, "accum": accum,
                       "lr_head": lr_head, "lr_enc": lr_enc}}
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
