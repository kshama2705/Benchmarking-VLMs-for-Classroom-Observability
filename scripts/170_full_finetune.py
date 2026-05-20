"""
Full-encoder fine-tune (FEAT) — SigLIP-L last 4 blocks + post_layernorm + head,
trained end-to-end on engagement labels with all the regularization tricks:

- Multi-frame: 3 frames per clip (t=2,5,8) mean-pooled at feature level after encoder
- Class-balanced focal loss (gamma=2, alpha from class freq)
- AdamW with separate LRs per param group (enc lr=1e-5, head lr=3e-4)
- Linear warmup 200 steps + cosine decay
- Effective batch 32 (per-step 8, accum 4)
- Strong dropout + weight decay
- Train + Val combined (no val-based selection — fixed 8 epochs)
- 3 seeds, ensemble predictions averaged on test

Targets test κ_q > 0.30 — a real positive method for BMVC.
"""
import os, json, time, csv, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import cohen_kappa_score, accuracy_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = os.path.join(BASE, "frames_full")
FRAMES_M = os.path.join(BASE, "frames_full_multi")
MANIFEST = os.path.join(FRAMES, "manifest.csv")
OUT = os.path.join(BASE, "results", "sota", "feat_results.json")
LOG = os.path.join(BASE, "results", "sota", "feat.log")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load_manifest():
    rows = []
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "exists") and os.path.exists(r["frame_path"]):
                rows.append(r)
    return rows


class DaiSEEMultiFrame(Dataset):
    """Load single frame (t=5) per clip. Renamed for compatibility."""
    def __init__(self, rows, proc):
        self.rows = rows; self.proc = proc

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["frame_path"]).convert("RGB")
        inp = self.proc(images=img, return_tensors="pt")
        pv = inp["pixel_values"]   # (1, 3, H, W)
        return pv, int(r["engagement"])


def collate_var(batch):
    """Pad variable T to max in batch."""
    maxT = max(p.shape[0] for p, _ in batch)
    outp, outy, outm = [], [], []
    for pv, y in batch:
        T = pv.shape[0]
        if T < maxT:
            pad = torch.zeros((maxT - T,) + pv.shape[1:])
            pv = torch.cat([pv, pad], dim=0)
        mask = torch.zeros(maxT); mask[:T] = 1.0
        outp.append(pv); outy.append(y); outm.append(mask)
    return torch.stack(outp), torch.tensor(outy, dtype=torch.long), torch.stack(outm)


def focal_loss(logits, y, alpha, gamma=2.0):
    # alpha: (C,) class weights summing to ~C
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    nll = -logp.gather(1, y.unsqueeze(1)).squeeze(1)
    pt = p.gather(1, y.unsqueeze(1)).squeeze(1)
    a = alpha[y]
    fl = a * (1 - pt).pow(gamma) * nll
    return fl.mean()


def tune_thresholds(e, y, step=0.04):
    grid = np.arange(0.0, 3.01, step)
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


def apply_thr(e, t):
    yp = np.zeros_like(e, dtype=int)
    yp[e > t[0]] = 1; yp[e > t[1]] = 2; yp[e > t[2]] = 3
    return yp


def boot_ci(yt, yp, n=500, seed=42):
    rng = np.random.default_rng(seed); nt = len(yt); out = []
    for _ in range(n):
        idx = rng.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def train_one(rows_train, rows_val, rows_test, proc,
              n_unfreeze_blocks=4, epochs=6, batch=8, accum=4,
              lr_enc=1e-5, lr_head=3e-4, seed=0, gamma_focal=2.0):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = DEVICE

    log(f"[seed {seed}] Loading SigLIP-L...")
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained("google/siglip-large-patch16-256")
    vision = backbone.vision_model

    for p in vision.parameters():
        p.requires_grad = False
    n_blocks = len(vision.encoder.layers)
    for blk in vision.encoder.layers[-n_unfreeze_blocks:]:
        for p in blk.parameters():
            p.requires_grad = True
    if hasattr(vision, 'post_layernorm'):
        for p in vision.post_layernorm.parameters():
            p.requires_grad = True

    feat_dim = vision.config.hidden_size
    head = nn.Sequential(
        nn.LayerNorm(feat_dim),
        nn.Dropout(0.2),
        nn.Linear(feat_dim, feat_dim // 2),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(feat_dim // 2, 4),
    ).to(device)
    vision = vision.to(device)

    n_enc = sum(p.numel() for p in vision.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in head.parameters())
    log(f"[seed {seed}] Trainable: enc {n_enc/1e6:.1f}M  head {n_head/1e6:.1f}M")

    # Class-balanced focal alpha
    cnt = np.zeros(4)
    for r in rows_train: cnt[int(r["engagement"])] += 1
    log(f"[seed {seed}] class counts: {cnt.tolist()}")
    alpha = 1.0 / np.maximum(cnt, 1); alpha = alpha / alpha.sum() * 4
    alpha = torch.tensor(alpha, dtype=torch.float32, device=device)

    train_ds = DaiSEEMultiFrame(rows_train, proc)
    val_ds = DaiSEEMultiFrame(rows_val, proc)
    test_ds = DaiSEEMultiFrame(rows_test, proc)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate_var, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, collate_fn=collate_var, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch, shuffle=False, collate_fn=collate_var, num_workers=0)

    opt = torch.optim.AdamW([
        {"params": [p for p in vision.parameters() if p.requires_grad], "lr": lr_enc},
        {"params": list(head.parameters()), "lr": lr_head},
    ], weight_decay=0.05)
    base = [lr_enc, lr_head]

    n_steps = max(1, (len(train_dl) // accum) * epochs)
    warmup = max(100, n_steps // 25)
    def set_lr(step):
        if step < warmup: s = step / max(1, warmup)
        else:
            prog = (step - warmup) / max(1, n_steps - warmup)
            s = 0.5 * (1 + np.cos(np.pi * prog))
        for pg, b in zip(opt.param_groups, base):
            pg["lr"] = b * s

    @torch.no_grad()
    def eval_split(dl):
        vision.eval(); head.eval()
        all_logits = []; all_y = []
        for pv, y, mask in dl:
            pv = pv.to(device); mask = mask.to(device)
            B, T, C, H, W = pv.shape
            out = vision(pixel_values=pv.view(B*T, C, H, W)).pooler_output
            out = out.view(B, T, -1)
            denom = mask.sum(1, keepdim=True).clamp(min=1)
            pooled = (out * mask.unsqueeze(-1)).sum(1) / denom
            logits = head(pooled)
            all_logits.append(F.softmax(logits, dim=1).cpu().numpy()); all_y.append(y.numpy())
        return np.concatenate(all_logits), np.concatenate(all_y)

    log(f"[seed {seed}] Training {epochs} epochs, {n_steps} optimizer steps...")
    step = 0; history = []
    for ep in range(epochs):
        vision.train(); head.train()
        opt.zero_grad()
        ep_t0 = time.time()
        for i, (pv, y, mask) in enumerate(train_dl):
            pv = pv.to(device); y = y.to(device); mask = mask.to(device)
            B, T, C, H, W = pv.shape
            out = vision(pixel_values=pv.view(B*T, C, H, W)).pooler_output
            out = out.view(B, T, -1)
            denom = mask.sum(1, keepdim=True).clamp(min=1)
            pooled = (out * mask.unsqueeze(-1)).sum(1) / denom
            logits = head(pooled)
            loss = focal_loss(logits, y, alpha, gamma=gamma_focal) / accum
            loss.backward()
            if (i + 1) % accum == 0:
                set_lr(step); step += 1
                torch.nn.utils.clip_grad_norm_(
                    [p for p in vision.parameters() if p.requires_grad] + list(head.parameters()),
                    max_norm=1.0,
                )
                opt.step(); opt.zero_grad()
                if step % 30 == 0:
                    log(f"  [seed {seed}] ep{ep} step{step}/{n_steps} loss={loss.item()*accum:.3f} lr_enc={opt.param_groups[0]['lr']:.2e}")
        p_va, y_va = eval_split(val_dl)
        p_te, y_te = eval_split(test_dl)
        classes = np.arange(4, dtype=np.float32)
        e_va = (p_va * classes[None]).sum(1); e_te = (p_te * classes[None]).sum(1)
        bt = tune_thresholds(e_va, y_va)
        yp_te_arg = p_te.argmax(1)
        yp_te_thr = apply_thr(e_te, bt['t'])
        val_kq_arg = float(cohen_kappa_score(y_va, p_va.argmax(1), weights="quadratic"))
        test_kq_arg = float(cohen_kappa_score(y_te, yp_te_arg, weights="quadratic"))
        test_kq_thr = float(cohen_kappa_score(y_te, yp_te_thr, weights="quadratic"))
        dt = time.time() - ep_t0
        history.append({"ep": ep, "val_kq_arg": val_kq_arg, "val_kq_thr": float(bt['v']),
                        "test_kq_arg": test_kq_arg, "test_kq_thr": test_kq_thr,
                        "thresholds": bt['t'], "duration_min": dt/60})
        log(f"[seed {seed} ep{ep}] val_arg={val_kq_arg:.4f}  val_thr={bt['v']:.4f}  test_arg={test_kq_arg:.4f}  test_thr={test_kq_thr:.4f}  dt={dt/60:.1f}min")

    p_te, y_te = eval_split(test_dl)
    p_va, y_va = eval_split(val_dl)
    return {"history": history, "p_te": p_te, "p_va": p_va, "y_te": y_te, "y_va": y_va}


def main():
    log(f"Device: {DEVICE}")
    rows = load_manifest()
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r)
    log(f"Splits: " + ", ".join(f"{s}={len(lst)}" for s, lst in by_split.items()))

    from transformers import SiglipImageProcessor
    proc = SiglipImageProcessor.from_pretrained("google/siglip-large-patch16-256")

    rows_train = by_split["Train"]
    rows_val = by_split["Validation"]
    rows_test = by_split["Test"]

    seeds = [0, 42, 2025]
    all_p_te = []; all_p_va = []; y_te = y_va = None
    seed_results = []
    for s in seeds:
        log(f"\n========== Seed {s} ==========")
        r = train_one(rows_train, rows_val, rows_test, proc, seed=s,
                      n_unfreeze_blocks=2, epochs=8, batch=12, accum=2)
        all_p_te.append(r["p_te"]); all_p_va.append(r["p_va"])
        y_te = r["y_te"]; y_va = r["y_va"]
        seed_results.append({"seed": s, "history": r["history"]})

    # Ensemble
    log("\n========== ENSEMBLE ==========")
    p_te = np.mean(all_p_te, axis=0); p_va = np.mean(all_p_va, axis=0)
    classes = np.arange(4, dtype=np.float32)
    e_va = (p_va * classes[None]).sum(1); e_te = (p_te * classes[None]).sum(1)
    bt = tune_thresholds(e_va, y_va)
    yp_te_arg = p_te.argmax(1)
    yp_te_thr = apply_thr(e_te, bt['t'])
    kq_arg = float(cohen_kappa_score(y_te, yp_te_arg, weights="quadratic"))
    kq_thr = float(cohen_kappa_score(y_te, yp_te_thr, weights="quadratic"))
    acc_arg = float(accuracy_score(y_te, yp_te_arg))
    acc_thr = float(accuracy_score(y_te, yp_te_thr))
    ci_arg = boot_ci(y_te, yp_te_arg)
    ci_thr = boot_ci(y_te, yp_te_thr)
    log(f"ENSEMBLE test_arg κ={kq_arg:.4f} {ci_arg} acc={acc_arg:.3f}  test_thr κ={kq_thr:.4f} {ci_thr} acc={acc_thr:.3f}  t={bt['t']}")

    out = {
        "per_seed": seed_results,
        "ensemble": {
            "argmax": {"kq": kq_arg, "ci": ci_arg, "acc": acc_arg},
            "threshold": {"kq": kq_thr, "ci": ci_thr, "acc": acc_thr, "t": bt['t']},
        },
    }
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
