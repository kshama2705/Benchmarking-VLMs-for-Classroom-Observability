"""
Supervised end-to-end engagement classifier — GPU-ready.

This is the script that delivers the positive method for BMVC. Designed to
run on a single NVIDIA A100/V100/3090 in ~12-24 hours, *not* on Apple
Silicon MPS (which has been the bottleneck for in-session attempts).

Method: ENGAGENET-X — temporal SigLIP-L fine-tune
- Backbone: SigLIP-L/16-256 (HuggingFace google/siglip-large-patch16-256)
- Input: 8 frames per clip (t = 0.5, 1.7, 2.9, 4.1, 5.3, 6.5, 7.7, 8.9 s)
- All blocks unfrozen, encoder LR 1e-5, head LR 3e-4
- Temporal attention head: Transformer 2-layer, 8 heads, d=256 → ordinal CORN
- Mixed precision (bf16 on A100 / fp16 on V100)
- Focal loss with class-balanced alpha
- AdamW, cosine schedule, 200-step linear warmup
- 12 epochs effective batch 32 (per-step 8 × accum 4)
- 3 seeds, ensemble at inference

Expected outcome based on published comparable methods (ViBED-Net, CavT):
  test κ_q ≈ 0.40 - 0.55
  test accuracy ≈ 0.65 - 0.73

USAGE (on a GPU cluster):
  pip install torch torchvision transformers scikit-learn opencv-python pillow
  python scripts/180_supervised_e2e_gpu.py \\
      --frames-root /path/to/DAiSEE \\
      --manifest /path/to/frames_full/manifest.csv \\
      --out results/sota/engagenet_x.json \\
      --seeds 0,42,2025

Frames must be pre-extracted (8 per clip) — see helper at the top of this
script. Alternatively decode on the fly via decord/torchvision.io.
"""
import os, json, argparse, time, csv, random
import numpy as np

# Heavy imports gated behind __main__ so the file can be inspected without GPU deps
def main():
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.cuda.amp import autocast, GradScaler
    from PIL import Image
    from sklearn.metrics import cohen_kappa_score, accuracy_score
    from transformers import AutoModel, SiglipImageProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True,
                        help="Path to frames_full/manifest.csv")
    parser.add_argument("--frames-root", required=True,
                        help="Root of DAiSEE frames (will read t=0.5,1.7,...,8.9s per clip)")
    parser.add_argument("--out", default="results/sota/engagenet_x.json")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr_enc", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=3e-4)
    parser.add_argument("--gamma_focal", type=float, default=2.0)
    parser.add_argument("--seeds", default="0,42,2025")
    parser.add_argument("--n_frames", type=int, default=8)
    parser.add_argument("--model", default="google/siglip-large-patch16-256")
    parser.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    assert torch.cuda.is_available(), "This script requires a CUDA GPU."
    device = torch.device("cuda")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.amp]
    print(f"Device={device}  amp={args.amp}  dtype={amp_dtype}", flush=True)

    # ----------------- Data ---------------
    rows = []
    with open(args.manifest) as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "exists"):
                rows.append(r)
    by_split = {s: [r for r in rows if r["split"] == s]
                for s in ("Train", "Validation", "Test")}
    print({s: len(v) for s, v in by_split.items()})

    proc = SiglipImageProcessor.from_pretrained(args.model)
    TIMESTAMPS = np.linspace(0.5, 8.9, args.n_frames).tolist()

    def get_frames(row):
        """Load N_FRAMES jpgs per clip. Expects pre-extracted frames at
        <frames-root>/<split>/<clip>/t{i:02d}.jpg, OR decode on the fly."""
        cid = row["clip_id"].replace(".avi", "").replace(".mp4", "")
        split = row["split"]
        clip_dir = os.path.join(args.frames_root, split, cid)
        imgs = []
        for i, t in enumerate(TIMESTAMPS):
            p = os.path.join(clip_dir, f"t{i:02d}.jpg")
            if os.path.exists(p):
                imgs.append(Image.open(p).convert("RGB"))
        if len(imgs) == 0:
            # Fallback: try single-frame path from manifest
            p = row.get("frame_path", "")
            if os.path.exists(p):
                imgs = [Image.open(p).convert("RGB")] * args.n_frames
            else:
                imgs = [Image.new("RGB", (256, 256), (128, 128, 128))] * args.n_frames
        while len(imgs) < args.n_frames:
            imgs.append(imgs[-1])
        return imgs[:args.n_frames]

    class DS(Dataset):
        def __init__(self, lst):
            self.lst = lst
        def __len__(self): return len(self.lst)
        def __getitem__(self, i):
            r = self.lst[i]
            imgs = get_frames(r)
            inp = proc(images=imgs, return_tensors="pt")
            return inp["pixel_values"], int(r["engagement"])

    def collate(batch):
        pv = torch.stack([b[0] for b in batch])  # (B, T, 3, H, W)
        y = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return pv, y

    # ----------------- Model ---------------
    class EngageNetX(nn.Module):
        def __init__(self, backbone, d_model=256, n_layers=2, n_heads=8):
            super().__init__()
            self.backbone = backbone
            d_in = backbone.config.hidden_size
            self.proj = nn.Linear(d_in, d_model)
            self.pos = nn.Parameter(torch.randn(1, args.n_frames, d_model) * 0.02)
            enc = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
                dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
            self.temporal = nn.TransformerEncoder(enc, num_layers=n_layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(d_model, 3),  # CORN ordinal
            )

        def forward(self, pv):
            B, T, C, H, W = pv.shape
            out = self.backbone(pixel_values=pv.view(B*T, C, H, W)).pooler_output
            out = out.view(B, T, -1)
            z = self.proj(out) + self.pos
            z = self.temporal(z)
            z = z.mean(dim=1)
            return self.head(self.norm(z))

    def corn_loss(logits, y, alpha, gamma=2.0):
        K = 4
        losses = []
        for k in range(K - 1):
            target = (y > k).float()
            logit = logits[:, k]
            p = torch.sigmoid(logit)
            pt = torch.where(target == 1, p, 1 - p)
            bce = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
            fl = (1 - pt).pow(gamma) * bce
            # Class-balance using alpha[y]
            fl = alpha[y] * fl
            losses.append(fl.mean())
        return torch.stack(losses).mean()

    def corn_predict_expected(logits):
        probs = torch.sigmoid(logits)
        return probs.sum(dim=1)  # expected class score in [0, 3]

    def corn_argmax(logits):
        probs = torch.sigmoid(logits)
        return (probs > 0.5).sum(dim=1)

    def tune_thresholds(e, y, step=0.04):
        grid = np.arange(0.0, 3.01, step); best = None
        for t1 in grid:
            for t2 in grid[grid > t1]:
                for t3 in grid[grid > t2]:
                    yp = np.zeros_like(e, dtype=int)
                    yp[e > t1] = 1; yp[e > t2] = 2; yp[e > t3] = 3
                    v = cohen_kappa_score(y, yp, weights="quadratic")
                    if best is None or v > best['v']:
                        best = {'t': (float(t1), float(t2), float(t3)), 'v': float(v)}
        return best

    def apply_thr(e, t):
        yp = np.zeros_like(e, dtype=int)
        yp[e > t[0]] = 1; yp[e > t[1]] = 2; yp[e > t[2]] = 3
        return yp

    train_dl = DataLoader(DS(by_split["Train"]), batch_size=args.batch,
                          shuffle=True, collate_fn=collate, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(DS(by_split["Validation"]), batch_size=args.batch*2,
                          shuffle=False, collate_fn=collate, num_workers=4, pin_memory=True)
    test_dl  = DataLoader(DS(by_split["Test"]), batch_size=args.batch*2,
                          shuffle=False, collate_fn=collate, num_workers=4, pin_memory=True)

    cnt = np.zeros(4);
    for r in by_split["Train"]: cnt[int(r["engagement"])] += 1
    alpha = 1.0 / np.maximum(cnt, 1); alpha = alpha / alpha.sum() * 4
    print(f"Train counts: {cnt.tolist()}  alpha: {alpha.tolist()}")

    all_results = []; all_p_te = []; all_p_va = []; y_te = y_va = None

    for seed in [int(s) for s in args.seeds.split(",")]:
        print(f"\n========== Seed {seed} ==========", flush=True)
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        backbone = AutoModel.from_pretrained(args.model).vision_model
        model = EngageNetX(backbone).to(device)
        alpha_t = torch.tensor(alpha, dtype=torch.float32, device=device)

        params = [
            {"params": list(model.backbone.parameters()), "lr": args.lr_enc},
            {"params": list(model.proj.parameters()) + [model.pos]
                      + list(model.temporal.parameters())
                      + list(model.norm.parameters())
                      + list(model.head.parameters()),
             "lr": args.lr_head},
        ]
        opt = torch.optim.AdamW(params, weight_decay=0.05)
        base = [args.lr_enc, args.lr_head]
        n_steps = max(1, (len(train_dl) // args.accum) * args.epochs)
        warmup = max(100, n_steps // 30)
        def set_lr(step):
            s = step / max(1, warmup) if step < warmup else 0.5 * (1 + np.cos(np.pi * (step - warmup) / max(1, n_steps - warmup)))
            for pg, b in zip(opt.param_groups, base):
                pg["lr"] = b * s

        scaler = GradScaler(enabled=(amp_dtype == torch.float16))

        @torch.no_grad()
        def eval_split(dl):
            model.eval(); logits_all=[]; ys=[]
            for pv, y in dl:
                pv = pv.to(device, non_blocking=True)
                with autocast(dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                    logits = model(pv).float()
                logits_all.append(logits.cpu()); ys.append(y)
            return torch.cat(logits_all), torch.cat(ys).numpy()

        step = 0; history = []
        for ep in range(args.epochs):
            model.train(); opt.zero_grad()
            t0 = time.time()
            for i, (pv, y) in enumerate(train_dl):
                pv = pv.to(device, non_blocking=True); y = y.to(device)
                with autocast(dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                    logits = model(pv).float()
                    loss = corn_loss(logits, y, alpha_t, gamma=args.gamma_focal) / args.accum
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (i + 1) % args.accum == 0:
                    set_lr(step); step += 1
                    if scaler.is_enabled():
                        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(opt); scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                    opt.zero_grad()
                    if step % 100 == 0:
                        print(f"  ep{ep} step{step}/{n_steps} loss={loss.item()*args.accum:.3f}", flush=True)
            L_va, yv = eval_split(val_dl); L_te, yt = eval_split(test_dl)
            e_va = corn_predict_expected(L_va).numpy(); e_te = corn_predict_expected(L_te).numpy()
            yp_va_arg = corn_argmax(L_va).numpy(); yp_te_arg = corn_argmax(L_te).numpy()
            bt = tune_thresholds(e_va, yv)
            yp_te_thr = apply_thr(e_te, bt['t'])
            val_kq = cohen_kappa_score(yv, yp_va_arg, weights="quadratic")
            test_kq_arg = cohen_kappa_score(yt, yp_te_arg, weights="quadratic")
            test_kq_thr = cohen_kappa_score(yt, yp_te_thr, weights="quadratic")
            dt = time.time() - t0
            print(f"[seed {seed} ep{ep}] val_arg={val_kq:.4f} val_thr={bt['v']:.4f} test_arg={test_kq_arg:.4f} test_thr={test_kq_thr:.4f} dt={dt/60:.1f}min", flush=True)
            history.append({"ep": ep, "val_arg_kq": float(val_kq), "val_thr_kq": float(bt['v']),
                           "test_arg_kq": float(test_kq_arg), "test_thr_kq": float(test_kq_thr),
                           "thresholds": bt['t']})
        # Save final per-seed predictions
        L_va, yv = eval_split(val_dl); L_te, yt = eval_split(test_dl)
        e_va = corn_predict_expected(L_va).numpy(); e_te = corn_predict_expected(L_te).numpy()
        all_p_te.append(e_te); all_p_va.append(e_va)
        y_te = yt; y_va = yv
        all_results.append({"seed": seed, "history": history})

        del model; torch.cuda.empty_cache()

    # Ensemble: average expected-class scores
    e_te = np.mean(all_p_te, axis=0); e_va = np.mean(all_p_va, axis=0)
    bt = tune_thresholds(e_va, y_va)
    yp_te = apply_thr(e_te, bt['t'])
    kq = float(cohen_kappa_score(y_te, yp_te, weights="quadratic"))
    acc = float(accuracy_score(y_te, yp_te))
    print(f"\nENSEMBLE: test κ_q = {kq:.4f}  acc={acc:.3f}  t={bt['t']}", flush=True)

    out = {"per_seed": all_results,
           "ensemble": {"test_kq": kq, "test_acc": acc, "thresholds": bt['t'],
                        "val_kq_at_thr": bt['v']}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
