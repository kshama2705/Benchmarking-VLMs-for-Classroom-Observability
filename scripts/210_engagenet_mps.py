"""
ENGAGENET-X — MPS-native edition for Apple Silicon (M-series, optimized
for M5 Pro / M4 Pro / M3 Max with 36+GB unified memory).

Full SigLIP-L end-to-end fine-tune + 8-frame temporal transformer +
CORN ordinal head + focal loss. Single-stream supervised method.

Differences from scripts/180_supervised_e2e_gpu.py (CUDA version):
- torch.mps.empty_cache() instead of torch.cuda.empty_cache()
- No autocast/GradScaler (MPS bf16 inconsistent on torch <= 2.4 — use fp32)
- Smaller default batch (M5 Pro 36GB has less headroom than A100 80GB)
- Frame pre-extraction included inline (no separate 180b step needed)

USAGE on M5 Pro:
  cd "CVPR 2026 Workshop"
  source venv/bin/activate
  python3 scripts/210_engagenet_mps.py \
      --epochs 10 --batch 4 --accum 4 \
      --seeds 0,42,2025 --n_frames 8

Total: ~24-36h on M5 Pro for 3 seeds. Output: results/sota/engagenet_mps.json
plus best checkpoint per seed in models/.
"""
import os, json, argparse, time, csv, random, subprocess, glob, tempfile
import numpy as np

def main():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    from sklearn.metrics import cohen_kappa_score, accuracy_score
    from transformers import AutoModel, SiglipImageProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="frames_full/manifest.csv")
    parser.add_argument("--frames-root", default="frames_8",
                        help="Output dir for 8-frame extraction (will be auto-populated)")
    parser.add_argument("--video-root", default="DAiSEE/DataSet",
                        help="DAiSEE raw video root for frame extraction")
    parser.add_argument("--out", default="results/sota/engagenet_mps.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr_enc", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=3e-4)
    parser.add_argument("--gamma_focal", type=float, default=2.0)
    parser.add_argument("--n_unfreeze_blocks", type=int, default=24,
                        help="How many transformer blocks to unfreeze (24 = all of SigLIP-L)")
    parser.add_argument("--seeds", default="0,42,2025")
    parser.add_argument("--n_frames", type=int, default=8)
    parser.add_argument("--model", default="google/siglip-large-patch16-256")
    parser.add_argument("--skip_extract", action="store_true",
                        help="Skip frame extraction (use existing frames_root)")
    args = parser.parse_args()

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device={device}", flush=True)

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MANIFEST = os.path.join(BASE, args.manifest)
    FRAMES_ROOT = os.path.join(BASE, args.frames_root)
    VIDEO_ROOT = os.path.join(BASE, args.video_root)
    OUT = os.path.join(BASE, args.out)
    LOG = OUT.replace(".json", ".log")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    os.makedirs(os.path.join(BASE, "models"), exist_ok=True)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(LOG, "a") as f:
            f.write(line + "\n")

    # ----------------- Frame extraction (inline) -----------------
    TIMESTAMPS = np.linspace(0.5, 8.9, args.n_frames).tolist()

    def extract_frames_if_needed(rows):
        if args.skip_extract:
            return
        log(f"Extracting {args.n_frames} frames at t={TIMESTAMPS} to {FRAMES_ROOT}/")
        n_done = 0; n_skip = 0; n_fail = 0; t0 = time.time()
        for r in rows:
            cid = r["clip_id"].replace(".avi", "").replace(".mp4", "")
            split = r["split"]
            subject = cid[:6]
            video = None
            for ext in (".avi", ".mp4"):
                p = os.path.join(VIDEO_ROOT, split, subject, cid, cid + ext)
                if os.path.exists(p): video = p; break
            if video is None:
                n_fail += 1; continue
            out_clip = os.path.join(FRAMES_ROOT, split, cid)
            done = all(os.path.exists(os.path.join(out_clip, f"t{i:02d}.jpg"))
                       for i in range(args.n_frames))
            if done:
                n_skip += 1; continue
            os.makedirs(out_clip, exist_ok=True)
            ok = True
            for i, t in enumerate(TIMESTAMPS):
                fp = os.path.join(out_clip, f"t{i:02d}.jpg")
                rr = subprocess.run([
                    "ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", video,
                    "-frames:v", "1", "-q:v", "2", fp,
                ], capture_output=True, timeout=20)
                if rr.returncode != 0 or not os.path.exists(fp):
                    ok = False; break
            if ok:
                n_done += 1
            else:
                n_fail += 1
            if (n_done + n_skip + n_fail) % 200 == 0:
                dt = time.time() - t0
                rate = (n_done + n_skip + n_fail) / max(1, dt)
                log(f"  extract: done={n_done} skip={n_skip} fail={n_fail} rate={rate:.1f} clip/s")
        log(f"Extraction complete: done={n_done} skip={n_skip} fail={n_fail}")

    # ----------------- Data ---------------
    rows = []
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "exists"):
                rows.append(r)
    extract_frames_if_needed(rows)

    # Re-filter: only keep clips where all 8 frames now exist
    def has_all_frames(r):
        cid = r["clip_id"].replace(".avi", "").replace(".mp4", "")
        return all(os.path.exists(os.path.join(FRAMES_ROOT, r["split"], cid, f"t{i:02d}.jpg"))
                   for i in range(args.n_frames))
    rows_ok = [r for r in rows if has_all_frames(r)]
    log(f"Rows with all {args.n_frames} frames: {len(rows_ok)}/{len(rows)}")
    by_split = {s: [r for r in rows_ok if r["split"] == s]
                for s in ("Train", "Validation", "Test")}
    log(f"Splits: " + ", ".join(f"{k}={len(v)}" for k, v in by_split.items()))

    proc = SiglipImageProcessor.from_pretrained(args.model)

    def get_frames(row):
        cid = row["clip_id"].replace(".avi", "").replace(".mp4", "")
        split = row["split"]
        clip_dir = os.path.join(FRAMES_ROOT, split, cid)
        imgs = []
        for i in range(args.n_frames):
            p = os.path.join(clip_dir, f"t{i:02d}.jpg")
            imgs.append(Image.open(p).convert("RGB"))
        return imgs

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
        pv = torch.stack([b[0] for b in batch])
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
                nn.Linear(d_model, 3),  # CORN ordinal logits
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
        K = 4; losses = []
        for k in range(K - 1):
            target = (y > k).float()
            logit = logits[:, k]
            p = torch.sigmoid(logit)
            pt = torch.where(target == 1, p, 1 - p)
            bce = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
            fl = (1 - pt).pow(gamma) * bce
            fl = alpha[y] * fl
            losses.append(fl.mean())
        return torch.stack(losses).mean()

    def corn_expected(logits):
        return torch.sigmoid(logits).sum(dim=1)

    def corn_argmax(logits):
        return (torch.sigmoid(logits) > 0.5).sum(dim=1)

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

    def boot_ci(yt, yp, n=1000, seed=42):
        rng = np.random.default_rng(seed); nt = len(yt); out = []
        for _ in range(n):
            idx = rng.integers(0, nt, size=nt)
            out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
        return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]

    train_dl = DataLoader(DS(by_split["Train"]), batch_size=args.batch,
                          shuffle=True, collate_fn=collate, num_workers=0)
    val_dl = DataLoader(DS(by_split["Validation"]), batch_size=args.batch*2,
                        shuffle=False, collate_fn=collate, num_workers=0)
    test_dl = DataLoader(DS(by_split["Test"]), batch_size=args.batch*2,
                         shuffle=False, collate_fn=collate, num_workers=0)

    cnt = np.zeros(4)
    for r in by_split["Train"]: cnt[int(r["engagement"])] += 1
    alpha = 1.0 / np.maximum(cnt, 1); alpha = alpha / alpha.sum() * 4
    log(f"Train counts: {cnt.tolist()}  alpha: {alpha.tolist()}")

    all_results = []; all_e_te = []; all_e_va = []; y_te = y_va = None

    for seed in [int(s) for s in args.seeds.split(",")]:
        log(f"\n========== Seed {seed} ==========")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        backbone = AutoModel.from_pretrained(args.model).vision_model
        # Selective unfreeze (default = all 24 blocks)
        n_blocks = len(backbone.encoder.layers)
        for p in backbone.parameters():
            p.requires_grad = False
        for blk in backbone.encoder.layers[-args.n_unfreeze_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        if hasattr(backbone, 'post_layernorm'):
            for p in backbone.post_layernorm.parameters():
                p.requires_grad = True

        model = EngageNetX(backbone).to(device)
        alpha_t = torch.tensor(alpha, dtype=torch.float32, device=device)

        enc_params = [p for p in model.backbone.parameters() if p.requires_grad]
        head_params = (list(model.proj.parameters()) + [model.pos]
                       + list(model.temporal.parameters())
                       + list(model.norm.parameters())
                       + list(model.head.parameters()))
        n_enc = sum(p.numel() for p in enc_params)
        n_head = sum(p.numel() for p in head_params)
        log(f"Trainable: enc {n_enc/1e6:.1f}M  head {n_head/1e6:.2f}M")

        opt = torch.optim.AdamW(
            [{"params": enc_params, "lr": args.lr_enc},
             {"params": head_params, "lr": args.lr_head}],
            weight_decay=0.05,
        )
        base = [args.lr_enc, args.lr_head]
        n_steps = max(1, (len(train_dl) // args.accum) * args.epochs)
        warmup = max(100, n_steps // 30)
        def set_lr(step):
            s = step / max(1, warmup) if step < warmup else 0.5 * (1 + np.cos(np.pi * (step - warmup) / max(1, n_steps - warmup)))
            for pg, b in zip(opt.param_groups, base):
                pg["lr"] = b * s

        @torch.no_grad()
        def eval_split(dl):
            model.eval(); logits_all=[]; ys=[]
            for pv, y in dl:
                pv = pv.to(device, non_blocking=True)
                logits = model(pv).float()
                logits_all.append(logits.cpu()); ys.append(y)
            return torch.cat(logits_all), torch.cat(ys).numpy()

        history = []; step = 0; best_val_kq = -1; best_state = None
        for ep in range(args.epochs):
            model.train(); opt.zero_grad()
            t0 = time.time()
            for i, (pv, y) in enumerate(train_dl):
                pv = pv.to(device, non_blocking=True); y = y.to(device)
                logits = model(pv).float()
                loss = corn_loss(logits, y, alpha_t, gamma=args.gamma_focal) / args.accum
                loss.backward()
                if (i + 1) % args.accum == 0:
                    set_lr(step); step += 1
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()), 1.0)
                    opt.step(); opt.zero_grad()
                    if step % 50 == 0:
                        log(f"  ep{ep} step{step}/{n_steps} loss={loss.item()*args.accum:.3f} lr_enc={opt.param_groups[0]['lr']:.2e}")
            L_va, yv = eval_split(val_dl); L_te, yt = eval_split(test_dl)
            e_va = corn_expected(L_va).numpy(); e_te = corn_expected(L_te).numpy()
            yp_va_arg = corn_argmax(L_va).numpy(); yp_te_arg = corn_argmax(L_te).numpy()
            bt = tune_thresholds(e_va, yv)
            yp_te_thr = apply_thr(e_te, bt['t'])
            val_kq = cohen_kappa_score(yv, yp_va_arg, weights="quadratic")
            test_kq_arg = cohen_kappa_score(yt, yp_te_arg, weights="quadratic")
            test_kq_thr = cohen_kappa_score(yt, yp_te_thr, weights="quadratic")
            test_acc_thr = accuracy_score(yt, yp_te_thr)
            dt = time.time() - t0
            log(f"[seed {seed} ep{ep}] val_arg={val_kq:.4f} val_thr={bt['v']:.4f} test_arg={test_kq_arg:.4f} test_thr={test_kq_thr:.4f} acc_thr={test_acc_thr:.3f} dt={dt/60:.1f}min")
            history.append({"ep": ep, "val_arg_kq": float(val_kq), "val_thr_kq": float(bt['v']),
                           "test_arg_kq": float(test_kq_arg), "test_thr_kq": float(test_kq_thr),
                           "test_acc_thr": float(test_acc_thr), "thresholds": bt['t']})
            if bt['v'] > best_val_kq:
                best_val_kq = bt['v']
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # Use best checkpoint for ensemble probs
        if best_state is not None:
            model.load_state_dict(best_state)
        L_va, yv = eval_split(val_dl); L_te, yt = eval_split(test_dl)
        e_va = corn_expected(L_va).numpy(); e_te = corn_expected(L_te).numpy()
        all_e_te.append(e_te); all_e_va.append(e_va)
        y_te = yt; y_va = yv
        all_results.append({"seed": seed, "history": history,
                           "best_val_thr_kq": float(best_val_kq)})
        torch.save({"state_dict": best_state, "history": history},
                   os.path.join(BASE, "models", f"engagenet_mps_seed{seed}.pt"))

        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        # Intermediate save
        partial = {"per_seed": all_results, "ensemble": None}
        with open(OUT.replace(".json", "_partial.json"), 'w') as f:
            json.dump(partial, f, indent=2)

    # Ensemble: average expected-class
    e_te = np.mean(all_e_te, axis=0); e_va = np.mean(all_e_va, axis=0)
    bt = tune_thresholds(e_va, y_va)
    yp_te = apply_thr(e_te, bt['t'])
    kq = float(cohen_kappa_score(y_te, yp_te, weights="quadratic"))
    acc = float(accuracy_score(y_te, yp_te))
    ci = boot_ci(y_te, yp_te)
    log(f"\nENSEMBLE: test κ_q={kq:.4f} CI={ci} acc={acc:.3f} t={bt['t']}")

    out = {"per_seed": all_results,
           "ensemble": {"test_kq": kq, "test_acc": acc, "ci": ci,
                        "thresholds": bt['t']}}
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
