"""
MEME-V — Multimodal Engagement with Video-temporal motion stream.
M5 Pro / MPS-native.

Why MEME (211) failed: MediaPipe blendshapes/gaze/pose carry weak and
identity-correlated signal on classroom webcam footage. Engagement
discrimination requires *motion patterns*, not static landmarks.

MEME-V replaces the MediaPipe stream with a frozen Kinetics-pretrained
3D CNN (R3D-18 from torchvision). 3D motion features are inherently
identity-less: the *way* a subject moves when engaged vs bored is the
engagement signal, not their face geometry.

Streams:
1. SigLIP-L cached features (3 frames × 1024) — frozen vision context
2. R3D-18 on 16 evenly-spaced frames per clip (Kinetics-400 pretrained,
   last 2 blocks unfrozen) — temporal motion
3. FiLM fusion: R3D motion features modulate SigLIP temporal sequence
4. CORN ordinal head + multi-task auxiliary (B, C, F) heads
5. Class-balanced focal loss + WeightedRandomSampler

Runtime estimate on M5 Pro: ~8-12h per seed (R3D-18 is small but
processes 16 frames per clip).

Targets test κ_q ≥ 0.35, accuracy ≥ 60%. If it lands there it's the
positive method for BMVC.

USAGE:
  python3 scripts/212_meme_v_mps.py \
      --epochs 15 --batch 4 --accum 4 --seeds 0,42,2025
"""
import os, json, argparse, time, csv, random, subprocess, tempfile, glob
import numpy as np


def main():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision.models.video import r3d_18, R3D_18_Weights
    from torchvision import transforms
    from PIL import Image
    from sklearn.metrics import cohen_kappa_score, accuracy_score

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/sota/meme_v.json")
    parser.add_argument("--video-root", default="DAiSEE/DataSet")
    parser.add_argument("--frames-root", default="frames_16",
                        help="Where to cache 16 evenly-spaced frames per clip")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr_enc", type=float, default=5e-5)
    parser.add_argument("--lr_head", type=float, default=5e-4)
    parser.add_argument("--gamma_focal", type=float, default=2.0)
    parser.add_argument("--aux_weight", type=float, default=0.2)
    parser.add_argument("--seeds", default="0,42,2025")
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--d_h", type=int, default=256)
    parser.add_argument("--skip_extract", action="store_true")
    args = parser.parse_args()

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUT = os.path.join(BASE, args.out)
    LOG = OUT.replace(".json", ".log")
    FRAMES_ROOT = os.path.join(BASE, args.frames_root)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(LOG, "a") as f:
            f.write(line + "\n")

    log(f"Device: {device}")

    # ----------------- Load cached SigLIP-L multiframe -----------------
    log("Loading SigLIP-L 3-frame features...")
    F1 = np.load(os.path.join(BASE, "features/daisee_siglip_l_multiframe_features.npz"),
                 allow_pickle=True)
    feat_vis = F1["feat"].astype(np.float32)  # (N, 3, 1024)
    clip_ids = F1["clip_id"]
    split = F1["split"]
    subject_id = F1["subject_id"]
    y_eng = F1["engagement"].astype(np.int64)
    N = len(clip_ids)
    log(f"  vision feat: {feat_vis.shape}")

    # Load auxiliary labels
    log("Loading auxiliary B/C/F labels...")
    aux_y = {"Boredom": np.full(N, -1, dtype=np.int64),
             "Confusion": np.full(N, -1, dtype=np.int64),
             "Frustration": np.full(N, -1, dtype=np.int64)}
    clip_to_idx = {cid.replace(".avi", "").replace(".mp4", ""): i for i, cid in enumerate(clip_ids)}
    for split_name in ("Train", "Validation", "Test"):
        fp = os.path.join(BASE, "DAiSEE", "Labels", f"{split_name}Labels.csv")
        if not os.path.exists(fp): continue
        with open(fp) as f:
            for r in csv.DictReader(f):
                cid = r["ClipID"].replace(".avi", "").replace(".mp4", "")
                if cid in clip_to_idx:
                    i = clip_to_idx[cid]
                    for col in ("Boredom", "Confusion", "Frustration"):
                        if col in r and r[col] != "":
                            aux_y[col][i] = int(r[col])
    log(f"  aux coverage: " + ", ".join(f"{k}={(v>=0).sum()}" for k, v in aux_y.items()))

    tr_mask = (split == "Train"); va_mask = (split == "Validation"); te_mask = (split == "Test")
    log(f"Splits: tr={tr_mask.sum()} va={va_mask.sum()} te={te_mask.sum()}")

    # ----------------- Inline frame extraction (16 frames per clip) -----------------
    TIMESTAMPS = np.linspace(0.3, 9.6, args.n_frames).tolist()

    def find_video(split_name, cid):
        subject = cid[:6]
        for ext in (".avi", ".mp4"):
            p = os.path.join(BASE, args.video_root, split_name, subject, cid, cid + ext)
            if os.path.exists(p): return p
        return None

    def has_frames(split_name, cid):
        clip_dir = os.path.join(FRAMES_ROOT, split_name, cid)
        return all(os.path.exists(os.path.join(clip_dir, f"t{i:02d}.jpg"))
                   for i in range(args.n_frames))

    def extract_if_needed():
        if args.skip_extract: return
        log(f"Extracting {args.n_frames} frames per clip → {FRAMES_ROOT}/")
        n_done = 0; n_skip = 0; n_fail = 0; t0 = time.time()
        for i in range(N):
            cid = clip_ids[i].replace(".avi", "").replace(".mp4", "")
            sp = split[i]
            if has_frames(sp, cid):
                n_skip += 1; continue
            vid = find_video(sp, cid)
            if vid is None:
                n_fail += 1; continue
            out_clip = os.path.join(FRAMES_ROOT, sp, cid)
            os.makedirs(out_clip, exist_ok=True)
            ok = True
            for j, t in enumerate(TIMESTAMPS):
                fp = os.path.join(out_clip, f"t{j:02d}.jpg")
                r = subprocess.run([
                    "ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", vid,
                    "-frames:v", "1", "-q:v", "2", "-vf", "scale=112:112", fp,
                ], capture_output=True, timeout=20)
                if r.returncode != 0 or not os.path.exists(fp):
                    ok = False; break
            if ok: n_done += 1
            else: n_fail += 1
            if (n_done + n_skip + n_fail) % 200 == 0:
                dt = time.time() - t0
                log(f"  extract: done={n_done} skip={n_skip} fail={n_fail} rate={(n_done+n_skip+n_fail)/max(1,dt):.1f}/s")
        log(f"Extraction complete: done={n_done} skip={n_skip} fail={n_fail}")

    extract_if_needed()

    # Filter to clips with frames
    has_all = np.array([has_frames(split[i], clip_ids[i].replace(".avi","").replace(".mp4",""))
                        for i in range(N)])
    log(f"Clips with all {args.n_frames} frames: {has_all.sum()}/{N}")

    # ----------------- Dataset -----------------
    KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
    KINETICS_STD = [0.22803, 0.22145, 0.216989]
    video_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(KINETICS_MEAN, KINETICS_STD),
    ])

    class MemeVDataset(Dataset):
        def __init__(self, indices):
            self.indices = indices
        def __len__(self): return len(self.indices)
        def __getitem__(self, i):
            idx = self.indices[i]
            cid = clip_ids[idx].replace(".avi","").replace(".mp4","")
            sp = split[idx]
            clip_dir = os.path.join(FRAMES_ROOT, sp, cid)
            # Load 16 frames as (T, 3, H, W) → R3D wants (C, T, H, W)
            frames = []
            for j in range(args.n_frames):
                p = os.path.join(clip_dir, f"t{j:02d}.jpg")
                img = Image.open(p).convert("RGB")
                frames.append(video_tfm(img))
            vid_t = torch.stack(frames, dim=1)  # (3, T, H, W)
            x_vis = torch.tensor(feat_vis[idx], dtype=torch.float32)  # (3, 1024)
            return vid_t, x_vis, int(y_eng[idx]), int(aux_y["Boredom"][idx]), \
                   int(aux_y["Confusion"][idx]), int(aux_y["Frustration"][idx])

    # ----------------- Model -----------------
    class FiLMBlock(nn.Module):
        def __init__(self, d_h, d_film):
            super().__init__()
            self.gamma = nn.Linear(d_film, d_h)
            self.beta = nn.Linear(d_film, d_h)
            nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
            nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)
        def forward(self, x, film):
            g = self.gamma(film).unsqueeze(1)
            b = self.beta(film).unsqueeze(1)
            return (1 + g) * x + b

    class MEMEV(nn.Module):
        def __init__(self, d_vis=1024, d_h=256, n_layers=2, n_heads=8, n_frames=3):
            super().__init__()
            # Vision branch: SigLIP-L 3-frame features → temporal transformer
            self.proj_vis = nn.Linear(d_vis, d_h)
            self.pos = nn.Parameter(torch.randn(1, n_frames, d_h) * 0.02)
            self.films = nn.ModuleList([FiLMBlock(d_h, d_h) for _ in range(n_layers)])
            enc_layer = lambda: nn.TransformerEncoderLayer(
                d_model=d_h, nhead=n_heads, dim_feedforward=4*d_h,
                dropout=0.15, activation="gelu", batch_first=True, norm_first=True)
            self.layers = nn.ModuleList([enc_layer() for _ in range(n_layers)])
            # Video motion branch: R3D-18 Kinetics-pretrained, last 2 blocks unfrozen
            self.r3d = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            for p in self.r3d.parameters(): p.requires_grad = False
            # Unfreeze layer3, layer4, avgpool, fc
            for blk in (self.r3d.layer3, self.r3d.layer4):
                for p in blk.parameters(): p.requires_grad = True
            d_r3d = self.r3d.fc.in_features  # 512
            self.r3d.fc = nn.Identity()
            self.proj_motion = nn.Sequential(
                nn.Linear(d_r3d, d_h), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(d_h, d_h),
            )
            self.norm = nn.LayerNorm(d_h)
            # Engagement head: CORN ordinal
            self.head_e = nn.Sequential(
                nn.Linear(d_h*2, d_h), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(d_h, 3),
            )
            self.head_b = nn.Sequential(nn.Linear(d_h*2, d_h), nn.GELU(),
                                        nn.Dropout(0.2), nn.Linear(d_h, 4))
            self.head_c = nn.Sequential(nn.Linear(d_h*2, d_h), nn.GELU(),
                                        nn.Dropout(0.2), nn.Linear(d_h, 4))
            self.head_f = nn.Sequential(nn.Linear(d_h*2, d_h), nn.GELU(),
                                        nn.Dropout(0.2), nn.Linear(d_h, 4))

        def forward(self, vid, x_vis):
            # vid: (B, 3, T, H, W); x_vis: (B, 3, d_vis)
            motion = self.r3d(vid)              # (B, 512)
            motion_h = self.proj_motion(motion) # (B, d_h)
            z = self.proj_vis(x_vis) + self.pos
            for film, layer in zip(self.films, self.layers):
                z = film(z, motion_h)
                z = layer(z)
            z_pool = self.norm(z.mean(dim=1))   # (B, d_h)
            fused = torch.cat([z_pool, motion_h], dim=1)  # (B, 2*d_h)
            return {"e": self.head_e(fused), "b": self.head_b(fused),
                    "c": self.head_c(fused), "f": self.head_f(fused)}

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

    def corn_expected(logits): return torch.sigmoid(logits).sum(dim=1)
    def corn_argmax(logits): return (torch.sigmoid(logits) > 0.5).sum(dim=1)

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

    # Build indices (only clips with frames)
    tr_idx = np.where(tr_mask & has_all)[0]
    va_idx = np.where(va_mask & has_all)[0]
    te_idx = np.where(te_mask & has_all)[0]
    log(f"With frames: tr={len(tr_idx)} va={len(va_idx)} te={len(te_idx)}")

    cnt_e = np.bincount(y_eng[tr_idx], minlength=4).astype(np.float32)
    alpha_e = 1.0 / np.maximum(cnt_e, 1); alpha_e = alpha_e / alpha_e.sum() * 4
    sample_w = (1.0 / np.maximum(cnt_e, 1))[y_eng[tr_idx]]

    all_e_te = []; all_e_va = []; y_te = y_va = None; seed_results = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        log(f"\n========== Seed {seed} ==========")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        sampler = WeightedRandomSampler(weights=torch.tensor(sample_w, dtype=torch.float32),
                                        num_samples=len(tr_idx), replacement=True)
        train_dl = DataLoader(MemeVDataset(tr_idx), batch_size=args.batch,
                              sampler=sampler, num_workers=2, pin_memory=False,
                              persistent_workers=True)
        val_dl = DataLoader(MemeVDataset(va_idx), batch_size=args.batch*2,
                            shuffle=False, num_workers=2, persistent_workers=True)
        test_dl = DataLoader(MemeVDataset(te_idx), batch_size=args.batch*2,
                             shuffle=False, num_workers=2, persistent_workers=True)

        model = MEMEV(d_h=args.d_h).to(device)
        alpha_t = torch.tensor(alpha_e, dtype=torch.float32, device=device)
        enc_params = [p for p in model.r3d.parameters() if p.requires_grad]
        head_params = [p for p in model.parameters()
                       if p.requires_grad and not any(p is q for q in enc_params)]
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

        n_enc = sum(p.numel() for p in enc_params)
        n_head = sum(p.numel() for p in head_params)
        log(f"Trainable: r3d(last2) {n_enc/1e6:.1f}M  head+vis {n_head/1e6:.2f}M  total={n_enc/1e6 + n_head/1e6:.1f}M")

        @torch.no_grad()
        def eval_split(dl):
            model.eval(); logits_all=[]; ys=[]
            for vid, x_vis, y_e, y_b, y_c, y_f in dl:
                vid = vid.to(device, non_blocking=True); x_vis = x_vis.to(device, non_blocking=True)
                out = model(vid, x_vis)
                logits_all.append(out["e"].cpu()); ys.append(y_e)
            return torch.cat(logits_all), torch.cat(ys).numpy()

        history = []; step = 0; best_val_kq = -1; best_state = None
        for ep in range(args.epochs):
            model.train(); opt.zero_grad()
            t0 = time.time()
            for i, (vid, x_vis, y_e, y_b, y_c, y_f) in enumerate(train_dl):
                vid = vid.to(device, non_blocking=True); x_vis = x_vis.to(device, non_blocking=True)
                y_e = y_e.to(device); y_b = y_b.to(device); y_c = y_c.to(device); y_f = y_f.to(device)
                out = model(vid, x_vis)
                loss_e = corn_loss(out["e"], y_e, alpha_t, gamma=args.gamma_focal)
                aux = 0.0
                for logits, y_aux in ((out["b"], y_b), (out["c"], y_c), (out["f"], y_f)):
                    mask = (y_aux >= 0)
                    if mask.sum() > 0:
                        aux = aux + F.cross_entropy(logits[mask], y_aux[mask])
                loss = (loss_e + args.aux_weight * aux) / args.accum
                loss.backward()
                if (i + 1) % args.accum == 0:
                    set_lr(step); step += 1
                    torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
                    opt.step(); opt.zero_grad()
                    if step % 25 == 0:
                        log(f"  ep{ep} step{step}/{n_steps} loss={loss.item()*args.accum:.3f} lr_enc={opt.param_groups[0]['lr']:.2e}")
            L_va, yv = eval_split(val_dl); L_te, yt = eval_split(test_dl)
            e_va = corn_expected(L_va).numpy(); e_te = corn_expected(L_te).numpy()
            yp_va_arg = corn_argmax(L_va).numpy(); yp_te_arg = corn_argmax(L_te).numpy()
            bt = tune_thresholds(e_va, yv)
            yp_te_thr = apply_thr(e_te, bt['t'])
            val_kq = float(cohen_kappa_score(yv, yp_va_arg, weights="quadratic"))
            test_kq_arg = float(cohen_kappa_score(yt, yp_te_arg, weights="quadratic"))
            test_kq_thr = float(cohen_kappa_score(yt, yp_te_thr, weights="quadratic"))
            test_acc_thr = float(accuracy_score(yt, yp_te_thr))
            dt = time.time() - t0
            log(f"[seed {seed} ep{ep:02d}] val_arg={val_kq:.4f} val_thr={bt['v']:.4f} test_arg={test_kq_arg:.4f} test_thr={test_kq_thr:.4f} acc_thr={test_acc_thr:.3f} dt={dt/60:.1f}min")
            history.append({"ep": ep, "val_arg": val_kq, "val_thr": float(bt['v']),
                           "test_arg": test_kq_arg, "test_thr": test_kq_thr,
                           "test_acc_thr": test_acc_thr, "t": bt['t']})
            if bt['v'] > best_val_kq:
                best_val_kq = bt['v']
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        L_va, yv = eval_split(val_dl); L_te, yt = eval_split(test_dl)
        all_e_te.append(corn_expected(L_te).numpy())
        all_e_va.append(corn_expected(L_va).numpy())
        y_te = yt; y_va = yv
        seed_results.append({"seed": seed, "history": history,
                             "best_val_thr_kq": float(best_val_kq)})

        del model
        if device.type == "mps": torch.mps.empty_cache()

    e_te = np.mean(all_e_te, axis=0); e_va = np.mean(all_e_va, axis=0)
    bt = tune_thresholds(e_va, y_va)
    yp_te = apply_thr(e_te, bt['t'])
    kq = float(cohen_kappa_score(y_te, yp_te, weights="quadratic"))
    acc = float(accuracy_score(y_te, yp_te))
    ci = boot_ci(y_te, yp_te)
    log(f"\nENSEMBLE: test κ_q={kq:.4f} CI={ci} acc={acc:.3f} t={bt['t']}")

    out = {"per_seed": seed_results,
           "ensemble": {"test_kq": kq, "test_acc": acc, "ci": ci, "thresholds": bt['t']}}
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
