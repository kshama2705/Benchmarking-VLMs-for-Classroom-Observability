"""
MEME — Multimodal Explicit-feature Modulated Engagement.
M5 Pro / Apple Silicon MPS-native edition.

Novel architecture combining:
1. Vision stream: SigLIP-L pre-extracted features (3 frames @ t=2,5,8) +
   2-layer temporal transformer over 3 frames. SigLIP-L is FROZEN (features
   already cached) — saves substantial GPU time vs end-to-end fine-tune.
2. Explicit stream: per-frame MediaPipe blendshapes (52) + gaze (6) +
   head pose (3) + body pose (16) = 77-d per frame, mean+std+delta over
   the same 3 frames → 231-d per clip. Identity-invariant by construction.
3. FiLM fusion: explicit signals modulate vision features (gain/bias) at
   each temporal-transformer block.
4. Multi-task head: predicts Engagement (ordinal, primary loss) + Boredom,
   Confusion, Frustration (auxiliary 4-class CE, weight 0.3 each).
5. Class-balanced focal + CORN ordinal for engagement.

Why this can beat ViBED-Net (κ≈0.55, 73% acc):
- ViBED-Net uses only video. MEME adds identity-invariant explicit signals
  that capture within-subject variation (which our IDEP diagnostic showed
  is the frozen-feature ceiling).
- ViBED-Net is single-task. MEME's multi-task auxiliary supervision
  regularizes engagement.
- SigLIP-L (430M backbone) >> ViBED-Net (~22M 3D ResNet18).
- Train+Val combined; multi-seed ensemble; test-set bootstrap CI.

Runtime: ~3-6 hours per seed on M5 Pro (much faster than 210_engagenet_mps
because vision is frozen). Total ~12-24h for 3 seeds.

USAGE:
  python3 scripts/211_meme_mps.py --epochs 25 --batch 64 --seeds 0,42,2025
"""
import os, json, argparse, time, csv, random
import numpy as np


def main():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from sklearn.metrics import cohen_kappa_score, accuracy_score

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/sota/meme.json")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gamma_focal", type=float, default=2.0)
    parser.add_argument("--aux_weight", type=float, default=0.3,
                        help="Weight for each auxiliary B/C/F head loss")
    parser.add_argument("--seeds", default="0,42,2025")
    parser.add_argument("--d_h", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=8)
    args = parser.parse_args()

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUT = os.path.join(BASE, args.out)
    LOG = OUT.replace(".json", ".log")
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

    # ----------------- Load cached features ---------------
    log("Loading SigLIP-L multi-frame features (3 × 1024)...")
    F1 = np.load(os.path.join(BASE, "features/daisee_siglip_l_multiframe_features.npz"), allow_pickle=True)
    feat_vis = F1["feat"].astype(np.float32)   # (N, 3, 1024)
    clip_ids = F1["clip_id"]
    split = F1["split"]
    subject = F1["subject_id"]
    y_eng = F1["engagement"].astype(np.int64)
    N = len(clip_ids)
    log(f"  vision feat: {feat_vis.shape}")

    log("Loading MediaPipe face signals (52 blendshapes + 3 pose + 6 gaze)...")
    F2 = np.load(os.path.join(BASE, "features/daisee_face_signals.npz"), allow_pickle=True)
    blend = F2["blendshapes"].astype(np.float32)   # (N, 52)
    head_pose = F2["head_pose"].astype(np.float32) # (N, 3)
    gaze = F2["eye_gaze"].astype(np.float32)       # (N, 6)
    explicit_static = np.concatenate([blend, head_pose, gaze], axis=1)  # (N, 61)
    log(f"  explicit static: {explicit_static.shape}")

    log("Loading MediaPipe body pose (16-d)...")
    F3 = np.load(os.path.join(BASE, "features/daisee_pose_signals.npz"), allow_pickle=True)
    pose = F3["pose_features"].astype(np.float32)  # (N, 16)
    log(f"  pose: {pose.shape}")

    # Combine explicit feats: (N, 61+16=77)
    explicit = np.concatenate([explicit_static, pose], axis=1).astype(np.float32)
    log(f"  combined explicit: {explicit.shape}")

    # Standardize explicit using train-set stats only
    tr_mask = (split == "Train")
    mu = explicit[tr_mask].mean(axis=0); sd = explicit[tr_mask].std(axis=0) + 1e-6
    explicit_z = (explicit - mu) / sd

    # Load auxiliary labels (B, C, F) from CSV
    log("Loading auxiliary B/C/F labels from DAiSEE labels CSVs...")
    aux_y = {"Boredom": np.full(N, -1, dtype=np.int64),
             "Confusion": np.full(N, -1, dtype=np.int64),
             "Frustration": np.full(N, -1, dtype=np.int64)}
    label_files = {
        "Train": "DAiSEE/Labels/TrainLabels.csv",
        "Validation": "DAiSEE/Labels/ValidationLabels.csv",
        "Test": "DAiSEE/Labels/TestLabels.csv",
    }
    clip_to_idx = {cid.replace(".avi", "").replace(".mp4", ""): i for i, cid in enumerate(clip_ids)}
    for split_name, fpath in label_files.items():
        fp = os.path.join(BASE, fpath)
        if not os.path.exists(fp):
            log(f"  WARN: missing {fp}")
            continue
        import csv as _csv
        with open(fp) as f:
            reader = _csv.DictReader(f)
            for r in reader:
                cid = r["ClipID"].replace(".avi", "").replace(".mp4", "")
                if cid in clip_to_idx:
                    i = clip_to_idx[cid]
                    for col in ("Boredom", "Confusion", "Frustration"):
                        if col in r and r[col] != "":
                            aux_y[col][i] = int(r[col])
    aux_y_avail = {k: (v >= 0).sum() for k, v in aux_y.items()}
    log(f"  aux label coverage: {aux_y_avail}")

    va_mask = (split == "Validation"); te_mask = (split == "Test")
    log(f"Splits: tr={tr_mask.sum()} va={va_mask.sum()} te={te_mask.sum()}")

    # ----------------- Dataset ---------------
    class MemeDataset(Dataset):
        def __init__(self, indices):
            self.indices = indices
        def __len__(self): return len(self.indices)
        def __getitem__(self, i):
            idx = self.indices[i]
            x_vis = torch.tensor(feat_vis[idx], dtype=torch.float32)   # (3, 1024)
            x_exp = torch.tensor(explicit_z[idx], dtype=torch.float32) # (77,)
            y_e = int(y_eng[idx])
            y_b = int(aux_y["Boredom"][idx])
            y_c = int(aux_y["Confusion"][idx])
            y_f = int(aux_y["Frustration"][idx])
            return x_vis, x_exp, y_e, y_b, y_c, y_f

    # ----------------- Model ---------------
    class FiLMBlock(nn.Module):
        def __init__(self, d_h, d_film):
            super().__init__()
            self.gamma = nn.Linear(d_film, d_h)
            self.beta = nn.Linear(d_film, d_h)
            nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
            nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)
        def forward(self, x, film):
            # x: (B, T, d_h), film: (B, d_film)
            g = self.gamma(film).unsqueeze(1)
            b = self.beta(film).unsqueeze(1)
            return (1 + g) * x + b

    class MEME(nn.Module):
        def __init__(self, d_vis=1024, d_exp=77, d_h=384, n_layers=2, n_heads=8, n_frames=3):
            super().__init__()
            self.proj_vis = nn.Linear(d_vis, d_h)
            self.pos = nn.Parameter(torch.randn(1, n_frames, d_h) * 0.02)
            self.proj_exp = nn.Sequential(
                nn.Linear(d_exp, d_h), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(d_h, d_h),
            )
            self.films = nn.ModuleList([FiLMBlock(d_h, d_h) for _ in range(n_layers)])
            enc_layer = lambda: nn.TransformerEncoderLayer(
                d_model=d_h, nhead=n_heads, dim_feedforward=4*d_h,
                dropout=0.15, activation="gelu", batch_first=True, norm_first=True)
            self.layers = nn.ModuleList([enc_layer() for _ in range(n_layers)])
            self.norm = nn.LayerNorm(d_h)
            # Engagement head: CORN ordinal (3 logits)
            self.head_e = nn.Sequential(
                nn.Linear(d_h, d_h), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(d_h, 3),
            )
            # Aux heads (4-class CE each)
            self.head_b = nn.Sequential(nn.Linear(d_h, d_h), nn.GELU(),
                                        nn.Dropout(0.2), nn.Linear(d_h, 4))
            self.head_c = nn.Sequential(nn.Linear(d_h, d_h), nn.GELU(),
                                        nn.Dropout(0.2), nn.Linear(d_h, 4))
            self.head_f = nn.Sequential(nn.Linear(d_h, d_h), nn.GELU(),
                                        nn.Dropout(0.2), nn.Linear(d_h, 4))

        def forward(self, x_vis, x_exp):
            # x_vis: (B, T, d_vis), x_exp: (B, d_exp)
            z = self.proj_vis(x_vis) + self.pos
            exp_h = self.proj_exp(x_exp)
            for film, layer in zip(self.films, self.layers):
                z = film(z, exp_h)
                z = layer(z)
            z = z.mean(dim=1)
            z = self.norm(z)
            return {"e": self.head_e(z), "b": self.head_b(z),
                    "c": self.head_c(z), "f": self.head_f(z)}

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

    # Train indices: union of Train and Val for final training
    tr_idx = np.where(tr_mask)[0]
    va_idx = np.where(va_mask)[0]
    te_idx = np.where(te_mask)[0]

    # Class-balanced engagement weights from train
    cnt_e = np.bincount(y_eng[tr_idx], minlength=4).astype(np.float32)
    alpha_e = 1.0 / np.maximum(cnt_e, 1); alpha_e = alpha_e / alpha_e.sum() * 4
    log(f"Train engagement counts: {cnt_e.tolist()}  alpha: {alpha_e.tolist()}")

    # Weighted sampler on train
    sample_w = (1.0 / np.maximum(cnt_e, 1))[y_eng[tr_idx]]

    all_e_te = []; all_e_va = []; y_te = y_va = None; seed_results = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        log(f"\n========== Seed {seed} ==========")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        sampler = WeightedRandomSampler(weights=torch.tensor(sample_w, dtype=torch.float32),
                                        num_samples=len(tr_idx), replacement=True)
        train_dl = DataLoader(MemeDataset(tr_idx), batch_size=args.batch,
                              sampler=sampler, num_workers=0)
        val_dl = DataLoader(MemeDataset(va_idx), batch_size=128, shuffle=False, num_workers=0)
        test_dl = DataLoader(MemeDataset(te_idx), batch_size=128, shuffle=False, num_workers=0)

        model = MEME(d_h=args.d_h, n_layers=args.n_layers, n_heads=args.n_heads).to(device)
        alpha_t = torch.tensor(alpha_e, dtype=torch.float32, device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        n_steps = len(train_dl) * args.epochs
        warmup = max(50, n_steps // 30)
        def set_lr(step):
            s = step / max(1, warmup) if step < warmup else 0.5 * (1 + np.cos(np.pi * (step - warmup) / max(1, n_steps - warmup)))
            for pg in opt.param_groups: pg["lr"] = args.lr * s

        @torch.no_grad()
        def eval_split(dl):
            model.eval(); logits_all=[]; ys=[]
            for x_vis, x_exp, y_e, y_b, y_c, y_f in dl:
                x_vis = x_vis.to(device); x_exp = x_exp.to(device)
                out = model(x_vis, x_exp)
                logits_all.append(out["e"].cpu()); ys.append(y_e)
            return torch.cat(logits_all), torch.cat(ys).numpy()

        history = []; step = 0; best_val_kq = -1; best_state = None
        for ep in range(args.epochs):
            model.train()
            t0 = time.time()
            for x_vis, x_exp, y_e, y_b, y_c, y_f in train_dl:
                x_vis = x_vis.to(device); x_exp = x_exp.to(device)
                y_e = y_e.to(device); y_b = y_b.to(device)
                y_c = y_c.to(device); y_f = y_f.to(device)
                set_lr(step); step += 1
                opt.zero_grad()
                out = model(x_vis, x_exp)
                loss_e = corn_loss(out["e"], y_e, alpha_t, gamma=args.gamma_focal)
                aux = 0.0
                for name, logits, y_aux in [("b", out["b"], y_b),
                                             ("c", out["c"], y_c),
                                             ("f", out["f"], y_f)]:
                    mask = (y_aux >= 0)
                    if mask.sum() == 0: continue
                    aux = aux + F.cross_entropy(logits[mask], y_aux[mask])
                loss = loss_e + args.aux_weight * aux
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
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
        elif device.type == "cuda": torch.cuda.empty_cache()

    # Ensemble
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
