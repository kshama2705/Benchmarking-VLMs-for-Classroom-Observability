"""
TEAM — Temporal Engagement via Attention Modulation

Small temporal transformer head trained on top of 3-frame frozen SigLIP-L
features (t = 2, 5, 8 seconds) with ordinal CORN loss + class-balanced
sampling.

Architecture:
    x: (B, 3, 1024)
    project: Linear(1024, d_h)           → (B, 3, d_h)   d_h=256
    add learned positional emb (3, d_h)
    transformer encoder: 2 layers, 4 heads, d_h=256, FFN=1024, GELU, drop=0.1
    pool: mean over T (or learned [CLS] token)
    head: LayerNorm → Linear(d_h, d_h) → GELU → Drop → Linear(d_h, 3)
                          # CORN-style 3 cumulative-threshold logits

Loss: CORN ordinal loss (Shi et al. 2023).
- 4 classes → 3 binary tasks: P(y > 0), P(y > 1), P(y > 2)
- Each binary task uses BCE with class-balanced weights derived from train.

Training: 25 epochs, batch 64, AdamW lr=1e-3 + cosine, weight_decay=0.01,
warmup 100 steps. Class-balanced sampling: sample weight = 1/freq(y).
Save best by val κ_q on argmax decoding of CORN logits.

Multi-seed (3 seeds), ensemble predictions at test time.

Targets test κ_q > 0.27 to break the frozen-feature ceiling.
"""

import os, json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import cohen_kappa_score, accuracy_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(BASE, "features", "daisee_siglip_l_multiframe_features.npz")
OUT = os.path.join(BASE, "results", "sota", "team_results.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))


class TempDataset(Dataset):
    def __init__(self, X, y, indices=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.indices = indices if indices is not None else np.arange(len(X))

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return self.X[idx], self.y[idx]


class TEAM(nn.Module):
    def __init__(self, d_in=1024, d_h=256, n_layers=2, n_heads=4, n_frames=3, n_classes=4, drop=0.1):
        super().__init__()
        self.n_frames = n_frames
        self.proj = nn.Linear(d_in, d_h)
        self.pos = nn.Parameter(torch.randn(1, n_frames, d_h) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_h, nhead=n_heads, dim_feedforward=4*d_h,
            dropout=drop, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_h)
        self.head = nn.Sequential(
            nn.Linear(d_h, d_h), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d_h, n_classes - 1),  # 3 CORN logits for 4 classes
        )

    def forward(self, x):
        # x: (B, T, d_in)
        z = self.proj(x) + self.pos
        z = self.encoder(z)
        z = z.mean(dim=1)  # (B, d_h)
        z = self.norm(z)
        return self.head(z)  # (B, K-1)


def corn_loss(logits, y, n_classes=4, pos_weight=None):
    """CORN ordinal loss: each k in {0..K-2} is a binary classifier of P(y > k).
    logits: (B, K-1)
    y: (B,) in {0..K-1}
    """
    B = y.shape[0]
    K = n_classes
    losses = []
    for k in range(K - 1):
        target_k = (y > k).float()
        l = logits[:, k]
        if pos_weight is not None:
            loss_k = F.binary_cross_entropy_with_logits(l, target_k, pos_weight=pos_weight[k])
        else:
            loss_k = F.binary_cross_entropy_with_logits(l, target_k)
        losses.append(loss_k)
    return torch.stack(losses).mean()


def corn_predict(logits):
    """CORN decoding: predicted class = number of binary tasks where P(y > k) > 0.5."""
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1)


def corn_expected_class(logits):
    """Expected class score = sum_k P(y > k)."""
    probs = torch.sigmoid(logits)
    return probs.sum(dim=1)


def tune_thresholds(e_va, y_va, step=0.04):
    grid = np.arange(0.0, 3.01, step)
    best = None
    for t1 in grid:
        for t2 in grid[grid > t1]:
            for t3 in grid[grid > t2]:
                yp = np.zeros_like(e_va, dtype=int)
                yp[e_va > t1] = 1; yp[e_va > t2] = 2; yp[e_va > t3] = 3
                v = cohen_kappa_score(y_va, yp, weights="quadratic")
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


def train_one(X, y, splits, seed=42, epochs=25, batch=64, d_h=256, n_layers=2, n_heads=4, drop=0.1, lr=1e-3, weight_decay=0.01):
    torch.manual_seed(seed); np.random.seed(seed)
    device = DEVICE

    tr_idx = np.where(splits == "Train")[0]
    va_idx = np.where(splits == "Validation")[0]
    te_idx = np.where(splits == "Test")[0]

    # Class-balanced sampling weights
    y_tr = y[tr_idx]
    cnt = np.bincount(y_tr, minlength=4).astype(np.float32)
    cls_w = 1.0 / np.maximum(cnt, 1)
    sample_w = cls_w[y_tr]
    sampler = WeightedRandomSampler(weights=torch.tensor(sample_w, dtype=torch.float32),
                                    num_samples=len(tr_idx), replacement=True)

    # CORN pos_weight: per binary task k, balance positives (y>k) vs negatives.
    K = 4
    pw_list = []
    for k in range(K - 1):
        pos = (y_tr > k).sum(); neg = ((y_tr <= k)).sum()
        pw_list.append(torch.tensor(float(neg) / max(1.0, float(pos)), device=device))

    train_ds = TempDataset(X, y, indices=tr_idx)
    val_ds = TempDataset(X, y, indices=va_idx)
    test_ds = TempDataset(X, y, indices=te_idx)
    train_dl = DataLoader(train_ds, batch_size=batch, sampler=sampler, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    model = TEAM(d_h=d_h, n_layers=n_layers, n_heads=n_heads, drop=drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_steps = len(train_dl) * epochs
    warmup = max(50, n_steps // 30)

    def set_lr(step):
        if step < warmup:
            s = step / max(1, warmup)
        else:
            prog = (step - warmup) / max(1, n_steps - warmup)
            s = 0.5 * (1 + np.cos(np.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr * s

    @torch.no_grad()
    def eval_split(dl):
        model.eval()
        all_logits = []; all_y = []
        for xb, yb in dl:
            xb = xb.to(device); logits = model(xb).cpu()
            all_logits.append(logits); all_y.append(yb)
        L = torch.cat(all_logits); Y = torch.cat(all_y).numpy()
        e = corn_expected_class(L).numpy()
        yp = corn_predict(L).numpy()
        return e, yp, Y

    best_val_kq = -1; best_state = None; history = []
    step = 0
    for ep in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device); yb = yb.to(device)
            set_lr(step); step += 1
            opt.zero_grad()
            logits = model(xb)
            loss = corn_loss(logits, yb, n_classes=4, pos_weight=pw_list)
            loss.backward()
            opt.step()
        # eval
        e_va, yp_va_arg, y_va = eval_split(val_dl)
        e_te, yp_te_arg, y_te = eval_split(test_dl)
        bt = tune_thresholds(e_va, y_va)
        yp_te_thr = apply_thr(e_te, bt['t'])
        val_kq_arg = cohen_kappa_score(y_va, yp_va_arg, weights="quadratic")
        test_kq_arg = cohen_kappa_score(y_te, yp_te_arg, weights="quadratic")
        test_kq_thr = cohen_kappa_score(y_te, yp_te_thr, weights="quadratic")
        history.append({"ep": ep, "val_kq_arg": float(val_kq_arg),
                        "val_kq_thr": float(bt['v']),
                        "test_kq_arg": float(test_kq_arg),
                        "test_kq_thr": float(test_kq_thr),
                        "thresholds": bt['t']})
        print(f"  [ep{ep:02d}] val_arg={val_kq_arg:.4f}  val_thr={bt['v']:.4f}  test_arg={test_kq_arg:.4f}  test_thr={test_kq_thr:.4f}", flush=True)
        if bt['v'] > best_val_kq:
            best_val_kq = bt['v']
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    # Final predictions with best state
    e_te, yp_te_arg, y_te = eval_split(test_dl)
    e_va, _, y_va = eval_split(val_dl)
    bt = tune_thresholds(e_va, y_va)
    yp_te_thr = apply_thr(e_te, bt['t'])
    test_kq_arg = float(cohen_kappa_score(y_te, yp_te_arg, weights="quadratic"))
    test_kq_thr = float(cohen_kappa_score(y_te, yp_te_thr, weights="quadratic"))
    print(f"  [seed {seed} best] test_arg={test_kq_arg:.4f}  test_thr={test_kq_thr:.4f}  best_val_thr_kq={best_val_kq:.4f}", flush=True)
    # Also return the probs for ensembling
    with torch.no_grad():
        model.eval()
        # full-test logits in one pass
        full_logits = []
        for xb, yb in test_dl:
            full_logits.append(model(xb.to(device)).cpu())
        L_te = torch.cat(full_logits).numpy()
        full_logits = []
        for xb, yb in val_dl:
            full_logits.append(model(xb.to(device)).cpu())
        L_va = torch.cat(full_logits).numpy()
    return {"history": history,
            "test_kq_arg": test_kq_arg, "test_kq_thr": test_kq_thr,
            "best_val_thr_kq": best_val_kq, "thresholds": bt['t'],
            "L_te": L_te, "L_va": L_va, "y_te": y_te, "y_va": y_va}


def main():
    print(f"Device: {DEVICE}", flush=True)
    d = np.load(FEAT, allow_pickle=True)
    feat = d['feat'].astype(np.float32)  # (N, 3, 1024)
    splits = d['split']; y = d['engagement'].astype(np.int64)
    if 'success' in d:
        succ = d['success'].astype(bool)
    else:
        succ = np.ones(len(feat), dtype=bool)
    print(f"feat shape: {feat.shape}  success: {succ.sum()}/{len(succ)}", flush=True)
    # Use only successfully-encoded clips for train; full test (we already established test=100%)
    # But ensure non-zero rows
    nonzero = (feat != 0).any(axis=(1,2))
    print(f"Non-zero rows: {nonzero.sum()}", flush=True)
    # Restrict to non-zero in train/val; test is full
    keep = nonzero | (splits == "Test")  # always keep all test
    print(f"Keep: {keep.sum()}/{len(keep)}", flush=True)
    feat_k = feat[keep]; splits_k = splits[keep]; y_k = y[keep]
    print(f"After filter: tr={(splits_k=='Train').sum()} va={(splits_k=='Validation').sum()} te={(splits_k=='Test').sum()}", flush=True)

    seeds = [0, 42, 2025]
    results = []
    L_te_list = []; L_va_list = []; y_te = y_va = None
    for s in seeds:
        print(f"\n=== Seed {s} ===", flush=True)
        r = train_one(feat_k, y_k, splits_k, seed=s, epochs=25)
        results.append({k: v for k, v in r.items() if k not in ("L_te", "L_va", "y_te", "y_va")})
        L_te_list.append(r["L_te"]); L_va_list.append(r["L_va"])
        y_te = r["y_te"]; y_va = r["y_va"]

    # Ensemble
    print("\n=== ENSEMBLE ===", flush=True)
    Lte = np.mean(L_te_list, axis=0); Lva = np.mean(L_va_list, axis=0)
    # Convert to expected-class via sigmoid sum
    e_te = (1 / (1 + np.exp(-Lte))).sum(axis=1)
    e_va = (1 / (1 + np.exp(-Lva))).sum(axis=1)
    bt = tune_thresholds(e_va, y_va)
    yp_te_thr = apply_thr(e_te, bt['t'])
    yp_te_arg = (1 / (1 + np.exp(-Lte)) > 0.5).sum(axis=1)
    kq_arg = float(cohen_kappa_score(y_te, yp_te_arg, weights="quadratic"))
    kq_thr = float(cohen_kappa_score(y_te, yp_te_thr, weights="quadratic"))
    ci_thr = boot_ci(y_te, yp_te_thr)
    print(f"ENSEMBLE test_arg={kq_arg:.4f}  test_thr={kq_thr:.4f}  ci={ci_thr}", flush=True)

    out = {"per_seed": results, "ensemble": {"test_arg_kq": kq_arg, "test_thr_kq": kq_thr,
                                              "ci_thr": ci_thr, "thresholds": bt['t']}}
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
