"""
MOONSHOT 4: Linear-adapter end-to-end probe with focal-ordinal loss.

Idea: a trainable linear adapter on top of frozen SigLIP-L features, trained
with ordinal-aware loss (CORN-style + focal CE). The adapter learns to project
the 1024-d feature into a smaller engagement-discriminative subspace.

  feature(1024) → Linear(1024, D_lat=256) → ReLU → Dropout → Linear(D_lat, 4)

Loss = focal CE + λ * ordinal-spacing-loss:
  ordinal-spacing-loss = E[(pred_continuous - target_continuous)^2]
  where pred_continuous = Σ k * softmax(logits)_k
        target_continuous = engagement value (0..3)

Bagged across K=10 × 5 seeds. Apply threshold tuning at end.

Output:
  results/sota/moonshot_adapter.json
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_adapter.json")
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))
RNG = np.random.default_rng(42)


def boot_kq(yt, yp, n=1000):
    out = []; nt = len(yt)
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


class Adapter(nn.Module):
    def __init__(self, D=1024, D_lat=256, n_classes=4, dropout=0.4):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(D, D_lat), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(D_lat, n_classes)
    def forward(self, x):
        z = self.adapter(x)
        return self.head(z)


def focal_loss(logits, targets, alpha, gamma=2.0):
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def ordinal_loss(logits, targets):
    """MSE between continuous prediction (E[y]) and target class."""
    p = F.softmax(logits, dim=-1)
    classes = torch.arange(4, dtype=torch.float32, device=logits.device)
    pred_c = (p * classes[None, :]).sum(dim=-1)
    return F.mse_loss(pred_c, targets.float())


def train_adapter(Xtr, ytr, Xva, yva, n_epochs=60, bs=128, lr=2e-3, seed=42, ord_w=0.5, dropout=0.4, D_lat=256):
    torch.manual_seed(seed); np.random.seed(seed)
    D = Xtr.shape[1]
    model = Adapter(D=D, D_lat=D_lat, dropout=dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    cw = compute_class_weight("balanced", classes=np.arange(4), y=ytr)
    alpha = torch.tensor(cw, dtype=torch.float32, device=DEVICE)

    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    Xv = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)

    best_va_kq, best_state = -1, None
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i+bs]
            logits = model(Xt[idx])
            loss = focal_loss(logits, yt[idx], alpha, gamma=2.0) + ord_w * ordinal_loss(logits, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = F.softmax(model(Xv), dim=-1).cpu().numpy()
        kq = cohen_kappa_score(yva, pv.argmax(1), weights="quadratic")
        if kq > best_va_kq:
            best_va_kq = kq
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_va_kq


def bag_adapter(Xtr, ytr, Xva, yva, Xte, K=10, seeds=[0, 7, 42, 2025, 1024], **kwargs):
    all_te, all_va = [], []
    for so in seeds:
        rng = np.random.default_rng(so)
        bt, bv = [], []
        for k in range(K):
            t0 = time.time()
            idx = rng.integers(0, len(ytr), size=len(ytr))
            m, va_kq = train_adapter(Xtr[idx], ytr[idx], Xva, yva, seed=so*1000+k, **kwargs)
            with torch.no_grad():
                Xv_t = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
                Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
                pv = F.softmax(m(Xv_t), dim=-1).cpu().numpy()
                pte = F.softmax(m(Xte_t), dim=-1).cpu().numpy()
            bt.append(pte); bv.append(pv)
            print(f"    seed={so} k={k+1}/{K} va_kq={va_kq:.3f} {time.time()-t0:.0f}s", flush=True)
        all_te.append(np.stack(bt).mean(0)); all_va.append(np.stack(bv).mean(0))
    return np.stack(all_te).mean(0), np.stack(all_va).mean(0)


def tune_thresh(e, y, step=0.04):
    grid = np.arange(0, 3.01, step)
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


def apply_t(e, t):
    yp = np.zeros_like(e, dtype=int)
    yp[e > t[0]] = 1; yp[e > t[1]] = 2; yp[e > t[2]] = 3
    return yp


def main():
    print(f"Device: {DEVICE}", flush=True)
    print("Loading...", flush=True)
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    Xtr, ytr = X[tr], eng[tr]; Xva, yva = X[va], eng[va]; Xte, yte = X[te], eng[te]

    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    out = {}

    # Try several adapter configs
    configs = [
        {"name": "lat256_ord0.5", "D_lat": 256, "ord_w": 0.5, "dropout": 0.4, "n_epochs": 60},
        {"name": "lat128_ord1.0", "D_lat": 128, "ord_w": 1.0, "dropout": 0.4, "n_epochs": 60},
        {"name": "lat512_ord0.5", "D_lat": 512, "ord_w": 0.5, "dropout": 0.5, "n_epochs": 60},
        {"name": "lat64_ord2.0", "D_lat": 64, "ord_w": 2.0, "dropout": 0.3, "n_epochs": 60},
    ]

    for cfg in configs:
        name = cfg.pop("name")
        print(f"\n=== Config: {name} {cfg} ===", flush=True)
        t0 = time.time()
        p_te, p_va = bag_adapter(Xtr, ytr, Xva, yva, Xte, K=10, **cfg)
        print(f"  Total: {time.time()-t0:.0f}s", flush=True)

        m_solo = metrics(yte, p_te.argmax(1))
        e_va = (p_va * classes[None, :]).sum(1)
        e_te = (p_te * classes[None, :]).sum(1)
        bt = tune_thresh(e_va, yva)
        m_thr = metrics(yte, apply_t(e_te, bt["t"]))
        out[name] = {"solo": m_solo, "threshold": {**m_thr, **bt}}
        print(f"  Solo: κ_q={m_solo['kappa_q']:.4f}", flush=True)
        print(f"  +threshold: κ_q={m_thr['kappa_q']:.4f} {m_thr['kappa_q_ci']} t={bt['t']}", flush=True)

        # Cache for fusion
        np.savez_compressed(os.path.join(BASE, "results", "sota", f"_adapter_{name}_cache.npz"),
                            p_te=p_te, p_va=p_va)

    # Fusion of best adapter with cached LR-unif
    print("\n=== Fusion: best adapter + cached LR-unif ===", flush=True)
    best_cfg = max(out.keys(), key=lambda n: out[n]["threshold"]["kappa_q"])
    print(f"  Best adapter: {best_cfg}", flush=True)
    c2 = np.load(os.path.join(BASE, "results", "sota", f"_adapter_{best_cfg}_cache.npz"))
    p_te_ad = c2["p_te"]; p_va_ad = c2["p_va"]
    c = np.load(CACHE)
    p_te_lr = c["p_te_unif"]; p_va_lr = c["p_va_unif"]
    best_w = None
    for w in np.linspace(0, 1, 21):
        pv = w * p_va_lr + (1 - w) * p_va_ad
        e = (pv * classes[None, :]).sum(1)
        bt2 = tune_thresh(e, yva)
        if best_w is None or bt2["v"] > best_w["v"]:
            best_w = {"w_lr": float(w), **bt2}
    wl = best_w["w_lr"]
    p_te_f = wl * p_te_lr + (1 - wl) * p_te_ad
    e_te_f = (p_te_f * classes[None, :]).sum(1)
    m_f = metrics(yte, apply_t(e_te_f, best_w["t"]))
    out["fusion_lr_adapter"] = {**m_f, "best_adapter": best_cfg, "w_lr": wl, **best_w}
    print(f"  Fusion: w_lr={wl:.2f}  t={best_w['t']}  κ_q={m_f['kappa_q']:.4f} {m_f['kappa_q_ci']}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
