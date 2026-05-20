"""
MOONSHOT 7: CORN (cumulative-link ordinal MLP) bagged on SigLIP-L.

True ordinal regression: predict K-1=3 binary outputs P(y > k) for k=0..2.
At test time, accumulate to get class probabilities.

Architecture:
  Linear(D, 256) → GELU → Dropout(0.3) → Linear(256, 3)  [3 binary outputs]

Loss: binary CE on each of 3 thresholds (with class weights per threshold).

Bagged K=10 × 5 seeds. Apply threshold tuning on derived E[y].

Output:
  results/sota/moonshot_corn.json
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_corn.json")
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


class CORN(nn.Module):
    def __init__(self, D=1024, n_classes=4, h=256, dropout=0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(D, h), nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(h, n_classes - 1)  # 3 thresholds for 4 classes
    def forward(self, x):
        return self.head(self.trunk(x))


def corn_loss(logits, y, n_classes=4):
    """Cumulative-link binary CE."""
    # y > k for k = 0, 1, 2
    targets = (y.unsqueeze(1) > torch.arange(n_classes - 1, device=y.device).unsqueeze(0)).float()
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    # mask: only consider thresholds up to and including k=y (CORN constraint)
    # Actually full CORN uses all thresholds; the proper "CORN loss" by Cao et al.
    # masks at conditional, but simple unconstrained binary CE works empirically.
    return losses.mean()


def corn_to_probs(logits):
    """Convert 3 cumulative-link logits to 4-class probs."""
    # p(y > k) = sigmoid(logit_k)
    # p(y = k) = p(y > k-1) - p(y > k), with conventions:
    #   p(y > -1) = 1, p(y > K-1) = 0
    p_gt = torch.sigmoid(logits)  # (B, 3): p(y>0), p(y>1), p(y>2)
    B = p_gt.shape[0]
    p = torch.zeros(B, 4, device=p_gt.device)
    p[:, 0] = 1 - p_gt[:, 0]
    p[:, 1] = p_gt[:, 0] - p_gt[:, 1]
    p[:, 2] = p_gt[:, 1] - p_gt[:, 2]
    p[:, 3] = p_gt[:, 2]
    # Clamp negative values to 0 and renormalize
    p = torch.clamp(p, min=0)
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-12)
    return p


def train_corn(Xtr, ytr, Xva, yva, n_epochs=60, bs=128, lr=2e-3, seed=42, h=256, dropout=0.3):
    torch.manual_seed(seed); np.random.seed(seed)
    D = Xtr.shape[1]
    model = CORN(D=D, h=h, dropout=dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

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
            loss = corn_loss(logits, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            p_va = corn_to_probs(model(Xv)).cpu().numpy()
        kq = cohen_kappa_score(yva, p_va.argmax(1), weights="quadratic")
        if kq > best_va_kq:
            best_va_kq = kq
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_va_kq


def bag_corn(Xtr, ytr, Xva, yva, Xte, K=10, seeds=[0, 7, 42, 2025, 1024], **kwargs):
    all_te, all_va = [], []
    for so in seeds:
        rng = np.random.default_rng(so)
        bt, bv = [], []
        for k in range(K):
            t0 = time.time()
            idx = rng.integers(0, len(ytr), size=len(ytr))
            m, va_kq = train_corn(Xtr[idx], ytr[idx], Xva, yva, seed=so*1000+k, **kwargs)
            with torch.no_grad():
                Xv_t = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
                Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
                pv = corn_to_probs(m(Xv_t)).cpu().numpy()
                pte = corn_to_probs(m(Xte_t)).cpu().numpy()
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
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    Xtr, ytr = X[tr], eng[tr]; Xva, yva = X[va], eng[va]; Xte, yte = X[te], eng[te]

    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    out = {}

    print("\nBagged CORN (K=10 × 5 seeds)...", flush=True)
    t0 = time.time()
    p_te, p_va = bag_corn(Xtr, ytr, Xva, yva, Xte, K=10)
    print(f"  Done ({time.time()-t0:.0f}s)", flush=True)

    m_solo = metrics(yte, p_te.argmax(1))
    e_va = (p_va * classes[None, :]).sum(1)
    e_te = (p_te * classes[None, :]).sum(1)
    bt = tune_thresh(e_va, yva)
    m_thr = metrics(yte, apply_t(e_te, bt["t"]))
    out["solo"] = m_solo
    out["threshold"] = {**m_thr, **bt}
    print(f"\nSolo CORN: κ_q={m_solo['kappa_q']:.4f}", flush=True)
    print(f"+threshold: κ_q={m_thr['kappa_q']:.4f} {m_thr['kappa_q_ci']}  t={bt['t']}", flush=True)

    # Save cache
    np.savez_compressed(os.path.join(BASE, "results", "sota", "_corn_bag_cache.npz"),
                        p_te=p_te, p_va=p_va)

    # Fusion with cached LR-unif
    print("\nFusion with cached LR-unif...", flush=True)
    c = np.load(CACHE)
    p_te_lr = c["p_te_unif"]; p_va_lr = c["p_va_unif"]
    best_w = None
    for w in np.linspace(0, 1, 21):
        pv = w * p_va_lr + (1 - w) * p_va
        e = (pv * classes[None, :]).sum(1)
        bt2 = tune_thresh(e, yva)
        if best_w is None or bt2["v"] > best_w["v"]:
            best_w = {"w_lr": float(w), **bt2}
    wl = best_w["w_lr"]
    p_te_f = wl * p_te_lr + (1 - wl) * p_te
    e_te_f = (p_te_f * classes[None, :]).sum(1)
    m_f = metrics(yte, apply_t(e_te_f, best_w["t"]))
    out["fusion_lr_corn"] = {**m_f, "w_lr": wl, **best_w}
    print(f"Fusion: w_lr={wl:.2f} t={best_w['t']} κ_q={m_f['kappa_q']:.4f} {m_f['kappa_q_ci']}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
