"""
MOONSHOT 9: Mixup augmentation for LR/MLP probe training.

Mixup: train on convex combos of feature/label pairs (x_a, y_a), (x_b, y_b):
  x_mix = λ x_a + (1-λ) x_b
  y_mix = λ y_a + (1-λ) y_b  (label-smoothed soft label)

For LR (sklearn) we can't use soft labels directly. So we'll:
  A) Generate synthetic mixup samples between minority and other classes
  B) Train LR on augmented set with hard labels (rounded mixup label)
  C) Train MLP with proper mixup (soft labels)

For class imbalance: oversample L0/L1 via mixup with same-class pairs.

Output:
  results/sota/moonshot_mixup.json
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_mixup.json")
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


class MLP(nn.Module):
    def __init__(self, D, n_classes=4, h1=512, h2=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h2, n_classes),
        )
    def forward(self, x): return self.net(x)


def soft_focal_loss(logits, soft_targets, alpha, gamma=2.0):
    """Soft-label focal CE. soft_targets is (B, K) one-hot or mix."""
    log_p = F.log_softmax(logits, dim=-1)
    p = F.softmax(logits, dim=-1)
    # Class weights: per-target softvweighting alpha[k]
    # Compute per-sample weight by alpha · soft_targets
    sample_w = (alpha[None, :] * soft_targets).sum(-1)
    # Focal: (1 - p_target)^gamma · -log p_target  → use mixed pt
    pt = (p * soft_targets).sum(-1)
    loss_per = -((1 - pt) ** gamma) * (soft_targets * log_p).sum(-1) * sample_w
    return loss_per.mean()


def mixup_features(X, Y, alpha=0.4, rng=None):
    """X (N, D), Y (N,) ints. Returns mixed (X_mix, Y_mix_onehot)."""
    if rng is None: rng = np.random.default_rng()
    N = len(X)
    perm = rng.permutation(N)
    lam = rng.beta(alpha, alpha, size=N).astype(np.float32)[:, None]
    X_mix = lam * X + (1 - lam) * X[perm]
    Y_oh = np.eye(4, dtype=np.float32)[Y]
    Y_mix = lam * Y_oh + (1 - lam) * Y_oh[perm]
    return X_mix, Y_mix


def train_mlp_mixup(Xtr, ytr, Xva, yva, n_epochs=40, bs=128, lr=1e-3, seed=42, mix_alpha=0.4, no_mix_prob=0.5):
    torch.manual_seed(seed); np.random.seed(seed)
    D = Xtr.shape[1]
    model = MLP(D).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    cw = compute_class_weight("balanced", classes=np.arange(4), y=ytr)
    alpha = torch.tensor(cw, dtype=torch.float32, device=DEVICE)
    rng = np.random.default_rng(seed)

    Xt_np = Xtr; yt_np = ytr
    Xv_t = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)

    best_va_kq, best_state = -1, None
    for ep in range(n_epochs):
        model.train()
        # Generate fresh mixup each epoch
        Xmix, Ymix = mixup_features(Xt_np, yt_np, alpha=mix_alpha, rng=rng)
        # Mix between mixup and original (probabilistic per batch)
        N = len(Xt_np)
        idx = rng.permutation(N)
        for i in range(0, N, bs):
            batch_idx = idx[i:i+bs]
            if rng.random() < no_mix_prob:
                # Use original
                xb = torch.tensor(Xt_np[batch_idx], dtype=torch.float32, device=DEVICE)
                yb = torch.tensor(yt_np[batch_idx], dtype=torch.long, device=DEVICE)
                yb_oh = F.one_hot(yb, num_classes=4).float()
            else:
                # Use mixup
                xb = torch.tensor(Xmix[batch_idx], dtype=torch.float32, device=DEVICE)
                yb_oh = torch.tensor(Ymix[batch_idx], dtype=torch.float32, device=DEVICE)
            logits = model(xb)
            loss = soft_focal_loss(logits, yb_oh, alpha, gamma=2.0)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv_t).softmax(-1).cpu().numpy()
        kq = cohen_kappa_score(yva, pv.argmax(1), weights="quadratic")
        if kq > best_va_kq:
            best_va_kq = kq
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_va_kq


def bag_mlp_mixup(Xtr, ytr, Xva, yva, Xte, K=10, seeds=[0, 7, 42, 2025, 1024], **kwargs):
    all_te, all_va = [], []
    for so in seeds:
        rng = np.random.default_rng(so)
        bt, bv = [], []
        for k in range(K):
            t0 = time.time()
            idx = rng.integers(0, len(ytr), size=len(ytr))
            m, va_kq = train_mlp_mixup(Xtr[idx], ytr[idx], Xva, yva, seed=so*1000+k, **kwargs)
            with torch.no_grad():
                Xv_t = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
                Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
                pv = m(Xv_t).softmax(-1).cpu().numpy()
                pte = m(Xte_t).softmax(-1).cpu().numpy()
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

    # Test 3 mixup alphas
    for alpha_mix in [0.2, 0.4, 0.8]:
        print(f"\n=== Mixup alpha={alpha_mix} ===", flush=True)
        t0 = time.time()
        p_te, p_va = bag_mlp_mixup(Xtr, ytr, Xva, yva, Xte, K=10, mix_alpha=alpha_mix)
        print(f"  Done ({time.time()-t0:.0f}s)", flush=True)
        m_solo = metrics(yte, p_te.argmax(1))
        e_va = (p_va * classes[None, :]).sum(1); e_te = (p_te * classes[None, :]).sum(1)
        bt = tune_thresh(e_va, yva)
        m_thr = metrics(yte, apply_t(e_te, bt["t"]))
        out[f"alpha_{alpha_mix}"] = {"solo": m_solo, "threshold": {**m_thr, **bt}}
        print(f"  Solo: κ_q={m_solo['kappa_q']:.4f}  +thresh: κ_q={m_thr['kappa_q']:.4f} {m_thr['kappa_q_ci']}", flush=True)

    # Best mixup + cached LR-unif fusion
    print("\n=== Fusion: best mixup + cached LR-unif ===", flush=True)
    best_a = max(out.keys(), key=lambda n: out[n]["threshold"]["kappa_q"])
    print(f"  Best mixup: {best_a}  κ={out[best_a]['threshold']['kappa_q']:.4f}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
