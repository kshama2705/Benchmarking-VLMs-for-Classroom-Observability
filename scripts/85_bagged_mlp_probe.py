"""
Bagged MLP probe with focal loss + threshold tuning.

Last untried direction at the readout level. SIEP earlier showed single MLP hits
κ ≈ 0.115 mean (worse than linear). Bagging multiple MLP seeds × bootstraps + final
threshold tuning could break or confirm the ceiling.

Architecture:
  Linear(D, 512) → GELU → Dropout(0.3) → Linear(512, 256) → GELU → Dropout(0.3) → Linear(256, 4)

Loss: focal-cross-entropy with class weights (gamma=2, alpha=balanced)
Optimizer: AdamW (lr=1e-3, wd=1e-4), cosine schedule
Bag: K=10 × 5 seeds, bootstrap rows + random init

Output:
  results/sota/bagged_mlp.json
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
MLP_CACHE = os.path.join(BASE, "results", "sota", "_mlp_bag_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "bagged_mlp.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))
print(f"Device: {DEVICE}", flush=True)


def boot_kq(yt, yp, n=1000):
    out = []
    nt = len(yt)
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


def focal_loss(logits, targets, alpha, gamma=2.0):
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def train_mlp(Xtr, ytr, Xva, yva, n_epochs=40, batch_size=128, lr=1e-3, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    D = Xtr.shape[1]
    model = MLP(D).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs)
    cw = compute_class_weight("balanced", classes=np.arange(4), y=ytr)
    alpha = torch.tensor(cw, dtype=torch.float32, device=DEVICE)

    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    Xv = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(yva, dtype=torch.long, device=DEVICE)

    best_va_kq, best_state = -1, None
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i+batch_size]
            logits = model(Xt[idx])
            loss = focal_loss(logits, yt[idx], alpha, gamma=2.0)
            optim.zero_grad(); loss.backward(); optim.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).softmax(-1).cpu().numpy()
        kq = cohen_kappa_score(yva, pv.argmax(1), weights="quadratic")
        if kq > best_va_kq:
            best_va_kq = kq
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # Load best
    model.load_state_dict(best_state)
    return model, best_va_kq


def bag_mlp(Xtr, ytr, Xva, yva, Xte, K=10, seeds=[0, 7, 42, 2025, 1024], n_epochs=40):
    all_te, all_va = [], []
    for s_outer in seeds:
        rng = np.random.default_rng(s_outer)
        bt, bv = [], []
        for k in range(K):
            t0 = time.time()
            idx = rng.integers(0, len(ytr), size=len(ytr))
            model, va_kq = train_mlp(Xtr[idx], ytr[idx], Xva, yva,
                                     n_epochs=n_epochs, seed=s_outer * 1000 + k)
            with torch.no_grad():
                Xv_t = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
                Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
                pv = model(Xv_t).softmax(-1).cpu().numpy()
                pte = model(Xte_t).softmax(-1).cpu().numpy()
            bt.append(pte); bv.append(pv)
            print(f"    seed={s_outer} k={k+1}/{K} va_kq={va_kq:.3f}  {time.time()-t0:.0f}s", flush=True)
        all_te.append(np.stack(bt).mean(0)); all_va.append(np.stack(bv).mean(0))
    return np.stack(all_te).mean(0), np.stack(all_va).mean(0)


def tune_thresholds(e, y, grid_step=0.02):
    grid = np.arange(0.0, 3.01, grid_step)
    best = None
    for t1 in grid:
        for t2 in grid[grid > t1]:
            for t3 in grid[grid > t2]:
                yp = np.zeros_like(e, dtype=int)
                yp[e > t1] = 1; yp[e > t2] = 2; yp[e > t3] = 3
                v = cohen_kappa_score(y, yp, weights="quadratic")
                if best is None or v > best["v"]:
                    best = {"t1": float(t1), "t2": float(t2), "t3": float(t3), "v": float(v)}
    return best


def apply_thresholds(e, t1, t2, t3):
    yp = np.zeros_like(e, dtype=int)
    yp[e > t1] = 1; yp[e > t2] = 2; yp[e > t3] = 3
    return yp


def main():
    print("Loading SigLIP-L features...")
    d = np.load(SIGLIP, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    sp = d["split"]; eng = d["engagement"].astype(np.int64)
    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    Xtr, ytr = X[tr], eng[tr]; Xva, yva = X[va], eng[va]; Xte, yte = X[te], eng[te]

    if os.path.exists(MLP_CACHE):
        m = np.load(MLP_CACHE)
        p_te_mlp = m["p_te"]; p_va_mlp = m["p_va"]
        print(f"  MLP bag loaded from cache: te={p_te_mlp.shape}")
    else:
        print("Training bagged MLP (K=10 × 5 seeds, 40 epochs, focal+balanced)...")
        t0 = time.time()
        p_te_mlp, p_va_mlp = bag_mlp(Xtr, ytr, Xva, yva, Xte, K=10, n_epochs=40)
        print(f"  Total: {time.time()-t0:.0f}s")
        np.savez_compressed(MLP_CACHE, p_te=p_te_mlp, p_va=p_va_mlp)

    classes = np.array([0, 1, 2, 3], dtype=np.float32)
    out = {}

    print("\n=== Solo bagged MLP ===")
    m = metrics(yte, p_te_mlp.argmax(1))
    out["solo_mlp"] = m
    print(f"  κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}")

    print("\n=== Bagged MLP + threshold ===")
    e_va = (p_va_mlp * classes[None, :]).sum(1)
    e_te = (p_te_mlp * classes[None, :]).sum(1)
    bt = tune_thresholds(e_va, yva)
    m2 = metrics(yte, apply_thresholds(e_te, bt["t1"], bt["t2"], bt["t3"]))
    out["mlp_threshold"] = {**m2, **bt}
    print(f"  t=({bt['t1']:.2f},{bt['t2']:.2f},{bt['t3']:.2f})  κ_q={m2['kappa_q']:.3f} {m2['kappa_q_ci']}")

    print("\n=== Fusion: clip-LR-unif + clip-LR-RSB + MLP ===")
    c = np.load(CACHE)
    p_te_unif = c["p_te_unif"]; p_va_unif = c["p_va_unif"]
    p_te_rsb = c["p_te_rsb"]; p_va_rsb = c["p_va_rsb"]
    grid = np.linspace(0, 1, 21)
    best_w = None
    for w1 in grid:
        for w2 in grid:
            w3 = 1 - w1 - w2
            if w3 < -1e-9 or w3 > 1+1e-9: continue
            pv = w1*p_va_unif + w2*p_va_rsb + w3*p_va_mlp
            v = cohen_kappa_score(yva, pv.argmax(1), weights="quadratic")
            if best_w is None or v > best_w["v"]:
                best_w = {"w": (float(w1), float(w2), float(w3)), "v": float(v)}
    w1, w2, w3 = best_w["w"]
    p_va_3 = w1*p_va_unif + w2*p_va_rsb + w3*p_va_mlp
    p_te_3 = w1*p_te_unif + w2*p_te_rsb + w3*p_te_mlp
    m_arg = metrics(yte, p_te_3.argmax(1))
    out["fusion3_argmax_lr_rsb_mlp"] = {**m_arg, "w": best_w["w"], "val_kq": best_w["v"]}
    print(f"  argmax: w=({w1:.2f},{w2:.2f},{w3:.2f})  κ_q={m_arg['kappa_q']:.3f}  val={best_w['v']:.3f}")
    # +threshold
    e_va_3 = (p_va_3 * classes[None, :]).sum(1)
    e_te_3 = (p_te_3 * classes[None, :]).sum(1)
    bt_3 = tune_thresholds(e_va_3, yva)
    m_thr = metrics(yte, apply_thresholds(e_te_3, bt_3["t1"], bt_3["t2"], bt_3["t3"]))
    out["fusion3_threshold_lr_rsb_mlp"] = {**m_thr, **bt_3, "w": best_w["w"]}
    print(f"  +thresh: t=({bt_3['t1']:.2f},{bt_3['t2']:.2f},{bt_3['t3']:.2f})  κ_q={m_thr['kappa_q']:.3f} {m_thr['kappa_q_ci']}  val={bt_3['v']:.3f}")

    # 5-fold majority vote
    from sklearn.model_selection import KFold
    print("\n=== 5-fold majority vote on (unif, rsb, mlp) ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_preds = []
    for fold_idx, (cal_idx, _) in enumerate(kf.split(yva)):
        rng = np.random.default_rng(1000 + fold_idx)
        best = None
        for _ in range(15000):
            a = rng.dirichlet(np.ones(3))
            pv = a[0]*p_va_unif[cal_idx] + a[1]*p_va_rsb[cal_idx] + a[2]*p_va_mlp[cal_idx]
            e = (pv * classes[None, :]).sum(1)
            ts = sorted(rng.uniform(0, 3, size=3))
            yp = apply_thresholds(e, *ts)
            v = cohen_kappa_score(yva[cal_idx], yp, weights="quadratic")
            if best is None or v > best["v"]:
                best = {"w": a.tolist(), "t": ts, "v": float(v)}
        pe = best["w"][0]*p_te_unif + best["w"][1]*p_te_rsb + best["w"][2]*p_te_mlp
        e_te2 = (pe * classes[None, :]).sum(1)
        yp_te = apply_thresholds(e_te2, *best["t"])
        fold_preds.append(yp_te)
        print(f"  fold {fold_idx}: w={[f'{x:.2f}' for x in best['w']]} test κ={cohen_kappa_score(yte, yp_te, weights='quadratic'):.3f}")
    fold_preds = np.stack(fold_preds)
    from scipy.stats import mode
    yp_vote = mode(fold_preds, axis=0, keepdims=False).mode
    m_vote = metrics(yte, yp_vote)
    out["fold_vote_lr_rsb_mlp"] = m_vote
    print(f"  vote κ_q={m_vote['kappa_q']:.3f} {m_vote['kappa_q_ci']}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
