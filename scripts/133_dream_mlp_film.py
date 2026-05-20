"""
DREAM-FiLM — small MLP head with anchor as conditioning signal.

Architecture:
    z = MLP_main(f(x))                  # 1024 -> 256
    gamma, beta = MLP_film(a_s)          # 1024 -> 256, 256
    h = (1 + gamma) * z + beta           # FiLM modulation
    logits = Linear(h)                   # 256 -> 4

Training: cross-entropy on engagement labels, class-balanced.
3 seeds × 60 epochs each, EMA on weights.

Comparison: against baseline LR on raw features and against DREAM-sub / DREAM-cat.
This tests whether non-linear anchor conditioning recovers signal that pure
subtraction destroys.
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, cohen_kappa_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
ANCHORS = os.path.join(BASE, "features", "dream_anchors.npz")
OUT = os.path.join(BASE, "results", "dream", "dream_mlp_film.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)
RNG = np.random.default_rng(42)


class DreamFiLM(nn.Module):
    def __init__(self, d_in=1024, d_h=256, n_cls=4):
        super().__init__()
        self.main = nn.Sequential(nn.Linear(d_in, d_h), nn.GELU(), nn.Dropout(0.2))
        self.film_g = nn.Linear(d_in, d_h)
        self.film_b = nn.Linear(d_in, d_h)
        self.head = nn.Sequential(nn.Linear(d_h, d_h), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(d_h, n_cls))
        # init film to identity
        nn.init.zeros_(self.film_g.weight); nn.init.zeros_(self.film_g.bias)
        nn.init.zeros_(self.film_b.weight); nn.init.zeros_(self.film_b.bias)

    def forward(self, x, a):
        z = self.main(x)
        g = self.film_g(a); b = self.film_b(a)
        h = (1 + g) * z + b
        return self.head(h)


def boot_ci(yt, yp, n=500):
    nt = len(yt); out = []
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


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


def train_one(Xtr, atr, ytr, Xva, ava, yva, Xte, ate, yte,
              seed=42, epochs=40, batch=128, lr=1e-3, weight_decay=1e-4):
    torch.manual_seed(seed); np.random.seed(seed)
    device = DEVICE

    # class-balanced weights
    cw = np.zeros(4)
    for k in range(4):
        cw[k] = 1.0 / max(1, (ytr == k).sum())
    cw = cw / cw.sum() * 4
    cw_t = torch.tensor(cw, dtype=torch.float32, device=device)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    atr_t = torch.tensor(atr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    ava_t = torch.tensor(ava, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    ate_t = torch.tensor(ate, dtype=torch.float32, device=device)

    ds = TensorDataset(Xtr_t, atr_t, ytr_t)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=0)

    model = DreamFiLM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss(weight=cw_t)

    best_val = -1; best_state = None
    for ep in range(epochs):
        model.train()
        for xb, ab, yb in dl:
            xb = xb.to(device); ab = ab.to(device); yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb, ab)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.softmax(model(Xva_t, ava_t), dim=1).cpu().numpy()
        e_va = (pv * np.arange(4, dtype=np.float32)[None]).sum(1)
        kq = cohen_kappa_score(yva, pv.argmax(1), weights="quadratic")
        if kq > best_val:
            best_val = kq
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pte = torch.softmax(model(Xte_t, ate_t), dim=1).cpu().numpy()
        pva = torch.softmax(model(Xva_t, ava_t), dim=1).cpu().numpy()
    return pte, pva, float(best_val)


def main():
    d = np.load(SIGLIP, allow_pickle=True)
    feat = d['feat'].astype(np.float32)
    split = d['split']; subject = d['subject_id']; eng = d['engagement'].astype(np.int64)
    a_npz = np.load(ANCHORS, allow_pickle=True)
    subjects = a_npz['subject_ids']

    tr_mask = (split == 'Train'); va_mask = (split == 'Validation'); te_mask = (split == 'Test')

    results = {}
    classes = np.arange(4, dtype=np.float32)

    for K in [1, 5, 10]:
        idx_key = f"p_train_anchor_idx_k{K}"
        anchor_idx = a_npz[idx_key]
        anchor_means = {}
        for si, s in enumerate(subjects):
            idx = anchor_idx[si]; idx = idx[idx >= 0]
            anchor_means[s] = feat[idx].mean(0) if len(idx) > 0 else np.zeros(feat.shape[1], np.float32)
        a_full = np.stack([anchor_means[s] for s in subject], axis=0)

        Xtr = feat[tr_mask]; atr = a_full[tr_mask]; ytr = eng[tr_mask]
        Xva = feat[va_mask]; ava = a_full[va_mask]; yva = eng[va_mask]
        Xte = feat[te_mask]; ate = a_full[te_mask]; yte = eng[te_mask]

        seed_res = []
        all_pte = []; all_pva = []
        for seed in [0, 42, 2025]:
            t0 = time.time()
            pte, pva, bv = train_one(Xtr, atr, ytr, Xva, ava, yva, Xte, ate, yte,
                                     seed=seed, epochs=40)
            yp = pte.argmax(1)
            e_te = (pte * classes[None]).sum(1)
            e_va = (pva * classes[None]).sum(1)
            kq_arg = float(cohen_kappa_score(yte, yp, weights="quadratic"))
            bt = tune_thresholds(e_va, yva)
            yp_thr = apply_thr(e_te, bt['t'])
            kq_thr = float(cohen_kappa_score(yte, yp_thr, weights="quadratic"))
            dt = time.time() - t0
            print(f"[K={K} seed={seed}] val_best={bv:.4f}  arg κ={kq_arg:.4f}  thr κ={kq_thr:.4f} (t={bt['t']})  [{dt:.0f}s]", flush=True)
            seed_res.append({"seed": seed, "val_best": bv,
                             "kq_arg": kq_arg, "kq_thr": kq_thr, "thresholds": bt['t']})
            all_pte.append(pte); all_pva.append(pva)
        # Ensemble across seeds
        pte_ens = np.mean(all_pte, axis=0); pva_ens = np.mean(all_pva, axis=0)
        e_te = (pte_ens * classes[None]).sum(1); e_va = (pva_ens * classes[None]).sum(1)
        bt = tune_thresholds(e_va, yva)
        yp_thr = apply_thr(e_te, bt['t'])
        kq_thr = float(cohen_kappa_score(yte, yp_thr, weights="quadratic"))
        kq_arg = float(cohen_kappa_score(yte, pte_ens.argmax(1), weights="quadratic"))
        ci_thr = boot_ci(yte, yp_thr, n=500)
        print(f"[K={K} ENSEMBLE] arg κ={kq_arg:.4f}  thr κ={kq_thr:.4f} {ci_thr}", flush=True)
        results[f"K{K}"] = {"seeds": seed_res,
                            "ensemble": {"kq_arg": kq_arg, "kq_thr": kq_thr, "ci_thr": ci_thr,
                                         "thresholds": bt['t']}}

    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
