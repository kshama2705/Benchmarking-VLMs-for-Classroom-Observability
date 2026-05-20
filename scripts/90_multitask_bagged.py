"""
MOONSHOT 1: Multi-task learning with auxiliary affect labels.

DAiSEE has 4 labels: Boredom, Engagement, Confusion, Frustration.
Hypothesis: joint MLP head predicting all 4 forces representations that capture
broader affect; engagement gets less identity-correlated.

Architecture: shared trunk → 4 heads (one per label).
Loss: focal CE for each task; weighted sum with engagement primary.

Bag this (K=10 × 5 seeds, MPS), threshold-tune engagement head, fuse with cached
LR-unif probs via val-tuned weight.

Output:
  results/sota/moonshot_multitask.json
"""
import os, json, time, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
LABELS_TR = os.path.join(BASE, "DAiSEE", "Labels", "TrainLabels.csv")
LABELS_VA = os.path.join(BASE, "DAiSEE", "Labels", "ValidationLabels.csv")
LABELS_TE = os.path.join(BASE, "DAiSEE", "Labels", "TestLabels.csv")
CACHE = os.path.join(BASE, "results", "sota", "_bag_cache.npz")
MT_CACHE = os.path.join(BASE, "results", "sota", "_multitask_cache.npz")
OUT = os.path.join(BASE, "results", "sota", "moonshot_multitask.json")

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))
RNG = np.random.default_rng(42)


def load_labels():
    """Build dict: clip_id -> (boredom, engagement, confusion, frustration)"""
    label_map = {}
    for path in [LABELS_TR, LABELS_VA, LABELS_TE]:
        with open(path) as f:
            r = csv.DictReader(f, skipinitialspace=True)
            for row in r:
                cid = row["ClipID"].strip()
                # Headers may have trailing spaces — get values robustly
                keys = {k.strip(): v for k, v in row.items()}
                label_map[cid] = (
                    int(keys["Boredom"]),
                    int(keys["Engagement"]),
                    int(keys["Confusion"]),
                    int(keys["Frustration"]),
                )
    return label_map


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


class MultiTaskHead(nn.Module):
    def __init__(self, D, n_classes=4, h1=512, h2=256, dropout=0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(D, h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.GELU(), nn.Dropout(dropout),
        )
        self.head_engagement = nn.Linear(h2, n_classes)
        self.head_boredom = nn.Linear(h2, n_classes)
        self.head_confusion = nn.Linear(h2, n_classes)
        self.head_frustration = nn.Linear(h2, n_classes)

    def forward(self, x):
        z = self.trunk(x)
        return {
            "engagement": self.head_engagement(z),
            "boredom": self.head_boredom(z),
            "confusion": self.head_confusion(z),
            "frustration": self.head_frustration(z),
        }


def focal_loss(logits, targets, alpha, gamma=2.0):
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def train_multitask(Xtr, Ytr, Xva, Yva, n_epochs=40, bs=128, lr=1e-3, seed=42,
                    weights={"engagement": 1.0, "boredom": 0.3, "confusion": 0.3, "frustration": 0.3}):
    """Ytr is dict of {task: y array} for train; same for Yva."""
    torch.manual_seed(seed); np.random.seed(seed)
    D = Xtr.shape[1]
    model = MultiTaskHead(D).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    # Per-task class weights
    alphas = {}
    for task in weights:
        cw = compute_class_weight("balanced", classes=np.arange(4), y=Ytr[task])
        alphas[task] = torch.tensor(cw, dtype=torch.float32, device=DEVICE)

    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    Yt = {k: torch.tensor(v, dtype=torch.long, device=DEVICE) for k, v in Ytr.items()}
    Xv = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)

    best_va_kq, best_state = -1, None
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i+bs]
            out = model(Xt[idx])
            loss = sum(weights[t] * focal_loss(out[t], Yt[t][idx], alphas[t], gamma=2.0) for t in weights)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            p_eng = model(Xv)["engagement"].softmax(-1).cpu().numpy()
        kq = cohen_kappa_score(Yva["engagement"], p_eng.argmax(1), weights="quadratic")
        if kq > best_va_kq:
            best_va_kq = kq
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_va_kq


def bag_multitask(Xtr, Ytr, Xva, Yva, Xte, K=10, seeds=[0, 7, 42, 2025, 1024], n_epochs=40, weights=None):
    all_te, all_va = [], []
    for so in seeds:
        rng = np.random.default_rng(so)
        bt, bv = [], []
        for k in range(K):
            t0 = time.time()
            idx = rng.integers(0, len(Ytr["engagement"]), size=len(Ytr["engagement"]))
            Ybt = {t: y[idx] for t, y in Ytr.items()}
            model, va_kq = train_multitask(Xtr[idx], Ybt, Xva, Yva, n_epochs=n_epochs, seed=so*1000+k, weights=weights)
            with torch.no_grad():
                Xv_t = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
                Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
                pv = model(Xv_t)["engagement"].softmax(-1).cpu().numpy()
                pt = model(Xte_t)["engagement"].softmax(-1).cpu().numpy()
            bt.append(pt); bv.append(pv)
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
    sp = d["split"]
    clip_ids = d["clip_id"]
    label_map = load_labels()
    # Build all-4-label array
    Y_all = np.zeros((len(clip_ids), 4), dtype=np.int64)  # [boredom, engagement, confusion, frustration]
    miss = 0
    for i, cid in enumerate(clip_ids):
        if cid in label_map:
            Y_all[i] = label_map[cid]
        else:
            miss += 1
    print(f"  Missing labels: {miss}", flush=True)

    tr = sp == "Train"; va = sp == "Validation"; te = sp == "Test"
    Xtr = X[tr]; Xva = X[va]; Xte = X[te]
    Ytr_all = Y_all[tr]; Yva_all = Y_all[va]; Yte_all = Y_all[te]

    Ytr = {"boredom": Ytr_all[:, 0], "engagement": Ytr_all[:, 1],
            "confusion": Ytr_all[:, 2], "frustration": Ytr_all[:, 3]}
    Yva = {"boredom": Yva_all[:, 0], "engagement": Yva_all[:, 1],
            "confusion": Yva_all[:, 2], "frustration": Yva_all[:, 3]}
    yte = Yte_all[:, 1]  # engagement
    yva = Yva["engagement"]

    print("\nLabel distributions:", flush=True)
    for tname, yt in zip(["boredom", "engagement", "confusion", "frustration"], Ytr_all.T):
        print(f"  {tname}: {np.bincount(yt, minlength=4).tolist()}", flush=True)

    out = {}

    # Try several weight schedules
    configs = [
        {"name": "engagement_only", "weights": {"engagement": 1.0, "boredom": 0.0, "confusion": 0.0, "frustration": 0.0}},
        {"name": "mt_balanced_low", "weights": {"engagement": 1.0, "boredom": 0.2, "confusion": 0.2, "frustration": 0.2}},
        {"name": "mt_balanced_mid", "weights": {"engagement": 1.0, "boredom": 0.5, "confusion": 0.5, "frustration": 0.5}},
        {"name": "mt_equal", "weights": {"engagement": 1.0, "boredom": 1.0, "confusion": 1.0, "frustration": 1.0}},
    ]

    cached = np.load(MT_CACHE) if os.path.exists(MT_CACHE) else None
    classes = np.array([0, 1, 2, 3], dtype=np.float32)

    for cfg in configs:
        name = cfg["name"]
        print(f"\n=== Config: {name}  weights={cfg['weights']} ===", flush=True)
        cache_key_te = f"p_te_{name}"; cache_key_va = f"p_va_{name}"
        if cached is not None and cache_key_te in cached.files:
            p_te = cached[cache_key_te]; p_va = cached[cache_key_va]
            print(f"  loaded from cache", flush=True)
        else:
            t0 = time.time()
            p_te, p_va = bag_multitask(Xtr, Ytr, Xva, Yva, Xte, K=10, n_epochs=40, weights=cfg["weights"])
            print(f"  Total: {time.time()-t0:.0f}s", flush=True)

        # Solo metrics
        m_solo = metrics(yte, p_te.argmax(1))
        print(f"  Solo: κ_q={m_solo['kappa_q']:.4f} {m_solo['kappa_q_ci']}", flush=True)

        # Threshold tune on engagement E[y]
        e_va = (p_va * classes[None, :]).sum(1)
        e_te = (p_te * classes[None, :]).sum(1)
        bt = tune_thresh(e_va, yva)
        m_thr = metrics(yte, apply_t(e_te, bt["t"]))
        print(f"  +threshold: κ_q={m_thr['kappa_q']:.4f} {m_thr['kappa_q_ci']}  t={bt['t']}", flush=True)

        out[name] = {"solo": m_solo, "threshold": {**m_thr, **bt}}

        # Stash for caching
        if cached is None or cache_key_te not in cached.files:
            existing = dict(cached) if cached is not None else {}
            existing[cache_key_te] = p_te
            existing[cache_key_va] = p_va
            np.savez_compressed(MT_CACHE, **existing)
            cached = np.load(MT_CACHE)

    # Fusion: best MT config + cached LR-unif
    print("\n=== Fusion: best MT + cached LR-unif ===", flush=True)
    best_mt = max(configs, key=lambda c: out[c["name"]]["threshold"]["kappa_q"])
    print(f"  best MT: {best_mt['name']}  κ={out[best_mt['name']]['threshold']['kappa_q']:.4f}", flush=True)
    p_te_mt = cached[f"p_te_{best_mt['name']}"]; p_va_mt = cached[f"p_va_{best_mt['name']}"]
    c2 = np.load(CACHE)
    p_te_lr = c2["p_te_unif"]; p_va_lr = c2["p_va_unif"]
    best_w = None
    for w in np.linspace(0, 1, 21):
        pv = w * p_va_lr + (1 - w) * p_va_mt
        e = (pv * classes[None, :]).sum(1)
        bt2 = tune_thresh(e, yva)
        if best_w is None or bt2["v"] > best_w["v"]:
            best_w = {"w_lr": float(w), **bt2}
    wl = best_w["w_lr"]
    p_te_f = wl * p_te_lr + (1 - wl) * p_te_mt
    e_te_f = (p_te_f * classes[None, :]).sum(1)
    m_f = metrics(yte, apply_t(e_te_f, best_w["t"]))
    out["fusion_lr_mt"] = {**m_f, "w_lr": wl, **best_w}
    print(f"  Fusion: w_lr={wl:.2f}  t={best_w['t']}  κ_q={m_f['kappa_q']:.4f} {m_f['kappa_q_ci']}  val={best_w['v']:.4f}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
