"""
N7: SIEP v2 — stabilization + contrastive variants.

Adds:
  - Stabilization: cosine LR with warmup, EMA on weights (SWA-style),
    longer training (60 epochs), larger effective batch (128).
  - Contrastive subject-stratified loss: positives = same engagement DIFFERENT
    subject; negatives = same subject DIFFERENT engagement. Forces engagement
    embedding to be subject-invariant by construction (InfoNCE).
  - SupCon variant: standard supervised contrastive on engagement label only
    (no subject term).

Modes: 'adversarial' (DANN, like v1), 'subject_contrastive', 'supcon'.

Each run: 3 seeds. Reports mean ± std κ_q on test.

Output:
  results/siep/siep_v2_results.json
"""

import os, json, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_squared_error

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "results", "siep")
os.makedirs(OUT_DIR, exist_ok=True)
RNG = np.random.default_rng(42)
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))

FEATS = {
    "DINOv2_ViT-B14": os.path.join(BASE, "features", "dinov2_vitb14_features.npz"),
    "CLIP_ViT-B32": os.path.join(BASE, "features", "clip_vitb32_features.npz"),
}

SEEDS = [42, 7, 2025]


# ============ Gradient Reversal ============
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return g.neg() * ctx.lambd, None


def grl(x, lambd):
    return GradReverse.apply(x, lambd)


# ============ Model ============
class SIEPv2(nn.Module):
    def __init__(self, d_in, d_hidden, n_eng, n_subj, d_proj=128, n_layers=2, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_d = d_in if i == 0 else d_hidden
            layers += [nn.Linear(in_d, d_hidden), nn.ReLU(), nn.Dropout(dropout)]
        self.encoder = nn.Sequential(*layers)
        self.eng_head = nn.Linear(d_hidden, n_eng)
        self.subj_head = nn.Linear(d_hidden, n_subj)
        # Projection for contrastive losses (separate head, L2-normalized)
        self.proj = nn.Sequential(
            nn.Linear(d_hidden, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_proj),
        )

    def forward(self, x, lambd=1.0, want_proj=False):
        z = self.encoder(x)
        eng_logits = self.eng_head(z)
        subj_logits = self.subj_head(grl(z, lambd))
        if want_proj:
            p = F.normalize(self.proj(z), dim=-1)
            return eng_logits, subj_logits, z, p
        return eng_logits, subj_logits, z


# ============ EMA ============
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self):
        return self.shadow


# ============ Contrastive losses ============
def supcon_loss(proj, labels, temperature=0.1):
    """Standard supervised contrastive (Khosla 2020) on a label."""
    device = proj.device
    n = proj.shape[0]
    sim = proj @ proj.T / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values  # for stability
    mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)
    mask_pos.fill_diagonal_(0)
    mask_self = torch.ones((n, n), device=device) - torch.eye(n, device=device)
    exp_sim = sim.exp() * mask_self
    log_prob = sim - exp_sim.sum(dim=1, keepdim=True).log()
    n_pos = mask_pos.sum(dim=1).clamp(min=1)
    loss = -(mask_pos * log_prob).sum(dim=1) / n_pos
    return loss.mean()


def subject_stratified_loss(proj, eng, subj, temperature=0.1):
    """
    Subject-stratified InfoNCE:
      positives = same engagement, DIFFERENT subject
      negatives = same subject, DIFFERENT engagement (and "neutral" = different subj different eng)

    Loss form: pull together (same eng, diff subj); push apart (same subj, diff eng).
    Implemented as a softmax over positive set vs (positive + hard negative + others).
    """
    device = proj.device
    n = proj.shape[0]
    sim = proj @ proj.T / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values

    # Masks
    same_eng = (eng.unsqueeze(0) == eng.unsqueeze(1)).to(device)
    same_subj = (subj.unsqueeze(0) == subj.unsqueeze(1)).to(device)
    self_mask = torch.eye(n, dtype=torch.bool, device=device)

    # Positive: same engagement, DIFFERENT subject (and not self)
    pos_mask = same_eng & ~same_subj & ~self_mask
    # Hard-negative: same subject, DIFFERENT engagement
    hardneg_mask = same_subj & ~same_eng
    # All non-self entries
    not_self = ~self_mask

    # Build denominator over (positives + hard negatives) for each anchor.
    # Skip anchors with no positives.
    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Use all non-self as "denominator" (standard InfoNCE) but only count positives in numerator
    exp_sim = sim.exp() * not_self.float()
    log_prob = sim - exp_sim.sum(dim=1, keepdim=True).log()

    # average over positives per anchor (with positives)
    n_pos = pos_mask.float().sum(dim=1).clamp(min=1)
    loss_per = -(pos_mask.float() * log_prob).sum(dim=1) / n_pos
    return loss_per[has_pos].mean()


# ============ Helpers ============
def class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1)
    w = counts.sum() / (n_classes * counts)
    return torch.from_numpy(w).float().to(DEVICE)


def evaluate(state_dict, model_template, X, y, batch=256):
    model_template.load_state_dict(state_dict)
    model_template.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).float().to(DEVICE)
            eng_logits, _, _ = model_template(xb, lambd=0.0)
            preds.append(eng_logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def boot_kappa(yt, yp, n=1000):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def cosine_warmup_lr(step, total_steps, base_lr, warmup_frac=0.1):
    warmup_steps = int(total_steps * warmup_frac)
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * base_lr * (1 + np.cos(np.pi * p))


def load_split(path):
    data = np.load(path, allow_pickle=True)
    splits = data["split"]; feats = data["feat"].astype(np.float32)
    eng = data["engagement"]; subj = data["subject_id"]
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    unique = sorted(set(subj[tr].tolist()))
    sid_map = {s: i for i, s in enumerate(unique)}
    str_arr = np.array([sid_map[s] for s in subj[tr]], dtype=np.int64)
    return (feats[tr], eng[tr].astype(np.int64), str_arr,
            feats[va], eng[va].astype(np.int64),
            feats[te], eng[te].astype(np.int64), len(unique))


# ============ Training ============
def train_v2(Xtr, ytr, str_arr, Xva, yva, n_subj, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    d_in = Xtr.shape[1]
    model = SIEPv2(d_in, cfg["d_hidden"], 4, n_subj, d_proj=cfg.get("d_proj", 128),
                   n_layers=cfg["n_layers"], dropout=cfg["dropout"]).to(DEVICE)
    eng_w = class_weights(ytr, 4)
    eng_loss_fn = nn.CrossEntropyLoss(weight=eng_w)
    subj_loss_fn = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    ema = EMA(model, decay=cfg["ema_decay"]) if cfg.get("use_ema", True) else None

    Xtr_t = torch.from_numpy(Xtr).float().to(DEVICE)
    ytr_t = torch.from_numpy(ytr).long().to(DEVICE)
    str_t = torch.from_numpy(str_arr).long().to(DEVICE)
    n = len(Xtr); bs = cfg["batch_size"]; n_epochs = cfg["epochs"]
    steps_per_epoch = (n + bs - 1) // bs
    total_steps = steps_per_epoch * n_epochs

    best = {"v": -1e9, "state": None}
    step = 0
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr_t[idx]; yb = ytr_t[idx]; sb = str_t[idx]
            # LR schedule
            for g in opt.param_groups:
                g["lr"] = cosine_warmup_lr(step, total_steps, cfg["lr"])
            # GRL ramp
            progress = ep / max(n_epochs - 1, 1)
            lambd = cfg["lambda_max"] * (2.0 / (1.0 + np.exp(-10 * progress)) - 1.0) \
                    if cfg["mode"] == "adversarial" else 0.0

            if cfg["mode"] in ("supcon", "subject_contrastive"):
                eng_logits, subj_logits, _, proj = model(xb, lambd=0.0, want_proj=True)
                le = eng_loss_fn(eng_logits, yb)
                if cfg["mode"] == "supcon":
                    lc = supcon_loss(proj, yb, temperature=cfg.get("temperature", 0.1))
                else:
                    lc = subject_stratified_loss(proj, yb, sb, temperature=cfg.get("temperature", 0.1))
                loss = le + cfg.get("contrastive_weight", 1.0) * lc
            else:  # adversarial
                eng_logits, subj_logits, _ = model(xb, lambd=float(lambd))
                le = eng_loss_fn(eng_logits, yb)
                ls = subj_loss_fn(subj_logits, sb)
                loss = le + ls

            opt.zero_grad(); loss.backward(); opt.step()
            if ema is not None:
                ema.update(model)
            step += 1

        # Validate using EMA weights if available
        eval_state = ema.state_dict() if ema is not None else model.state_dict()
        yhat_va = evaluate(eval_state, model, Xva, yva)
        v = float(cohen_kappa_score(yva, yhat_va, weights="quadratic"))
        if v > best["v"]:
            best["v"] = v
            best["state"] = {k: vv.detach().cpu().clone() for k, vv in eval_state.items()}
    if best["state"] is not None:
        # Move back to device
        best_state = {k: v.to(DEVICE) for k, v in best["state"].items()}
    else:
        best_state = model.state_dict()
    return model, best_state, best["v"]


def run_config(label, feats_path, cfg):
    Xtr, ytr, str_, Xva, yva, Xte, yte, n_subj = load_split(feats_path)
    print(f"\n=== {label} | mode={cfg['mode']} | "
          f"d_hidden={cfg['d_hidden']} λ={cfg.get('lambda_max',0)} cw={cfg.get('contrastive_weight',0)} ===",
          flush=True)
    rows = []
    for seed in SEEDS:
        model, best_state, val_kq = train_v2(Xtr, ytr, str_, Xva, yva, n_subj, cfg, seed)
        yhat = evaluate(best_state, model, Xte, yte)
        kq = float(cohen_kappa_score(yte, yhat, weights="quadratic"))
        acc = float(accuracy_score(yte, yhat))
        ci = boot_kappa(yte, yhat)
        rows.append({"seed": seed, "kappa_q": kq, "kappa_q_ci95": ci, "accuracy": acc, "val_kq": val_kq})
        print(f"  seed={seed:5d}  val_κq={val_kq:.3f}  test κ_q={kq:.3f} [{ci[0]:.3f},{ci[1]:.3f}]  acc={acc:.3f}",
              flush=True)
    kappas = np.array([r["kappa_q"] for r in rows])
    summary = {
        "encoder": label, "config": cfg,
        "mean_kq": float(kappas.mean()),
        "std_kq": float(kappas.std(ddof=1) if len(kappas) > 1 else 0.0),
        "min_kq": float(kappas.min()), "max_kq": float(kappas.max()),
        "per_seed": rows,
    }
    print(f"  => mean={summary['mean_kq']:.3f} ± {summary['std_kq']:.3f}  "
          f"(min={summary['min_kq']:.3f}, max={summary['max_kq']:.3f})", flush=True)
    return summary


def main():
    base = dict(d_hidden=256, n_layers=2, dropout=0.2, lr=5e-4, batch_size=128,
                epochs=60, ema_decay=0.999, use_ema=True, d_proj=128)
    runs = [
        # === DINOv2 ===
        # Stabilized adversarial (C)
        ("DINOv2_ViT-B14", FEATS["DINOv2_ViT-B14"],
         {**base, "mode": "adversarial", "lambda_max": 1.0}),
        ("DINOv2_ViT-B14", FEATS["DINOv2_ViT-B14"],
         {**base, "mode": "adversarial", "lambda_max": 2.0}),
        # Stabilized MLP-only (no subject loss; serves as the "stabilized linear" comp)
        ("DINOv2_ViT-B14", FEATS["DINOv2_ViT-B14"],
         {**base, "mode": "adversarial", "lambda_max": 0.0}),
        # SupCon (engagement-only contrastive)
        ("DINOv2_ViT-B14", FEATS["DINOv2_ViT-B14"],
         {**base, "mode": "supcon", "lambda_max": 0.0, "contrastive_weight": 1.0, "temperature": 0.1}),
        # Subject-stratified contrastive (B headline)
        ("DINOv2_ViT-B14", FEATS["DINOv2_ViT-B14"],
         {**base, "mode": "subject_contrastive", "lambda_max": 0.0, "contrastive_weight": 1.0, "temperature": 0.1}),
        ("DINOv2_ViT-B14", FEATS["DINOv2_ViT-B14"],
         {**base, "mode": "subject_contrastive", "lambda_max": 0.0, "contrastive_weight": 2.0, "temperature": 0.1}),
        # === CLIP ===
        ("CLIP_ViT-B32", FEATS["CLIP_ViT-B32"],
         {**base, "mode": "adversarial", "lambda_max": 0.0}),
        ("CLIP_ViT-B32", FEATS["CLIP_ViT-B32"],
         {**base, "mode": "subject_contrastive", "lambda_max": 0.0, "contrastive_weight": 1.0, "temperature": 0.1}),
    ]
    summaries = []
    for label, path, cfg in runs:
        summaries.append(run_config(label, path, cfg))
    out_path = os.path.join(OUT_DIR, "siep_v2_results.json")
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
