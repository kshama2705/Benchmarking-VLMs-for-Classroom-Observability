"""
EMBER-MLP — non-linear fusion of explicit face signals with frozen-VLM CLS.

Architecture:
  Branch A (explicit): MLP(73-d → 128 → 64)
  Branch B (CLS):      MLP(1024-d → 256 → 64)
  Fusion:              concat(64+64=128) → MLP(128 → 64) → linear(4)
  Optional:            grad-reversal subject head for subject-invariance

Training:
  - Subject-disjoint Train (5,358), Val (1,429), Test (1,784)
  - Class-balanced cross-entropy
  - 5 seeds per config
  - Early-stop on val κ_q

Output:
  results/ember/ember_mlp_results.json
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGNALS = os.path.join(BASE, "features", "daisee_face_signals.npz")
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
CLIPL = os.path.join(BASE, "features", "daisee_clip_l_14_features.npz")
DINO = os.path.join(BASE, "features", "dinov2_vitb14_features.npz")
OUT = os.path.join(BASE, "results", "ember", "ember_mlp_results.json")
LOG = os.path.join(BASE, "results", "ember", "ember_mlp_status.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
RNG = np.random.default_rng(42)
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))
SEEDS = [42, 7, 2025]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


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
class EMBER(nn.Module):
    def __init__(self, d_explicit, d_cls, n_subj=70, d_proj=64, dropout=0.3):
        super().__init__()
        self.exp_branch = nn.Sequential(
            nn.Linear(d_explicit, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, d_proj), nn.ReLU(),
        )
        self.cls_branch = nn.Sequential(
            nn.Linear(d_cls, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, d_proj), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_proj, 64), nn.ReLU(), nn.Dropout(dropout),
        )
        self.eng_head = nn.Linear(64, 4)
        self.subj_head = nn.Linear(64, n_subj)

    def forward(self, x_exp, x_cls, lambd=1.0):
        a = self.exp_branch(x_exp)
        b = self.cls_branch(x_cls)
        h = self.fusion(torch.cat([a, b], dim=-1))
        eng = self.eng_head(h)
        subj = self.subj_head(grl(h, lambd))
        return eng, subj, h


def class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1)
    w = counts.sum() / (n_classes * counts)
    return torch.from_numpy(w).float().to(DEVICE)


def boot_kq(yt, yp, n=1000):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def align_by_clip_id(target_clip_ids, source_clip_ids, source_feats):
    s_idx = {c: i for i, c in enumerate(source_clip_ids)}
    out = np.zeros((len(target_clip_ids), source_feats.shape[1]), dtype=np.float32)
    for j, c in enumerate(target_clip_ids):
        if c in s_idx:
            out[j] = source_feats[s_idx[c]]
    return out


def load_data():
    sig = np.load(SIGNALS, allow_pickle=True)
    clip_ids = sig["clip_id"]
    splits = sig["split"]
    eng = sig["engagement"].astype(np.int64)
    subj = sig["subject_id"]
    explicit = np.nan_to_num(
        np.concatenate([sig["blendshapes"], sig["head_pose"], sig["eye_gaze"], sig["landmark_summary"]], axis=1)
    )
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"

    # z-score explicit using train stats
    sc = StandardScaler().fit(explicit[tr])
    explicit = sc.transform(explicit).astype(np.float32)

    # Subject id integer mapping (train only)
    unique = sorted(set(subj[tr].tolist()))
    sid_map = {s: i for i, s in enumerate(unique)}
    subj_int_tr = np.array([sid_map[s] for s in subj[tr]], dtype=np.int64)

    feats = {}
    for label, path in [("SigLIP_L", SIGLIP), ("CLIP_L", CLIPL), ("DINOv2", DINO)]:
        if not os.path.exists(path):
            continue
        d = np.load(path, allow_pickle=True)
        key = "feat" if "feat" in d.files else "cls_feat"
        feats[label] = align_by_clip_id(clip_ids, d["clip_id"], d[key].astype(np.float32))
    return clip_ids, tr, va, te, eng, subj_int_tr, explicit, feats, len(unique)


def train_one(explicit, cls_feats, eng, subj_int_tr, tr, va, te, n_subj, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xe_t = torch.from_numpy(explicit[tr]).float().to(DEVICE)
    Xc_t = torch.from_numpy(cls_feats[tr]).float().to(DEVICE)
    y_t = torch.from_numpy(eng[tr]).long().to(DEVICE)
    s_t = torch.from_numpy(subj_int_tr).long().to(DEVICE)

    Xe_v = torch.from_numpy(explicit[va]).float().to(DEVICE)
    Xc_v = torch.from_numpy(cls_feats[va]).float().to(DEVICE)
    y_v = torch.from_numpy(eng[va]).long().to(DEVICE)

    Xe_te = torch.from_numpy(explicit[te]).float().to(DEVICE)
    Xc_te = torch.from_numpy(cls_feats[te]).float().to(DEVICE)
    y_te = torch.from_numpy(eng[te]).long().to(DEVICE)

    model = EMBER(explicit.shape[1], cls_feats.shape[1], n_subj=n_subj,
                  d_proj=cfg["d_proj"], dropout=cfg["dropout"]).to(DEVICE)
    eng_w = class_weights(eng[tr], 4)
    eng_loss_fn = nn.CrossEntropyLoss(weight=eng_w)
    subj_loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    n = len(y_t); bs = cfg["batch_size"]; n_epochs = cfg["epochs"]
    best = {"v": -1e9, "state": None}

    for ep in range(n_epochs):
        progress = ep / max(n_epochs - 1, 1)
        lambd = cfg["lambda_max"] * (2.0 / (1.0 + np.exp(-10 * progress)) - 1.0)

        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xe = Xe_t[idx]; xc = Xc_t[idx]; yb = y_t[idx]; sb = s_t[idx]
            eng_logits, subj_logits, _ = model(xe, xc, lambd=float(lambd))
            le = eng_loss_fn(eng_logits, yb)
            ls = subj_loss_fn(subj_logits, sb) if cfg["lambda_max"] > 0 else torch.tensor(0.0, device=DEVICE)
            loss = le + ls
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            ev, _, _ = model(Xe_v, Xc_v, lambd=0.0)
            yhat_v = ev.argmax(dim=1).cpu().numpy()
        v = float(cohen_kappa_score(y_v.cpu().numpy(), yhat_v, weights="quadratic"))
        if v > best["v"]:
            best = {"v": v, "state": {k: vv.detach().cpu().clone() for k, vv in model.state_dict().items()}}

    if best["state"] is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best["state"].items()})
    model.eval()
    with torch.no_grad():
        et, _, _ = model(Xe_te, Xc_te, lambd=0.0)
        yhat_te = et.argmax(dim=1).cpu().numpy()
    return yhat_te, best["v"]


def run(label, cls_feats, explicit, eng, subj_int_tr, tr, va, te, n_subj):
    cfgs = [
        {"name": f"{label}-no-adv",  "d_proj": 64, "dropout": 0.3, "lr": 1e-3, "batch_size": 128, "epochs": 40, "lambda_max": 0.0},
        {"name": f"{label}-adv-1",   "d_proj": 64, "dropout": 0.3, "lr": 1e-3, "batch_size": 128, "epochs": 40, "lambda_max": 1.0},
        {"name": f"{label}-adv-2",   "d_proj": 64, "dropout": 0.3, "lr": 1e-3, "batch_size": 128, "epochs": 40, "lambda_max": 2.0},
    ]
    results = []
    yte_np = eng[te]
    for cfg in cfgs:
        per_seed = []
        for seed in SEEDS:
            yhat, val_v = train_one(explicit, cls_feats, eng, subj_int_tr, tr, va, te, n_subj, cfg, seed)
            kq = float(cohen_kappa_score(yte_np, yhat, weights="quadratic"))
            acc = float(accuracy_score(yte_np, yhat))
            ci = boot_kq(yte_np, yhat)
            per_seed.append({"seed": seed, "kappa_q": kq, "kappa_q_ci": ci, "accuracy": acc, "val_kq": val_v})
            log(f"  {cfg['name']:25}  seed={seed:5d}  val={val_v:.3f}  test κ_q={kq:.3f} {ci}  acc={acc:.3f}")
        kappas = np.array([r["kappa_q"] for r in per_seed])
        summary = {
            "cfg": cfg,
            "mean_kq": float(kappas.mean()),
            "std_kq": float(kappas.std(ddof=1) if len(kappas) > 1 else 0.0),
            "min_kq": float(kappas.min()),
            "max_kq": float(kappas.max()),
            "per_seed": per_seed,
        }
        log(f"  {cfg['name']:25}  => mean={summary['mean_kq']:.3f} ± {summary['std_kq']:.3f}  "
            f"min={summary['min_kq']:.3f}  max={summary['max_kq']:.3f}")
        results.append(summary)
    return results


def main():
    open(LOG, "w").close()
    log("Loading features...")
    clip_ids, tr, va, te, eng, subj_int_tr, explicit, feats, n_subj = load_data()
    log(f"  Train={tr.sum()} Val={va.sum()} Test={te.sum()} n_subjects={n_subj}  explicit={explicit.shape[1]}-d")
    out = {}
    for label, X in feats.items():
        log(f"\n=== {label} ({X.shape[1]}-d) ===")
        out[label] = run(label, X, explicit, eng, subj_int_tr, tr, va, te, n_subj)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {OUT}")
    log("EMBER_MLP_DONE")


if __name__ == "__main__":
    main()
