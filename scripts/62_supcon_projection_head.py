"""
Supervised-Contrastive Projection Head for engagement.

Input: concat(SigLIP-L CLS, explicit signals) - 1024+73 = 1097-d
Encoder: 2-layer MLP → 128-d projection
Loss: CE on engagement + InfoNCE/SupCon (positives = same engagement, regardless
      of subject). Subject-stratified positives optional.

Multi-seed validation.

Output:
  results/sota/supcon_head_results.json
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGLIP = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
SIGNALS = os.path.join(BASE, "features", "daisee_face_signals.npz")
TEMPORAL = os.path.join(BASE, "features", "daisee_face_signals_temporal.npz")
POSE = os.path.join(BASE, "features", "daisee_pose_signals.npz")
OUT = os.path.join(BASE, "results", "sota", "supcon_head_results.json")
LOG = os.path.join(BASE, "results", "sota", "supcon_status.txt")
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


class Head(nn.Module):
    def __init__(self, d_in, d_hidden=256, d_proj=128, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(d_hidden, 4)
        self.proj = nn.Sequential(
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_proj),
        )

    def forward(self, x):
        h = self.encoder(x)
        return self.classifier(h), F.normalize(self.proj(h), dim=-1)


def supcon_loss(proj, labels, subj, temperature=0.1, subject_stratified=False):
    """SupCon: positives = same engagement label. If subject_stratified, also
    require different subject for positives, same subject for negatives."""
    n = proj.shape[0]
    sim = proj @ proj.T / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values  # stability
    same_eng = (labels.unsqueeze(0) == labels.unsqueeze(1))
    self_mask = torch.eye(n, dtype=torch.bool, device=proj.device)
    not_self = ~self_mask
    if subject_stratified:
        same_subj = (subj.unsqueeze(0) == subj.unsqueeze(1))
        pos_mask = same_eng & ~same_subj & ~self_mask
    else:
        pos_mask = same_eng & ~self_mask
    if not pos_mask.any():
        return torch.tensor(0.0, device=proj.device, requires_grad=True)
    exp_sim = sim.exp() * not_self.float()
    log_prob = sim - exp_sim.sum(dim=1, keepdim=True).log()
    n_pos = pos_mask.float().sum(dim=1).clamp(min=1)
    has_pos = pos_mask.any(dim=1)
    loss_per = -(pos_mask.float() * log_prob).sum(dim=1) / n_pos
    return loss_per[has_pos].mean()


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


def class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1)
    w = counts.sum() / (n_classes * counts)
    return torch.from_numpy(w).float().to(DEVICE)


def align(target_ids, source_ids, source_feats):
    s_idx = {c: i for i, c in enumerate(source_ids)}
    out = np.zeros((len(target_ids), source_feats.shape[1]), dtype=np.float32)
    for j, c in enumerate(target_ids):
        if c in s_idx:
            out[j] = source_feats[s_idx[c]]
    return out


def train_one(Xtr, ytr, str_, Xva, yva, Xte, yte, n_subj, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Head(Xtr.shape[1], d_hidden=cfg["d_hidden"], d_proj=cfg["d_proj"],
                 dropout=cfg["dropout"]).to(DEVICE)
    eng_w = class_weights(ytr, 4)
    eng_loss_fn = nn.CrossEntropyLoss(weight=eng_w)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    Xtr_t = torch.from_numpy(Xtr).float().to(DEVICE)
    ytr_t = torch.from_numpy(ytr).long().to(DEVICE)
    str_t = torch.from_numpy(str_).long().to(DEVICE)
    Xva_t = torch.from_numpy(Xva).float().to(DEVICE)
    yva_t = torch.from_numpy(yva).long().to(DEVICE)
    Xte_t = torch.from_numpy(Xte).float().to(DEVICE)
    yte_t = torch.from_numpy(yte).long().to(DEVICE)

    n = len(ytr); bs = cfg["batch_size"]
    best = {"val_kq": -1e9, "state": None}
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr_t[idx]; yb = ytr_t[idx]; sb = str_t[idx]
            logits, proj = model(xb)
            le = eng_loss_fn(logits, yb)
            lc = supcon_loss(proj, yb, sb, temperature=cfg["temperature"],
                              subject_stratified=cfg["subject_stratified"])
            loss = le + cfg["contrastive_weight"] * lc
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            logits_va, _ = model(Xva_t)
            yhat_va = logits_va.argmax(dim=1).cpu().numpy()
        v = float(cohen_kappa_score(yva, yhat_va, weights="quadratic"))
        if v > best["val_kq"]:
            best["val_kq"] = v
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Load best and evaluate on test
    if best["state"] is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best["state"].items()})
    model.eval()
    with torch.no_grad():
        logits_te, _ = model(Xte_t)
        yhat_te = logits_te.argmax(dim=1).cpu().numpy()
    m = metrics(yte, yhat_te)
    return m, best["val_kq"]


def main():
    open(LOG, "w").close()
    log("Loading features...")
    d_sig = np.load(SIGLIP, allow_pickle=True)
    clip_ids = d_sig["clip_id"]; splits = d_sig["split"]
    eng = d_sig["engagement"].astype(np.int64)
    subj = d_sig["subject_id"]
    sig_feats = d_sig["feat"].astype(np.float32)

    d_sf = np.load(SIGNALS, allow_pickle=True)
    explicit = align(clip_ids, d_sf["clip_id"], np.nan_to_num(np.concatenate(
        [d_sf["blendshapes"], d_sf["head_pose"], d_sf["eye_gaze"], d_sf["landmark_summary"]], axis=1)))

    d_tmp = np.load(TEMPORAL, allow_pickle=True)
    temporal = align(clip_ids, d_tmp["clip_id"], np.nan_to_num(np.concatenate(
        [d_tmp["blendshapes_mean"], d_tmp["blendshapes_std"], d_tmp["blendshapes_delta"],
         d_tmp["head_pose_mean"], d_tmp["head_pose_std"],
         d_tmp["eye_gaze_mean"], d_tmp["eye_gaze_std"],
         d_tmp["landmark_summary_mean"]], axis=1)))

    d_pose = np.load(POSE, allow_pickle=True)
    pose_raw = align(clip_ids, d_pose["clip_id"], np.nan_to_num(d_pose["pose_features"]))

    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"

    # Z-score
    sc_e = StandardScaler().fit(explicit[tr]); explicit_z = sc_e.transform(explicit).astype(np.float32)
    sc_t = StandardScaler().fit(temporal[tr]); temporal_z = sc_t.transform(temporal).astype(np.float32)
    sc_p = StandardScaler().fit(pose_raw[tr]); pose_z = sc_p.transform(pose_raw).astype(np.float32)
    # SigLIP-L is already L2-normalized; leave as is

    # Subject int mapping (train only)
    unique = sorted(set(subj[tr].tolist()))
    sid_map = {s: i for i, s in enumerate(unique)}
    str_arr = np.array([sid_map[s] for s in subj[tr]], dtype=np.int64)
    n_subj = len(unique)
    log(f"  Train={tr.sum()} Val={va.sum()} Test={te.sum()}  n_subjects={n_subj}")

    # Build input feature matrix combinations
    configs = [
        {"name": "sig+single",        "X": np.concatenate([sig_feats, explicit_z], axis=1)},
        {"name": "sig+temporal",      "X": np.concatenate([sig_feats, temporal_z], axis=1)},
        {"name": "sig+temp+pose",     "X": np.concatenate([sig_feats, temporal_z, pose_z], axis=1)},
        {"name": "sig_only",          "X": sig_feats},
    ]
    hparams = {"d_hidden": 256, "d_proj": 128, "dropout": 0.3, "lr": 5e-4,
               "batch_size": 128, "epochs": 30, "temperature": 0.1,
               "contrastive_weight": 1.0, "subject_stratified": False}

    out = {"hparams": hparams}
    for cfg_grp in configs:
        log(f"\n=== {cfg_grp['name']} (dim {cfg_grp['X'].shape[1]}) ===")
        per_seed = []
        for seed in SEEDS:
            m, val_v = train_one(cfg_grp["X"][tr], eng[tr], str_arr,
                                   cfg_grp["X"][va], eng[va],
                                   cfg_grp["X"][te], eng[te],
                                   n_subj, hparams, seed)
            per_seed.append({"seed": seed, "val_kq": val_v, **m})
            log(f"  seed={seed}  val={val_v:.3f}  test κ_q={m['kappa_q']:.3f} {m['kappa_q_ci']}")
        ks = np.array([r["kappa_q"] for r in per_seed])
        summary = {"per_seed": per_seed, "mean_kq": float(ks.mean()),
                    "std_kq": float(ks.std(ddof=1) if len(ks) > 1 else 0.0),
                    "min_kq": float(ks.min()), "max_kq": float(ks.max()),
                    "dim_in": int(cfg_grp["X"].shape[1])}
        log(f"  => mean={summary['mean_kq']:.3f} ± {summary['std_kq']:.3f}")
        out[cfg_grp["name"]] = summary

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {OUT}")
    log("SUPCON_DONE")


if __name__ == "__main__":
    main()
