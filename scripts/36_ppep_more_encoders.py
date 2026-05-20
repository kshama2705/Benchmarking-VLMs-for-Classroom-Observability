"""
PPEP on SigLIP-L/16 and CLIP-L/14 (Day 2).

Re-encodes DAiSEE frames at their native patch grids:
  - SigLIP-L/16 with 256x256 input → 16x16 patch grid = 256 patches, 1024-dim
  - CLIP-L/14 with 224x224 input → 16x16 patch grid = 256 patches, 1024-dim

Vectorized face-mask computation (no nested Python loops over patches).

Output:
  features/siglip_l_patch_face_features.npz
  features/clip_l_14_patch_face_features.npz
  results/ppep/ppep_results_more_encoders.json
"""
import os, csv, json, time
import numpy as np
import torch
from PIL import Image
import open_clip
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(BASE, "frames_full", "manifest.csv")
BBOX_JSON = os.path.join(BASE, "features", "face_bboxes.json")
RES_OUT = os.path.join(BASE, "results", "ppep", "ppep_results_more_encoders.json")
os.makedirs(os.path.dirname(RES_OUT), exist_ok=True)
RNG = np.random.default_rng(42)
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))


def vectorized_face_mask(bbox, grid, face_thresh=0.3):
    """Compute (grid*grid,) face mask without Python loops."""
    if bbox is None:
        return np.zeros(grid * grid, dtype=np.int64)
    x, y, w, h = bbox
    # Patch index ranges (j is column / x-axis, i is row / y-axis)
    j_idx, i_idx = np.meshgrid(np.arange(grid), np.arange(grid))
    # Convert bbox to patch coords
    px0, py0 = x * grid, y * grid
    px1, py1 = (x + w) * grid, (y + h) * grid
    ix0 = np.maximum(j_idx, px0); iy0 = np.maximum(i_idx, py0)
    ix1 = np.minimum(j_idx + 1, px1); iy1 = np.minimum(i_idx + 1, py1)
    overlap = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    is_face = (overlap >= face_thresh).astype(np.int64).reshape(-1)
    return is_face


def get_open_clip_patches(model, x, encoder_label):
    """Return patch_tokens (B, N, D) and cls_token (B, D) for an open_clip model.
    Different open_clip backbones expose patches via different attributes."""
    # SigLIP and CLIP both use vision transformer; we can hook into visual.transformer
    # The simplest: use trunk_with_grid_pos approach via visual.forward returning intermediate
    # Easiest reliable path: pass through visual encoder and capture pre-pool tokens.
    v = model.visual
    # For ViT-based CLIP/SigLIP via open_clip, visual exposes `trunk` (TimmModel) or a `transformer`
    # Try a forward hook on the final block.
    hooks = []
    captured = {}
    def hook(mod, inp, out):
        # out can be tensor or tuple; for ViT block, it's the tensor (B, T, D)
        captured["x"] = out if isinstance(out, torch.Tensor) else out[0]
    # Find last transformer block
    found = False
    for name, mod in v.named_modules():
        if name.endswith("blocks") and hasattr(mod, "__iter__"):
            last = list(mod)[-1]
            hooks.append(last.register_forward_hook(hook))
            found = True
            break
        if "transformer.resblocks" in name:
            pass
    if not found:
        # Try direct named "transformer.resblocks"
        for name, mod in v.named_modules():
            if "transformer.resblocks" in name and isinstance(mod, torch.nn.Module):
                # iterate child indexes
                try:
                    children = list(mod.children())
                    if len(children) > 0:
                        hooks.append(children[-1].register_forward_hook(hook))
                        found = True
                        break
                except Exception:
                    pass
    # Run forward
    out = v(x)  # CLS feature; hook captures last-block tokens
    for h in hooks:
        h.remove()
    if "x" not in captured:
        raise RuntimeError(f"Could not hook tokens for {encoder_label}")
    tokens = captured["x"]  # (B, T, D) or (T, B, D)
    # Normalize to (B, T, D)
    if tokens.shape[1] != x.shape[0] and tokens.shape[0] == x.shape[0]:
        pass  # (B, T, D)
    else:
        # might be (T, B, D); transpose
        if tokens.shape[1] == x.shape[0]:
            tokens = tokens.transpose(0, 1)
    # OpenCLIP ViT-L/14 has prepended CLS at position 0
    cls = tokens[:, 0]
    patches = tokens[:, 1:]
    return patches, cls


def encode_one(label, model_name, pretrained, input_size, grid, out_path):
    if os.path.exists(out_path):
        print(f"  features exist at {out_path}, skipping")
        return
    print(f"\nEncoding DAiSEE with {label} (patches + face/bg pooling)...")
    bboxes = json.load(open(BBOX_JSON))

    rows = []
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "exists") and os.path.exists(r["frame_path"]):
                rows.append(r)
    print(f"  {len(rows)} frames")

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(DEVICE).eval()

    # Probe one image for patch shape
    sample = preprocess(Image.open(rows[0]["frame_path"]).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        patches, cls = get_open_clip_patches(model, sample, label)
    feat_dim = patches.shape[-1]
    n_patches = patches.shape[1]
    print(f"  patch tokens: {patches.shape}  (expected grid {grid}x{grid}={grid*grid})")

    BATCH = 16
    face_feats = np.zeros((len(rows), feat_dim), dtype=np.float32)
    bg_feats = np.zeros((len(rows), feat_dim), dtype=np.float32)
    cls_feats = np.zeros((len(rows), feat_dim), dtype=np.float32)
    mean_feats = np.zeros((len(rows), feat_dim), dtype=np.float32)
    face_counts = np.zeros(len(rows), dtype=np.int32)

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            imgs = torch.stack([preprocess(Image.open(r["frame_path"]).convert("RGB"))
                                for r in chunk]).to(DEVICE)
            patches, cls = get_open_clip_patches(model, imgs, label)
            # Compute face masks (vectorized for whole batch)
            masks = np.stack([vectorized_face_mask(bboxes.get(r["clip_id"], {}).get("bbox"),
                                                   grid=grid) for r in chunk])  # (B, grid*grid)
            # Handle patch count mismatch (SigLIP uses average pooling; no CLS)
            if patches.shape[1] != grid * grid:
                # Some ViTs may have CLS not stripped; adapt
                if patches.shape[1] == grid * grid + 1:
                    patches = patches[:, 1:]
                else:
                    # Use mean over what we have; mask shape will mismatch — fall back
                    print(f"  WARN: patches {patches.shape[1]} != {grid*grid}")
            masks_t = torch.from_numpy(masks).to(DEVICE).float()  # (B, grid*grid)
            face_t = masks_t  # (B, T)
            bg_t = 1.0 - masks_t
            face_sum = face_t.sum(dim=1, keepdim=True).clamp(min=1)
            bg_sum = bg_t.sum(dim=1, keepdim=True).clamp(min=1)
            face_emb = (patches * face_t.unsqueeze(-1)).sum(dim=1) / face_sum
            bg_emb = (patches * bg_t.unsqueeze(-1)).sum(dim=1) / bg_sum
            mean_emb = patches.mean(dim=1)
            face_feats[i:i + len(chunk)] = face_emb.cpu().numpy()
            bg_feats[i:i + len(chunk)] = bg_emb.cpu().numpy()
            cls_feats[i:i + len(chunk)] = cls.cpu().numpy()
            mean_feats[i:i + len(chunk)] = mean_emb.cpu().numpy()
            face_counts[i:i + len(chunk)] = masks.sum(axis=1)
            if (i // BATCH) % 30 == 0:
                dt = time.time() - t0
                print(f"  [{label}] {i + len(chunk)}/{len(rows)}  elapsed={dt:.0f}s", flush=True)

    np.savez_compressed(
        out_path,
        clip_id=np.array([r["clip_id"] for r in rows]),
        split=np.array([r["split"] for r in rows]),
        subject_id=np.array([r["subject_id"] for r in rows]),
        engagement=np.array([int(r["engagement"]) for r in rows], dtype=np.int64),
        face_feat=face_feats, bg_feat=bg_feats, cls_feat=cls_feats, mean_feat=mean_feats,
        face_patch_counts=face_counts,
    )
    print(f"  Saved: {out_path}")
    print(f"  face_patch_counts: median={int(np.median(face_counts))}  zero_count_frames={(face_counts == 0).sum()}")
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def boot_kq(yt, yp, n=1000):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def probe(Xtr, ytr, Xva, yva, Xte, yte, label):
    best = None
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                                 solver="lbfgs", random_state=42, n_jobs=-1)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": v, "clf": clf}
    yhat_lr = best["clf"].predict(Xte)
    m_lr = {"best_C": best["C"],
            "kappa_q": float(cohen_kappa_score(yte, yhat_lr, weights="quadratic")),
            "kappa_q_ci": boot_kq(yte, yhat_lr),
            "accuracy": float(accuracy_score(yte, yhat_lr)),
            "f1_macro": float(f1_score(yte, yhat_lr, average="macro", zero_division=0)),
            "pred_dist": {int(k): int((yhat_lr == k).sum()) for k in range(4)}}
    best = None
    for alpha in [1.0, 10.0, 100.0, 1000.0]:
        reg = Ridge(alpha=alpha, random_state=42)
        reg.fit(Xtr, ytr.astype(np.float32))
        v = cohen_kappa_score(yva, np.clip(np.round(reg.predict(Xva)), 0, 3).astype(int), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"a": alpha, "v": v, "reg": reg}
    yhat_rd = np.clip(np.round(best["reg"].predict(Xte)), 0, 3).astype(int)
    m_rd = {"best_alpha": best["a"],
            "kappa_q": float(cohen_kappa_score(yte, yhat_rd, weights="quadratic")),
            "kappa_q_ci": boot_kq(yte, yhat_rd),
            "accuracy": float(accuracy_score(yte, yhat_rd)),
            "f1_macro": float(f1_score(yte, yhat_rd, average="macro", zero_division=0)),
            "pred_dist": {int(k): int((yhat_rd == k).sum()) for k in range(4)}}
    print(f"  {label:10} LR    κ_q={m_lr['kappa_q']:.3f} {m_lr['kappa_q_ci']}  acc={m_lr['accuracy']:.3f}")
    print(f"  {label:10} Ridge κ_q={m_rd['kappa_q']:.3f} {m_rd['kappa_q_ci']}  acc={m_rd['accuracy']:.3f}")
    return {"logreg": m_lr, "ridge_ordinal": m_rd}


def run_probes(npz_path, label):
    d = np.load(npz_path, allow_pickle=True)
    splits = d["split"]; eng = d["engagement"]
    tr = splits == "Train"; va = splits == "Validation"; te = splits == "Test"
    out = {}
    for kind in ["face_feat", "bg_feat", "cls_feat", "mean_feat"]:
        Xt = d[kind].astype(np.float32)
        print(f"\n--- {label} | {kind} ---")
        out[kind] = probe(Xt[tr], eng[tr], Xt[va], eng[va], Xt[te], eng[te], kind)
    return out


def main():
    if not os.path.exists(BBOX_JSON):
        print(f"ERROR: face bboxes missing")
        return
    out = {}
    runs = [
        ("SigLIP-L/16", "ViT-L-16-SigLIP-256", "webli", 256, 16,
         os.path.join(BASE, "features", "siglip_l_patch_face_features.npz")),
        ("CLIP-L/14", "ViT-L-14", "laion2b_s32b_b82k", 224, 16,
         os.path.join(BASE, "features", "clip_l_14_patch_face_features.npz")),
    ]
    for label, mname, pre, isize, grid, opath in runs:
        try:
            encode_one(label, mname, pre, isize, grid, opath)
            out[label] = run_probes(opath, label)
        except Exception as e:
            import traceback
            print(f"  {label} FAILED: {e}")
            print(traceback.format_exc())
            out[label] = {"error": str(e)}

    with open(RES_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {RES_OUT}")


if __name__ == "__main__":
    main()
