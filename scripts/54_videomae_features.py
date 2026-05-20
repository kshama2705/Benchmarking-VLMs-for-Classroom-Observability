"""
Extract VideoMAE features for full DAiSEE clips (10-second videos, 16 evenly-spaced frames).

Per clip:
  - decode 16 uniform frames via ffmpeg
  - run VideoMAE-base forward → (1, 1568, 768) hidden states
  - mean-pool spatially and temporally → 768-d clip-level feature

This gives the first proper temporal/video features for the engagement task,
distinct from our t=2/5/8 single-frame baselines.

Output:
  features/daisee_videomae_features.npz
  results/sota/videomae_probe.json
"""
import os, csv, json, time, subprocess
import numpy as np
import torch
from PIL import Image
from transformers import VideoMAEModel, VideoMAEImageProcessor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAISEE_DIR = os.path.join(BASE, "DAiSEE")
FEAT_OUT = os.path.join(BASE, "features", "daisee_videomae_features.npz")
RES_OUT = os.path.join(BASE, "results", "sota", "videomae_probe.json")
LOG = os.path.join(BASE, "results", "sota", "videomae_status.txt")
os.makedirs(os.path.dirname(RES_OUT), exist_ok=True)
RNG = np.random.default_rng(42)
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))
NUM_FRAMES = 16


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def find_video(split, clip_id):
    base = clip_id.replace(".avi", "").replace(".mp4", "")
    subject = base[:6]
    clip_dir = os.path.join(DAISEE_DIR, "DataSet", split, subject, base)
    for ext in (".avi", ".mp4"):
        p = os.path.join(clip_dir, base + ext)
        if os.path.exists(p):
            return p
    return None


def extract_frames(video_path, n=16):
    """Use ffmpeg to extract n evenly-spaced frames from a 10s clip.
    Returns list of np.uint8 RGB arrays."""
    # Use select filter for uniform sampling. Clip is 10s; we want n frames at uniform times.
    # ffmpeg -i in -vf select='eq(mod(n,K))' or just use -vf fps=
    # Simpler: get frames at specific timestamps t=0.5, 1.5, ..., 9.5 (10/n=0.625 spacing if n=16; positions [0.3125, 0.9375, ..., 9.6875])
    timestamps = np.linspace(0.5, 9.5, n)  # 0.5 to 9.5 in n steps
    frames = []
    for t in timestamps:
        # Use ffmpeg to extract single frame at t
        result = subprocess.run([
            "ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", video_path,
            "-frames:v", "1", "-q:v", "2", "-f", "image2pipe",
            "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "pipe:1",
        ], capture_output=True, timeout=30)
        if result.returncode != 0 or len(result.stdout) == 0:
            return None
        # Determine size from header — actually with rawvideo we just need to know dimensions
        # Simpler: use ffmpeg to write a temp JPEG and load it back
        # ... actually use a different approach: ffmpeg output to a directory of JPEGs
        # Rewrite: extract all 16 frames in one ffmpeg call to a tempdir
        break  # Use the simpler approach below

    # Simpler approach: one ffmpeg call with multiple outputs via -vf fps
    # For 10s clip at n=16 frames, fps = n/10 = 1.6 fps
    import tempfile, glob
    with tempfile.TemporaryDirectory() as td:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"fps={n}/10",
            "-q:v", "2",
            os.path.join(td, "f_%03d.jpg"),
        ], capture_output=True, timeout=60)
        if result.returncode != 0:
            return None
        files = sorted(glob.glob(os.path.join(td, "f_*.jpg")))
        if len(files) < n:
            # Pad with last frame
            if len(files) == 0:
                return None
        for f in files[:n]:
            img = Image.open(f).convert("RGB")
            frames.append(np.array(img))
        while len(frames) < n:
            frames.append(frames[-1])
    return frames[:n]


def encode():
    if os.path.exists(FEAT_OUT):
        log(f"features exist at {FEAT_OUT}, skipping")
        return

    log("Loading VideoMAE-base...")
    proc = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(DEVICE).eval()

    # Build clip list from manifest
    manifest_path = os.path.join(BASE, "frames_full", "manifest.csv")
    rows = []
    with open(manifest_path) as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "exists"):
                rows.append(r)
    log(f"Processing {len(rows)} clips")

    N = len(rows)
    feats = np.zeros((N, 768), dtype=np.float32)
    success = np.zeros(N, dtype=np.int8)

    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(rows):
            video_path = find_video(r["split"], r["clip_id"])
            if video_path is None:
                continue
            frames = extract_frames(video_path, n=NUM_FRAMES)
            if frames is None:
                continue
            try:
                inputs = proc(frames, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(DEVICE)
                out = model(pixel_values=pixel_values)
                # last_hidden_state: (1, T, 768). Mean-pool.
                feat = out.last_hidden_state.mean(dim=1).squeeze(0)
                feats[i] = feat.cpu().numpy()
                success[i] = 1
            except Exception as e:
                log(f"  ERR on {r['clip_id']}: {e}")
                continue
            if (i + 1) % 50 == 0:
                dt = time.time() - t0
                rate = (i + 1) / dt
                eta = (N - i - 1) / rate
                log(f"  {i+1}/{N}  rate={rate:.2f} clips/s  elapsed={dt:.0f}s  ETA={eta:.0f}s  "
                    f"success={int(success[:i+1].sum())}/{i+1}")

    np.savez_compressed(
        FEAT_OUT,
        clip_id=np.array([r["clip_id"] for r in rows]),
        split=np.array([r["split"] for r in rows]),
        subject_id=np.array([r["subject_id"] for r in rows]),
        engagement=np.array([int(r["engagement"]) for r in rows], dtype=np.int64),
        feat=feats, success=success,
    )
    log(f"Saved: {FEAT_OUT}  success={int(success.sum())}/{N}")


def boot_kq(yt, yp, n=1000):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def fit_lr(Xtr, ytr, Xva, yva):
    best = None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=10000,
                                 solver="lbfgs", random_state=42)
        clf.fit(Xtr, ytr)
        v = cohen_kappa_score(yva, clf.predict(Xva), weights="quadratic")
        if best is None or v > best["v"]:
            best = {"C": C, "v": float(v), "clf": clf}
    return best["clf"], best["v"], best["C"]


def metrics(yt, yp):
    return {
        "kappa_q": float(cohen_kappa_score(yt, yp, weights="quadratic")),
        "kappa_q_ci": boot_kq(yt, yp),
        "accuracy": float(accuracy_score(yt, yp)),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "pred_dist": {int(k): int((yp == k).sum()) for k in range(4)},
    }


def probe_and_fuse():
    d = np.load(FEAT_OUT, allow_pickle=True)
    X = d["feat"].astype(np.float32)
    eng = d["engagement"].astype(np.int64)
    splits = d["split"]; clip_ids = d["clip_id"]
    success = d["success"]
    log(f"VideoMAE features loaded: success rate {success.sum()}/{len(success)} ({100*success.sum()/len(success):.1f}%)")
    keep = success == 1
    log(f"  using {keep.sum()} clips")

    tr = (splits == "Train") & keep
    va = (splits == "Validation") & keep
    te = (splits == "Test") & keep
    log(f"  Train={tr.sum()} Val={va.sum()} Test={te.sum()}")

    out = {}
    clf, val_v, C = fit_lr(X[tr], eng[tr], X[va], eng[va])
    yhat = clf.predict(X[te])
    out["videomae_solo_LR"] = {**metrics(eng[te], yhat), "val_kq": val_v, "C": C, "dim": X.shape[1]}
    log(f"  VideoMAE solo LR κ_q={out['videomae_solo_LR']['kappa_q']:.3f} {out['videomae_solo_LR']['kappa_q_ci']}  val={val_v:.3f}")

    with open(RES_OUT, "w") as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {RES_OUT}")
    log("VIDEOMAE_DONE")


def main():
    open(LOG, "w").close()
    encode()
    probe_and_fuse()


if __name__ == "__main__":
    main()
