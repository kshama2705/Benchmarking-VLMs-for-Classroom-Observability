"""
VideoMAE-base feature extraction on full DAiSEE — RESUMABLE.

Per clip: 16 evenly-spaced frames -> VideoMAE-base -> mean-pool over T tokens
            -> 768-d clip feature.

Resumable: every CHECKPOINT_EVERY clips, write a partial .npz with all
features so far. On restart, load the partial and skip already-done clips.

Output:
  features/daisee_videomae_features.npz     (final)
  features/daisee_videomae_partial.npz      (checkpoint)
  results/sota/videomae_status.txt          (append-only log)
"""
import os, csv, json, time, subprocess, tempfile, glob
import numpy as np
import torch
from PIL import Image
from transformers import VideoMAEModel, VideoMAEImageProcessor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAISEE_DIR = os.path.join(BASE, "DAiSEE")
FEAT_OUT = os.path.join(BASE, "features", "daisee_videomae_features.npz")
PARTIAL = os.path.join(BASE, "features", "daisee_videomae_partial.npz")
LOG = os.path.join(BASE, "results", "sota", "videomae_status_v2.txt")
os.makedirs(os.path.dirname(LOG), exist_ok=True)

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))
NUM_FRAMES = 16
CHECKPOINT_EVERY = 200


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


def extract_frames(video_path, n=NUM_FRAMES):
    """Pull n evenly-spaced frames from a 10s clip via one ffmpeg call."""
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
        if len(files) == 0:
            return None
        frames = [np.array(Image.open(f).convert("RGB")) for f in files[:n]]
        while len(frames) < n:
            frames.append(frames[-1])
    return frames[:n]


def main():
    if os.path.exists(FEAT_OUT):
        log(f"FINAL features exist at {FEAT_OUT}, exiting")
        return

    # Build clip list from manifest
    manifest_path = os.path.join(BASE, "frames_full", "manifest.csv")
    rows = []
    with open(manifest_path) as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "exists"):
                rows.append(r)
    N = len(rows)
    log(f"Total clips in manifest: {N}")

    feats = np.zeros((N, 768), dtype=np.float32)
    success = np.zeros(N, dtype=np.int8)
    if os.path.exists(PARTIAL):
        log(f"Loading partial checkpoint: {PARTIAL}")
        d = np.load(PARTIAL, allow_pickle=True)
        feats = d["feat"]
        success = d["success"]
        log(f"Already encoded: {int(success.sum())}/{N}")

    log("Loading VideoMAE-base...")
    proc = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(DEVICE).eval()

    clip_ids = np.array([r["clip_id"] for r in rows])
    splits = np.array([r["split"] for r in rows])
    subjects = np.array([r["subject_id"] for r in rows])
    engagement = np.array([int(r["engagement"]) for r in rows], dtype=np.int64)

    t0 = time.time(); seen = int(success.sum())
    with torch.no_grad():
        for i, r in enumerate(rows):
            if success[i] == 1:
                continue
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
                feat = out.last_hidden_state.mean(dim=1).squeeze(0)
                feats[i] = feat.cpu().numpy()
                success[i] = 1
            except Exception as e:
                log(f"  ERR on {r['clip_id']}: {e}")
                continue
            seen_now = int(success.sum())
            if seen_now % 50 == 0 and seen_now != seen:
                dt = time.time() - t0
                done = seen_now - int(success.sum() - 0)  # no-op
                rate = (seen_now - seen) / max(1.0, dt) if seen_now > seen else 0
                eta = (N - seen_now) / max(0.01, rate)
                log(f"  {seen_now}/{N}  rate={rate:.2f}/s  ETA={eta/60:.0f}min")
            if seen_now % CHECKPOINT_EVERY == 0 and seen_now != seen:
                np.savez_compressed(
                    PARTIAL,
                    clip_id=clip_ids, split=splits, subject_id=subjects,
                    engagement=engagement, feat=feats, success=success,
                )
                log(f"  checkpoint saved at {seen_now}/{N}")

    # Final save
    np.savez_compressed(
        FEAT_OUT,
        clip_id=clip_ids, split=splits, subject_id=subjects,
        engagement=engagement, feat=feats, success=success,
    )
    log(f"FINAL saved: {FEAT_OUT}  success={int(success.sum())}/{N}")


if __name__ == "__main__":
    main()
