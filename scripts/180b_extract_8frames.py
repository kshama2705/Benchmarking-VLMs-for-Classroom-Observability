"""
Helper: pre-extract 8 frames per DAiSEE clip at t = 0.5, 1.7, 2.9, 4.1, 5.3, 6.5, 7.7, 8.9 s.

Run this once on any machine that has ffmpeg + the raw DAiSEE videos.
Resulting layout:
  <out>/<split>/<clip_id>/t00.jpg
  <out>/<split>/<clip_id>/t01.jpg
  ...
  <out>/<split>/<clip_id>/t07.jpg

Usage:
  python scripts/180b_extract_8frames.py --src DAiSEE/DataSet --out frames_8

This is mandatory before running 180_supervised_e2e_gpu.py on a GPU box that
does not have the source videos.
"""
import os, csv, subprocess, argparse, time
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                        help="DAiSEE/DataSet root (contains Train/, Validation/, Test/)")
    parser.add_argument("--out", default="frames_8",
                        help="Output directory for extracted frames")
    parser.add_argument("--n_frames", type=int, default=8)
    args = parser.parse_args()

    timestamps = np.linspace(0.5, 8.9, args.n_frames).tolist()
    print(f"Extracting {args.n_frames} frames at t = {timestamps}", flush=True)

    n_done = 0; n_fail = 0; t0 = time.time()
    for split in ("Train", "Validation", "Test"):
        split_dir = os.path.join(args.src, split)
        if not os.path.isdir(split_dir):
            print(f"skip {split_dir}: not a dir", flush=True); continue
        for subject in os.listdir(split_dir):
            sub_dir = os.path.join(split_dir, subject)
            if not os.path.isdir(sub_dir): continue
            for clip in os.listdir(sub_dir):
                clip_dir = os.path.join(sub_dir, clip)
                if not os.path.isdir(clip_dir): continue
                video = None
                for ext in (".avi", ".mp4"):
                    p = os.path.join(clip_dir, clip + ext)
                    if os.path.exists(p): video = p; break
                if video is None: continue
                out_clip = os.path.join(args.out, split, clip)
                # Skip if already done
                done = all(os.path.exists(os.path.join(out_clip, f"t{i:02d}.jpg"))
                           for i in range(args.n_frames))
                if done:
                    n_done += 1; continue
                os.makedirs(out_clip, exist_ok=True)
                ok = True
                for i, t in enumerate(timestamps):
                    fp = os.path.join(out_clip, f"t{i:02d}.jpg")
                    r = subprocess.run([
                        "ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", video,
                        "-frames:v", "1", "-q:v", "2", "-update", "1", fp,
                    ], capture_output=True, timeout=20)
                    if r.returncode != 0 or not os.path.exists(fp):
                        ok = False; break
                if ok:
                    n_done += 1
                else:
                    n_fail += 1
                if (n_done + n_fail) % 200 == 0:
                    dt = time.time() - t0
                    rate = (n_done + n_fail) / max(1, dt)
                    print(f"  {n_done} ok, {n_fail} fail  rate={rate:.2f} clip/s", flush=True)

    print(f"\nFINAL: ok={n_done}  fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
