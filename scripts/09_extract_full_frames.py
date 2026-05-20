"""
Extract t=5s single frame for ALL DAiSEE clips across train/val/test splits.

Output:
  frames_full/{Train,Validation,Test}/<clip_id>.jpg
  frames_full/manifest.csv  (clip_id, split, subject_id, engagement, frame_path)

Resumable: skips frames already extracted.
Parallel: uses multiprocessing (8 workers) for ffmpeg calls.
"""

import os
import csv
import subprocess
from multiprocessing import Pool
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAISEE_DIR = os.path.join(BASE_DIR, "DAiSEE")
OUTPUT_DIR = os.path.join(BASE_DIR, "frames_full")
MANIFEST = os.path.join(OUTPUT_DIR, "manifest.csv")

SPLITS = {
    "Train": "TrainLabels.csv",
    "Validation": "ValidationLabels.csv",
    "Test": "TestLabels.csv",
}


def find_video_path(split, clip_id):
    base = clip_id.replace(".avi", "").replace(".mp4", "")
    subject = base[:6]
    clip_dir = os.path.join(DAISEE_DIR, "DataSet", split, subject, base)
    for ext in (".avi", ".mp4"):
        p = os.path.join(clip_dir, base + ext)
        if os.path.exists(p):
            return p
    return None


def extract_one(args):
    split, clip_id, engagement, out_path = args
    if os.path.exists(out_path):
        return (clip_id, split, engagement, out_path, "exists")
    video = find_video_path(split, clip_id)
    if video is None:
        return (clip_id, split, engagement, out_path, "missing_video")
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-ss", "5", "-i", video,
             "-frames:v", "1", "-q:v", "2", out_path],
            capture_output=True, timeout=30,
        )
        return (clip_id, split, engagement, out_path,
                "ok" if os.path.exists(out_path) else "ffmpeg_fail")
    except Exception as e:
        return (clip_id, split, engagement, out_path, f"err:{e}")


def build_jobs():
    jobs = []
    for split, label_file in SPLITS.items():
        out_dir = os.path.join(OUTPUT_DIR, split)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(DAISEE_DIR, "Labels", label_file)) as f:
            reader = csv.DictReader(f)
            for row in reader:
                clip_id = row["ClipID"].strip()
                engagement = int(row["Engagement"].strip())
                base = clip_id.replace(".avi", "").replace(".mp4", "")
                out_path = os.path.join(out_dir, base + ".jpg")
                jobs.append((split, clip_id, engagement, out_path))
    return jobs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    jobs = build_jobs()
    print(f"Total clips to process: {len(jobs)}")

    results = []
    status_counter = Counter()

    with Pool(processes=8) as pool:
        for i, res in enumerate(pool.imap_unordered(extract_one, jobs, chunksize=10), 1):
            results.append(res)
            status_counter[res[4]] += 1
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}  status: {dict(status_counter)}")

    # Write manifest
    with open(MANIFEST, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "split", "subject_id", "engagement", "frame_path", "status"])
        for clip_id, split, engagement, out_path, status in sorted(results):
            base = clip_id.replace(".avi", "").replace(".mp4", "")
            subject = base[:6]
            w.writerow([clip_id, split, subject, engagement, out_path, status])

    print(f"\nDone. Manifest: {MANIFEST}")
    print(f"Status: {dict(status_counter)}")
    ok = sum(1 for r in results if r[4] in ("ok", "exists"))
    print(f"Successfully extracted/cached: {ok}/{len(jobs)}")


if __name__ == "__main__":
    main()
