"""
Extract t=2s and t=8s frames for all DAiSEE clips (t=5s already exists in frames_full/).

Output:
  frames_full_multi/{Train,Validation,Test}/<clip_id>_t{2,8}.jpg
"""

import os
import csv
import subprocess
from multiprocessing import Pool
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAISEE_DIR = os.path.join(BASE_DIR, "DAiSEE")
OUT_DIR = os.path.join(BASE_DIR, "frames_full_multi")

SPLITS = {
    "Train": "TrainLabels.csv",
    "Validation": "ValidationLabels.csv",
    "Test": "TestLabels.csv",
}
TIMESTAMPS = [2, 8]


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
    split, clip_id, t, out_path = args
    if os.path.exists(out_path):
        return "exists"
    video = find_video_path(split, clip_id)
    if video is None:
        return "missing_video"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(t), "-i", video,
             "-frames:v", "1", "-q:v", "2", out_path],
            capture_output=True, timeout=30,
        )
        return "ok" if os.path.exists(out_path) else "ffmpeg_fail"
    except Exception as e:
        return f"err:{e}"


def build_jobs():
    jobs = []
    for split, label_file in SPLITS.items():
        out_dir = os.path.join(OUT_DIR, split)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(DAISEE_DIR, "Labels", label_file)) as f:
            reader = csv.DictReader(f)
            for row in reader:
                clip_id = row["ClipID"].strip()
                base = clip_id.replace(".avi", "").replace(".mp4", "")
                for t in TIMESTAMPS:
                    out_path = os.path.join(out_dir, f"{base}_t{t}.jpg")
                    jobs.append((split, clip_id, t, out_path))
    return jobs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = build_jobs()
    print(f"Total frame jobs: {len(jobs)}  (2 timestamps × clips)")

    counter = Counter()
    with Pool(processes=8) as pool:
        for i, status in enumerate(pool.imap_unordered(extract_one, jobs, chunksize=20), 1):
            counter[status] += 1
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}  status: {dict(counter)}")
    print(f"Done. {dict(counter)}")


if __name__ == "__main__":
    main()
