"""
Step 1: Sample 300 clips from DAiSEE test set and extract frames at t=5s.

Sampling strategy (exact reproduction of coauthor's procedure, seed=42):
  Level 0: take all 4
  Level 1: take all 84
  Level 2: random.sample(clips, 106)
  Level 3: random.sample(clips, 106)
  Then random.shuffle(combined list)

Output:
  ../sampled_test.csv          (clip_id, engagement, frame_path)
  ../sampled_frames/<clip_id>.jpg
"""

import csv
import os
import random
import subprocess
from collections import defaultdict

DAISEE_ROOT   = "/home/ubuntu/SCB-05-Dataset/DAISEE/DAiSEE"
LABELS_CSV    = os.path.join(DAISEE_ROOT, "Labels", "TestLabels.csv")
VIDEO_TEST    = os.path.join(DAISEE_ROOT, "DataSet", "Test")

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_DIR    = os.path.join(BASE_DIR, "sampled_frames")
OUTPUT_CSV    = os.path.join(BASE_DIR, "sampled_test.csv")

TARGET_PER_LEVEL = {
    0: None,   # take all
    1: None,   # take all
    2: 106,
    3: 106,
}

def find_video(clip_id):
    """Resolve clip_id (e.g. '5000441001.avi') to absolute video path."""
    stem = os.path.splitext(clip_id)[0]
    subject_id = stem[:6]
    path = os.path.join(VIDEO_TEST, subject_id, stem, clip_id)
    if os.path.isfile(path):
        return path
    # Try .mp4 fallback
    path_mp4 = os.path.join(VIDEO_TEST, subject_id, stem, stem + ".mp4")
    if os.path.isfile(path_mp4):
        return path_mp4
    return None

def extract_frame(video_path, out_jpg):
    """Extract frame at t=5s using ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-ss", "5", "-i", video_path,
         "-frames:v", "1", "-q:v", "2", out_jpg],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=True,
    )

def main():
    os.makedirs(FRAMES_DIR, exist_ok=True)

    # Load labels
    by_level = defaultdict(list)
    with open(LABELS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            level = int(row["Engagement"])
            by_level[level].append(row["ClipID"])

    print("Test set distribution:")
    for lvl in sorted(by_level):
        print(f"  Level {lvl}: {len(by_level[lvl])} clips")

    # Sample
    random.seed(42)
    sampled = []
    for level in sorted(by_level):
        clips = by_level[level]
        target = TARGET_PER_LEVEL[level]
        if target is None or len(clips) <= target:
            chosen = clips[:]
        else:
            chosen = random.sample(clips, target)
        for clip_id in chosen:
            sampled.append({"clip_id": clip_id, "engagement": level})

    random.shuffle(sampled)
    print(f"\nSampled {len(sampled)} clips total.")

    # Extract frames
    rows = []
    skipped = []
    for i, item in enumerate(sampled):
        clip_id   = item["clip_id"]
        level     = item["engagement"]
        out_jpg   = os.path.join(FRAMES_DIR, os.path.splitext(clip_id)[0] + ".jpg")

        video_path = find_video(clip_id)
        if video_path is None:
            print(f"  [SKIP] Video not found: {clip_id}")
            skipped.append(clip_id)
            continue

        if not os.path.isfile(out_jpg):
            try:
                extract_frame(video_path, out_jpg)
            except subprocess.CalledProcessError as e:
                print(f"  [ERROR] ffmpeg failed for {clip_id}: {e}")
                skipped.append(clip_id)
                continue

        rows.append({
            "clip_id":    clip_id,
            "engagement": level,
            "frame_path": out_jpg,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sampled)} done...")

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "engagement", "frame_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} frames extracted → {OUTPUT_CSV}")
    if skipped:
        print(f"Skipped {len(skipped)}: {skipped[:5]}{'...' if len(skipped)>5 else ''}")

    # Print final distribution
    from collections import Counter
    dist = Counter(r["engagement"] for r in rows)
    print("Final distribution:", dict(sorted(dist.items())))

if __name__ == "__main__":
    main()
