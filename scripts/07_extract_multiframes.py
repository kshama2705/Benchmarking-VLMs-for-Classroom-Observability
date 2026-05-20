"""
Step 7: Extract 3 frames per clip (t=2s, t=5s, t=8s) from the 300 sampled DAiSEE clips.

This supports the multi-frame temporal ablation. The t=5s frame is identical to
the single-frame used in the original experiments, enabling direct comparison.

Output:
  - sampled_frames_multiframe/  directory with {clip_id}_t2.jpg, _t5.jpg, _t8.jpg
  - sampled_test_multiframe.csv with columns: clip_id, engagement, frame_t2, frame_t5, frame_t8

Usage:
  python 07_extract_multiframes.py
"""

import csv
import os
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAISEE_DIR = os.path.join(BASE_DIR, "DAiSEE")
DATASET_DIR = os.path.join(DAISEE_DIR, "DataSet", "Test")
SAMPLED_CSV = os.path.join(BASE_DIR, "sampled_test.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "sampled_frames_multiframe")
OUTPUT_CSV = os.path.join(BASE_DIR, "sampled_test_multiframe.csv")

FRAME_TIMES = [2, 5, 8]  # seconds into each 10-second clip


def find_video_path(clip_id):
    """Find the full path to a video clip in the dataset."""
    base_name = clip_id.replace(".avi", "").replace(".mp4", "")
    for subject in os.listdir(DATASET_DIR):
        subject_path = os.path.join(DATASET_DIR, subject)
        if not os.path.isdir(subject_path):
            continue
        clip_dir = os.path.join(subject_path, base_name)
        if os.path.isdir(clip_dir):
            for ext in [".avi", ".mp4"]:
                video_path = os.path.join(clip_dir, base_name + ext)
                if os.path.exists(video_path):
                    return video_path
    return None


def extract_frame(video_path, output_path, timestamp):
    """Extract a single frame at the given timestamp (seconds)."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(timestamp),
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",
                output_path,
            ],
            capture_output=True,
            timeout=30,
        )
        return os.path.exists(output_path)
    except Exception as e:
        print(f"  Error extracting frame at t={timestamp}s: {e}")
        return False


def load_sampled_clips():
    clips = []
    with open(SAMPLED_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            clips.append({
                "clip_id": row["clip_id"],
                "engagement": int(row["engagement"]),
            })
    return clips


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    clips = load_sampled_clips()
    print(f"Processing {len(clips)} clips, extracting {len(FRAME_TIMES)} frames each...")

    results = []
    success = 0
    skipped = 0

    for i, clip in enumerate(clips):
        clip_id = clip["clip_id"]
        base_id = clip_id.replace(".avi", "").replace(".mp4", "")

        video_path = find_video_path(clip_id)
        if video_path is None:
            print(f"  [{i+1}/{len(clips)}] SKIP {clip_id} - video not found")
            skipped += 1
            continue

        frame_paths = {}
        all_ok = True
        for t in FRAME_TIMES:
            frame_name = f"{base_id}_t{t}.jpg"
            frame_path = os.path.join(OUTPUT_DIR, frame_name)
            frame_paths[f"frame_t{t}"] = frame_path

            # Skip if already extracted
            if os.path.exists(frame_path):
                continue

            if not extract_frame(video_path, frame_path, t):
                print(f"  [{i+1}/{len(clips)}] FAIL {clip_id} at t={t}s")
                all_ok = False
                break

        if all_ok:
            results.append({
                "clip_id": clip_id,
                "engagement": clip["engagement"],
                "frame_t2": frame_paths["frame_t2"],
                "frame_t5": frame_paths["frame_t5"],
                "frame_t8": frame_paths["frame_t8"],
            })
            success += 1
            if success % 50 == 0:
                print(f"  [{success}/{len(clips)}] clips extracted...")

    # Save CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["clip_id", "engagement", "frame_t2", "frame_t5", "frame_t8"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone! {success} clips saved to {OUTPUT_DIR}")
    print(f"  Skipped: {skipped} (video not found)")
    print(f"  CSV saved to {OUTPUT_CSV}")

    from collections import Counter
    dist = Counter(r["engagement"] for r in results)
    print("\nLabel distribution:")
    for level in sorted(dist.keys()):
        print(f"  Level {level}: {dist[level]}")


if __name__ == "__main__":
    main()
