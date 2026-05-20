"""
Step 1: Sample 300 clips from DAiSEE test set (stratified) and extract middle frame.

Output:
  - sampled_frames/  directory with extracted .jpg frames
  - sampled_test.csv with columns: clip_id, engagement, frame_path
"""

import os
import csv
import random
import subprocess
import shutil

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAISEE_DIR = os.path.join(BASE_DIR, "DAiSEE")
DATASET_DIR = os.path.join(DAISEE_DIR, "DataSet", "Test")
LABELS_FILE = os.path.join(DAISEE_DIR, "Labels", "TestLabels.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "sampled_frames")
OUTPUT_CSV = os.path.join(BASE_DIR, "sampled_test.csv")

# Target samples per engagement level (stratified oversampling of rare classes)
TARGET_PER_LEVEL = {
    0: None,   # take all (only 4)
    1: None,   # take all (only 84)
    2: 106,
    3: 106,
}

def load_labels():
    """Load test labels and group by engagement level."""
    by_level = {0: [], 1: [], 2: [], 3: []}
    with open(LABELS_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = row["ClipID"].strip()
            engagement = int(row["Engagement"].strip())
            by_level[engagement].append(clip_id)
    return by_level


def find_video_path(clip_id):
    """Find the full path to a video clip in the dataset."""
    # clip_id format: 5000441001.avi -> subject folder is first 6 chars
    base_name = clip_id.replace(".avi", "").replace(".mp4", "")
    for subject in os.listdir(DATASET_DIR):
        subject_path = os.path.join(DATASET_DIR, subject)
        if not os.path.isdir(subject_path):
            continue
        clip_dir = os.path.join(subject_path, base_name)
        if os.path.isdir(clip_dir):
            # Try both .avi and .mp4 extensions
            for ext in [".avi", ".mp4"]:
                video_path = os.path.join(clip_dir, base_name + ext)
                if os.path.exists(video_path):
                    return video_path
    return None


def extract_middle_frame(video_path, output_path):
    """Extract the middle frame (at 5 seconds for 10-sec clips) from a video."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", "5",
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
        print(f"  Error extracting frame: {e}")
        return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and sample
    by_level = load_labels()
    sampled = []

    for level in sorted(by_level.keys()):
        clips = by_level[level]
        target = TARGET_PER_LEVEL[level]
        if target is None or target >= len(clips):
            selected = clips
        else:
            selected = random.sample(clips, target)
        print(f"Level {level}: {len(clips)} available, {len(selected)} selected")
        for clip_id in selected:
            sampled.append((clip_id, level))

    random.shuffle(sampled)
    print(f"\nTotal sampled: {len(sampled)} clips")

    # Extract frames
    results = []
    success = 0
    for i, (clip_id, engagement) in enumerate(sampled):
        video_path = find_video_path(clip_id)
        if video_path is None:
            print(f"  [{i+1}/{len(sampled)}] SKIP {clip_id} - video not found")
            continue

        base_id = clip_id.replace(".avi", "").replace(".mp4", "")
        frame_name = base_id + ".jpg"
        frame_path = os.path.join(OUTPUT_DIR, frame_name)

        if extract_middle_frame(video_path, frame_path):
            results.append({
                "clip_id": clip_id,
                "engagement": engagement,
                "frame_path": frame_path,
            })
            success += 1
            if success % 50 == 0:
                print(f"  [{success}/{len(sampled)}] frames extracted...")
        else:
            print(f"  [{i+1}/{len(sampled)}] FAIL {clip_id} - extraction failed")

    # Save CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "engagement", "frame_path"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone! {success} frames saved to {OUTPUT_DIR}")
    print(f"CSV saved to {OUTPUT_CSV}")

    # Print final distribution
    from collections import Counter
    dist = Counter(r["engagement"] for r in results)
    for level in sorted(dist.keys()):
        print(f"  Level {level}: {dist[level]}")


if __name__ == "__main__":
    main()
