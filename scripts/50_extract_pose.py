"""
Extract MediaPipe Pose Landmarker body pose features for all DAiSEE frames.

Computes 33 body landmarks per frame; derives compact pose features:
  - shoulder torso angle (forward lean indicator)
  - head-shoulder vertical offset (slouching indicator)
  - head pitch (from shoulder line)
  - presence flag (was any body landmark detected)

Output:
  features/daisee_pose_signals.npz
"""
import os, csv, time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_py
from mediapipe.tasks.python import vision

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(BASE, "frames_full", "manifest.csv")
MODEL = os.path.join(BASE, "models", "pose_landmarker_lite.task")
OUT = os.path.join(BASE, "features", "daisee_pose_signals.npz")
LOG = os.path.join(BASE, "results", "pose_signals_status.txt")
os.makedirs(os.path.dirname(LOG), exist_ok=True)

# Pose landmark indices (from MediaPipe Pose)
NOSE = 0
LEFT_EYE = 2; RIGHT_EYE = 5
LEFT_EAR = 7; RIGHT_EAR = 8
LEFT_SHOULDER = 11; RIGHT_SHOULDER = 12
LEFT_ELBOW = 13; RIGHT_ELBOW = 14
LEFT_HIP = 23; RIGHT_HIP = 24


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def pose_features(lm):
    """lm: list of 33 NormalizedLandmark, each .x .y .z in [0,1].
    Returns 16-dim feature vector."""
    arr = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32)
    nose = arr[NOSE, :3]
    le = arr[LEFT_SHOULDER, :3]; re = arr[RIGHT_SHOULDER, :3]
    lh = arr[LEFT_HIP, :3]; rh = arr[RIGHT_HIP, :3]
    le_eye = arr[LEFT_EYE, :3]; re_eye = arr[RIGHT_EYE, :3]
    # Shoulder midpoint
    sh_mid = (le + re) / 2
    hip_mid = (lh + rh) / 2
    eye_mid = (le_eye + re_eye) / 2
    # Shoulder spread (proxy for distance to camera)
    sh_dist = float(np.linalg.norm(le[:2] - re[:2]))
    # Head-shoulder vertical distance (slouch indicator)
    head_sh_dy = float(eye_mid[1] - sh_mid[1])
    # Torso angle (forward lean): shoulder-to-hip vector relative to vertical
    torso_v = hip_mid[:2] - sh_mid[:2]
    torso_angle = float(np.arctan2(torso_v[0], -torso_v[1]))  # 0 = upright, +/- = tilted
    # Nose forward of shoulders (z-axis): smaller z = closer to camera
    nose_z_rel = float(nose[2] - sh_mid[2])
    # Visibility of each landmark group
    vis_shoulders = float(min(arr[LEFT_SHOULDER, 3], arr[RIGHT_SHOULDER, 3]))
    vis_face = float(min(arr[LEFT_EYE, 3], arr[RIGHT_EYE, 3], arr[NOSE, 3]))
    vis_hips = float(min(arr[LEFT_HIP, 3], arr[RIGHT_HIP, 3]))
    # Head yaw from ear positions: if both ears visible, distance ratio
    left_ear_z = float(arr[LEFT_EAR, 2])
    right_ear_z = float(arr[RIGHT_EAR, 2])
    head_yaw_proxy = left_ear_z - right_ear_z
    # Elbow positions relative to shoulders (high elbows = hands up)
    le_elbow_y_rel = float(arr[LEFT_ELBOW, 1] - le[1])
    re_elbow_y_rel = float(arr[RIGHT_ELBOW, 1] - re[1])
    return np.array([
        sh_dist, head_sh_dy, torso_angle, nose_z_rel,
        vis_shoulders, vis_face, vis_hips, head_yaw_proxy,
        le_elbow_y_rel, re_elbow_y_rel,
        float(sh_mid[0]), float(sh_mid[1]), float(sh_mid[2]),
        float(hip_mid[0]), float(hip_mid[1]), float(hip_mid[2]),
    ], dtype=np.float32)


def main():
    open(LOG, "w").close()
    rows = []
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "exists") and os.path.exists(r["frame_path"]):
                rows.append(r)
    log(f"Extracting pose signals from {len(rows)} frames")

    opts = vision.PoseLandmarkerOptions(
        base_options=mp_py.BaseOptions(model_asset_path=MODEL),
        num_poses=1,
    )
    detector = vision.PoseLandmarker.create_from_options(opts)

    N = len(rows)
    pose_feats = np.zeros((N, 16), dtype=np.float32)
    detected = np.zeros(N, dtype=np.int8)

    t0 = time.time()
    for i, r in enumerate(rows):
        img = cv2.imread(r["frame_path"])
        if img is None:
            continue
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        res = detector.detect(mp_img)
        if not res.pose_landmarks:
            continue
        detected[i] = 1
        try:
            pose_feats[i] = pose_features(res.pose_landmarks[0])
        except Exception:
            pass
        if (i + 1) % 500 == 0:
            dt = time.time() - t0
            rate = (i + 1) / dt
            log(f"  {i+1}/{N}  rate={rate:.1f} fps  elapsed={dt:.0f}s  "
                f"detected={detected[:i+1].sum()}/{i+1} ({100*detected[:i+1].sum()/(i+1):.1f}%)")

    np.savez_compressed(
        OUT,
        clip_id=np.array([r["clip_id"] for r in rows]),
        split=np.array([r["split"] for r in rows]),
        subject_id=np.array([r["subject_id"] for r in rows]),
        engagement=np.array([int(r["engagement"]) for r in rows], dtype=np.int64),
        pose_features=pose_feats,
        detected=detected,
    )
    log(f"Saved: {OUT}  detection_rate={100*detected.sum()/N:.1f}%")
    log("POSE_SIGNALS_DONE")


if __name__ == "__main__":
    main()
