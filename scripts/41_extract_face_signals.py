"""
Extract face blendshapes (AU-equivalent), head pose, and eye-gaze signals from
all DAiSEE frames using MediaPipe FaceLandmarker.

Per frame, we extract:
  - blendshapes: 52-dim AU-equivalent intensities (e.g., mouthSmile, browDown,
    eyeBlink, jawOpen). Identity-invariant by construction.
  - head pose: 3 Euler angles (yaw, pitch, roll) decomposed from the 4x4
    transformation matrix.
  - gaze proxy: eye-aware features computed from iris/eye landmarks. We compute
    eye-openness, gaze direction (horizontal + vertical) for each eye.
  - landmark stats: aggregate features over the 478 landmarks (xyz spread,
    facial geometry summary).

Output:
  features/daisee_face_signals.npz
    keys: clip_id, split, subject_id, engagement,
          blendshapes (N, 52),
          head_pose (N, 3),
          eye_gaze (N, 6),  # left/right eye yaw, pitch, openness
          landmark_summary (N, 12),  # face geometry summary
          detected (N,)  # 1 if face was detected, 0 otherwise
"""
import os, csv, json, time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_py
from mediapipe.tasks.python import vision

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(BASE, "frames_full", "manifest.csv")
MODEL = os.path.join(BASE, "models", "face_landmarker.task")
OUT = os.path.join(BASE, "features", "daisee_face_signals.npz")
LOG = os.path.join(BASE, "results", "face_signals_status.txt")
os.makedirs(os.path.dirname(LOG), exist_ok=True)


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


# Indexes on the 478-point MediaPipe face mesh for eyes and iris
# Standard MediaPipe iris landmarks:
LEFT_IRIS = [468, 469, 470, 471, 472]  # center + 4 boundary
RIGHT_IRIS = [473, 474, 475, 476, 477]
# Eye corners and key points (MediaPipe FaceMesh canonical indices)
LEFT_EYE_TOP = 159; LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33; LEFT_EYE_RIGHT = 133
RIGHT_EYE_TOP = 386; RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 362; RIGHT_EYE_RIGHT = 263


def decompose_rotation(m):
    """Decompose 4x4 transformation matrix into Euler angles (yaw, pitch, roll) in radians.
    Assumes rotation is in the upper-left 3x3 block."""
    R = m[:3, :3]
    # Singularity-safe Tait-Bryan decomposition (ZYX convention)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.array([yaw, pitch, roll], dtype=np.float32)


def eye_features(lm_arr):
    """Per-eye yaw, pitch, openness from iris vs eye-corner landmarks.
    lm_arr: (478, 3) numpy array of normalized [0,1] xyz landmarks.
    Returns 6-vector: [L_yaw, L_pitch, L_open, R_yaw, R_pitch, R_open]."""
    out = np.zeros(6, dtype=np.float32)
    # Left eye
    iris_l = lm_arr[LEFT_IRIS[0]]  # iris center
    el = lm_arr[LEFT_EYE_LEFT]; er = lm_arr[LEFT_EYE_RIGHT]
    et = lm_arr[LEFT_EYE_TOP]; eb = lm_arr[LEFT_EYE_BOTTOM]
    eye_w = max(np.linalg.norm(el[:2] - er[:2]), 1e-6)
    eye_h = max(np.linalg.norm(et[:2] - eb[:2]), 1e-6)
    eye_center_x = (el[0] + er[0]) / 2
    eye_center_y = (et[1] + eb[1]) / 2
    out[0] = (iris_l[0] - eye_center_x) / eye_w   # horizontal gaze (yaw)
    out[1] = (iris_l[1] - eye_center_y) / eye_h   # vertical gaze (pitch)
    out[2] = eye_h / eye_w                         # openness ratio
    # Right eye
    iris_r = lm_arr[RIGHT_IRIS[0]]
    el = lm_arr[RIGHT_EYE_LEFT]; er = lm_arr[RIGHT_EYE_RIGHT]
    et = lm_arr[RIGHT_EYE_TOP]; eb = lm_arr[RIGHT_EYE_BOTTOM]
    eye_w = max(np.linalg.norm(el[:2] - er[:2]), 1e-6)
    eye_h = max(np.linalg.norm(et[:2] - eb[:2]), 1e-6)
    eye_center_x = (el[0] + er[0]) / 2
    eye_center_y = (et[1] + eb[1]) / 2
    out[3] = (iris_r[0] - eye_center_x) / eye_w
    out[4] = (iris_r[1] - eye_center_y) / eye_h
    out[5] = eye_h / eye_w
    return out


def landmark_summary(lm_arr):
    """Geometric summary of the 478 landmarks: bbox size, centroid, depth spread.
    12 dims."""
    xs = lm_arr[:, 0]; ys = lm_arr[:, 1]; zs = lm_arr[:, 2]
    return np.array([
        xs.mean(), ys.mean(), zs.mean(),
        xs.std(), ys.std(), zs.std(),
        xs.min(), xs.max(), ys.min(), ys.max(),
        zs.min(), zs.max(),
    ], dtype=np.float32)


def main():
    open(LOG, "w").close()
    rows = []
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "exists") and os.path.exists(r["frame_path"]):
                rows.append(r)
    log(f"Extracting face signals from {len(rows)} frames")

    opts = vision.FaceLandmarkerOptions(
        base_options=mp_py.BaseOptions(model_asset_path=MODEL),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(opts)

    N = len(rows)
    blendshapes = np.zeros((N, 52), dtype=np.float32)
    head_pose = np.zeros((N, 3), dtype=np.float32)
    eye_gaze = np.zeros((N, 6), dtype=np.float32)
    lm_summary = np.zeros((N, 12), dtype=np.float32)
    detected = np.zeros(N, dtype=np.int8)

    t0 = time.time()
    for i, r in enumerate(rows):
        img = cv2.imread(r["frame_path"])
        if img is None:
            continue
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        res = detector.detect(mp_img)
        if not res.face_landmarks:
            continue
        detected[i] = 1
        # Blendshapes
        bs = res.face_blendshapes[0]
        for j, b in enumerate(bs[:52]):
            blendshapes[i, j] = b.score
        # Head pose
        try:
            head_pose[i] = decompose_rotation(np.array(res.facial_transformation_matrixes[0]))
        except Exception:
            pass
        # Landmarks
        lm = res.face_landmarks[0]
        lm_arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        # Eyes
        try:
            eye_gaze[i] = eye_features(lm_arr)
        except Exception:
            pass
        # Geometry
        lm_summary[i] = landmark_summary(lm_arr)

        if (i + 1) % 500 == 0:
            dt = time.time() - t0
            rate = (i + 1) / dt
            eta = (N - i - 1) / rate
            log(f"  {i+1}/{N}  rate={rate:.1f} fps  elapsed={dt:.0f}s  ETA={eta:.0f}s  "
                f"detected={detected[:i+1].sum()}/{i+1} ({100*detected[:i+1].sum()/(i+1):.1f}%)")

    np.savez_compressed(
        OUT,
        clip_id=np.array([r["clip_id"] for r in rows]),
        split=np.array([r["split"] for r in rows]),
        subject_id=np.array([r["subject_id"] for r in rows]),
        engagement=np.array([int(r["engagement"]) for r in rows], dtype=np.int64),
        blendshapes=blendshapes,
        head_pose=head_pose,
        eye_gaze=eye_gaze,
        landmark_summary=lm_summary,
        detected=detected,
    )
    log(f"Saved: {OUT}")
    log(f"Detection rate: {detected.sum()}/{N} ({100*detected.sum()/N:.1f}%)")
    log("FACE_SIGNALS_DONE")


if __name__ == "__main__":
    main()
