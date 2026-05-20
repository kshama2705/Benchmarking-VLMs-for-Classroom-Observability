"""
Extract MediaPipe face signals on the t=2, t=5, t=8 frames per clip
(frames already extracted to frames_full/ and frames_full_multi/).
Build temporal features:
  - mean over 3 frames
  - std over 3 frames
  - first-derivative magnitude (sum of |x_t - x_{t-1}|)

Output:
  features/daisee_face_signals_temporal.npz
  Keys: clip_id, split, subject_id, engagement,
        blendshapes_mean (52), blendshapes_std (52), blendshapes_delta (52),
        head_pose_mean (3), head_pose_std (3),
        eye_gaze_mean (6), eye_gaze_std (6),
        landmark_summary_mean (12),
        detected_count (1)  # how many of 3 frames had a detected face
"""
import os, csv, json, time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_py
from mediapipe.tasks.python import vision

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(BASE, "frames_full", "manifest.csv")
MULTI_DIR = os.path.join(BASE, "frames_full_multi")
MODEL = os.path.join(BASE, "models", "face_landmarker.task")
OUT = os.path.join(BASE, "features", "daisee_face_signals_temporal.npz")
LOG = os.path.join(BASE, "results", "face_signals_temporal_status.txt")
os.makedirs(os.path.dirname(LOG), exist_ok=True)

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_TOP = 159; LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33; LEFT_EYE_RIGHT = 133
RIGHT_EYE_TOP = 386; RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 362; RIGHT_EYE_RIGHT = 263


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def decompose_rotation(m):
    R = m[:3, :3]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy < 1e-6:
        return np.array([0.0, np.arctan2(-R[2, 0], sy), np.arctan2(-R[1, 2], R[1, 1])], dtype=np.float32)
    return np.array([np.arctan2(R[1, 0], R[0, 0]), np.arctan2(-R[2, 0], sy),
                     np.arctan2(R[2, 1], R[2, 2])], dtype=np.float32)


def eye_features(lm_arr):
    out = np.zeros(6, dtype=np.float32)
    for k, (iris_i, el_i, er_i, et_i, eb_i, offset) in enumerate([
        (LEFT_IRIS[0], LEFT_EYE_LEFT, LEFT_EYE_RIGHT, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, 0),
        (RIGHT_IRIS[0], RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, 3),
    ]):
        iris = lm_arr[iris_i]
        el = lm_arr[el_i]; er = lm_arr[er_i]
        et = lm_arr[et_i]; eb = lm_arr[eb_i]
        eye_w = max(np.linalg.norm(el[:2] - er[:2]), 1e-6)
        eye_h = max(np.linalg.norm(et[:2] - eb[:2]), 1e-6)
        cx = (el[0] + er[0]) / 2; cy = (et[1] + eb[1]) / 2
        out[offset]     = (iris[0] - cx) / eye_w
        out[offset + 1] = (iris[1] - cy) / eye_h
        out[offset + 2] = eye_h / eye_w
    return out


def landmark_summary(lm_arr):
    xs = lm_arr[:, 0]; ys = lm_arr[:, 1]; zs = lm_arr[:, 2]
    return np.array([
        xs.mean(), ys.mean(), zs.mean(),
        xs.std(), ys.std(), zs.std(),
        xs.min(), xs.max(), ys.min(), ys.max(),
        zs.min(), zs.max(),
    ], dtype=np.float32)


def extract_one(detector, image_path):
    """Return (blendshapes_52, head_pose_3, eye_gaze_6, landmark_summary_12, detected_bool)."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    res = detector.detect(mp_img)
    if not res.face_landmarks:
        return None
    bs = np.array([b.score for b in res.face_blendshapes[0][:52]], dtype=np.float32)
    try:
        hp = decompose_rotation(np.array(res.facial_transformation_matrixes[0]))
    except Exception:
        hp = np.zeros(3, dtype=np.float32)
    lm = res.face_landmarks[0]
    lm_arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    try:
        eg = eye_features(lm_arr)
    except Exception:
        eg = np.zeros(6, dtype=np.float32)
    ls = landmark_summary(lm_arr)
    return (bs, hp, eg, ls)


def main():
    open(LOG, "w").close()
    rows = []
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "exists") and os.path.exists(r["frame_path"]):
                rows.append(r)
    log(f"Extracting temporal face signals from {len(rows)} clips × 3 frames")

    opts = vision.FaceLandmarkerOptions(
        base_options=mp_py.BaseOptions(model_asset_path=MODEL),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(opts)

    N = len(rows)
    bs_arr = np.zeros((N, 3, 52), dtype=np.float32)
    hp_arr = np.zeros((N, 3, 3), dtype=np.float32)
    eg_arr = np.zeros((N, 3, 6), dtype=np.float32)
    ls_arr = np.zeros((N, 3, 12), dtype=np.float32)
    det_arr = np.zeros((N, 3), dtype=np.int8)

    t0 = time.time()
    for i, r in enumerate(rows):
        base = r["clip_id"].replace(".avi", "").replace(".mp4", "")
        # Build 3 paths: t=2, t=5, t=8
        paths = [
            os.path.join(MULTI_DIR, r["split"], f"{base}_t2.jpg"),
            r["frame_path"],  # t=5 = frames_full
            os.path.join(MULTI_DIR, r["split"], f"{base}_t8.jpg"),
        ]
        for fi, p in enumerate(paths):
            if not os.path.exists(p):
                continue
            res = extract_one(detector, p)
            if res is None:
                continue
            bs_arr[i, fi] = res[0]; hp_arr[i, fi] = res[1]
            eg_arr[i, fi] = res[2]; ls_arr[i, fi] = res[3]
            det_arr[i, fi] = 1
        if (i + 1) % 500 == 0:
            dt = time.time() - t0
            log(f"  {i+1}/{N}  elapsed={dt:.0f}s  rate={(i+1)*3/dt:.1f} fps  "
                f"all-3-detected={int((det_arr[:i+1].sum(axis=1) == 3).sum())}/{i+1}")

    # Temporal features
    bs_mean = bs_arr.mean(axis=1); bs_std = bs_arr.std(axis=1)
    bs_delta = (np.abs(bs_arr[:, 1] - bs_arr[:, 0]) + np.abs(bs_arr[:, 2] - bs_arr[:, 1])) / 2
    hp_mean = hp_arr.mean(axis=1); hp_std = hp_arr.std(axis=1)
    eg_mean = eg_arr.mean(axis=1); eg_std = eg_arr.std(axis=1)
    ls_mean = ls_arr.mean(axis=1)
    det_count = det_arr.sum(axis=1)

    np.savez_compressed(
        OUT,
        clip_id=np.array([r["clip_id"] for r in rows]),
        split=np.array([r["split"] for r in rows]),
        subject_id=np.array([r["subject_id"] for r in rows]),
        engagement=np.array([int(r["engagement"]) for r in rows], dtype=np.int64),
        blendshapes_mean=bs_mean, blendshapes_std=bs_std, blendshapes_delta=bs_delta,
        head_pose_mean=hp_mean, head_pose_std=hp_std,
        eye_gaze_mean=eg_mean, eye_gaze_std=eg_std,
        landmark_summary_mean=ls_mean,
        detected_count=det_count,
    )
    log(f"Saved: {OUT}")
    log(f"Detection: median={int(np.median(det_count))}/3, all3={int((det_count==3).sum())}/{N} ({100*(det_count==3).sum()/N:.1f}%)")
    log("TEMPORAL_FACE_SIGNALS_DONE")


if __name__ == "__main__":
    main()
