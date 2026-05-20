"""
Detect a face bounding box in every DAiSEE frame using mediapipe FaceDetection.
Stores normalized (x_min, y_min, width, height) in [0,1].

Output:
  features/face_bboxes.json
    {clip_id: {"bbox": [x, y, w, h], "score": float} or {"bbox": None}}
"""
import os, csv, json, time
import numpy as np
import cv2

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(BASE, "frames_full", "manifest.csv")
OUT = os.path.join(BASE, "features", "face_bboxes.json")

# OpenCV Haar cascade — no download needed, shipped with opencv-python
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(HAAR_PATH)


def detect(path):
    img = cv2.imread(path)
    if img is None:
        return None, None, (0, 0)
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Equalize hist for low-light webcam frames
    gray = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(int(0.10 * min(W, H)), int(0.10 * min(W, H))),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return None, None, (W, H)
    # Pick the largest face
    faces_arr = np.array(faces)
    areas = faces_arr[:, 2] * faces_arr[:, 3]
    best = faces_arr[areas.argmax()]
    x, y, w, h = best
    # Normalize to [0, 1]
    nx, ny, nw, nh = x / W, y / H, w / W, h / H
    return (float(nx), float(ny), float(nw), float(nh)), float(areas.max() / (W * H)), (W, H)


def main():
    rows = []
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "exists") and os.path.exists(r["frame_path"]):
                rows.append(r)
    print(f"Processing {len(rows)} frames...")

    out = {}
    detected = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        bbox, score, size = detect(r["frame_path"])
        out[r["clip_id"]] = {"bbox": list(bbox) if bbox else None,
                             "score": score, "image_size": list(size)}
        if bbox is not None:
            detected += 1
        if (i + 1) % 500 == 0:
            dt = time.time() - t0
            print(f"  {i+1}/{len(rows)}  detected={detected} ({100*detected/(i+1):.1f}%)  "
                  f"elapsed={dt:.0f}s  est_remaining={dt/(i+1)*(len(rows)-i-1):.0f}s",
                  flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"Done. Detected face in {detected}/{len(rows)} ({100*detected/len(rows):.1f}%)")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
