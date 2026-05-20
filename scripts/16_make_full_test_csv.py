"""Build full_test.csv from frames_full/manifest.csv (Test split only),
matching the schema the existing inference scripts expect."""

import os, csv

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(BASE, "frames_full", "manifest.csv")
OUT = os.path.join(BASE, "full_test.csv")

rows = []
with open(M) as f:
    r = csv.DictReader(f)
    for row in r:
        if row["split"] != "Test":
            continue
        if row["status"] not in ("ok", "exists"):
            continue
        rows.append({
            "clip_id": row["clip_id"],
            "engagement": row["engagement"],
            "frame_path": row["frame_path"],
        })

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["clip_id", "engagement", "frame_path"])
    w.writeheader()
    w.writerows(rows)
print(f"Wrote {len(rows)} rows to {OUT}")
