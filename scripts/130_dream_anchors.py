"""
DREAM step 1 — per-subject neutral-anchor extraction.

For each subject in DAiSEE (train + val + test), score every frame by a
neutral-face proxy:
    neutral(x) = ||blendshapes(x)||_1  +  lambda * ||gaze(x) - gaze_forward||_2

Lower score => more neutral face. Pick the K frames with lowest score per
subject; that subject's anchor set is those frames.

Notes:
- We never look at engagement labels when selecting anchors.
- We report (for transparency) the distribution of engagement labels among
  the selected anchors, but only to characterize the criterion.
- Anchors for TEST subjects come from their OWN clips' blendshapes (NOT
  their engagement labels). This is the leave-labels-out protocol (P-train).
- We also emit the (P-zero) variant: anchors = first 10% of frames per
  subject in temporal-order (independent of blendshapes).

Outputs:
- features/dream_anchors.npz with arrays:
    subject_ids: (N_subjects,) unique subject IDs
    p_train_anchor_idx_k{K}: (N_subjects, K) clip indices into face_signals
    p_zero_anchor_idx_k{K}: (N_subjects, K) clip indices (temporal first-10%)
    neutral_score: (8571,) the score for every clip (debug)
- results/dream/anchor_stats.json: anchor engagement-label distribution
"""

import os, json, sys
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE = os.path.join(BASE, "features", "daisee_face_signals.npz")
OUT_NPZ = os.path.join(BASE, "features", "dream_anchors.npz")
OUT_JSON = os.path.join(BASE, "results", "dream", "anchor_stats.json")
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

LAMBDA_GAZE = 0.5            # weight on gaze deviation
K_LIST = [1, 3, 5, 10, 20]   # anchor budgets to precompute


def compute_neutral_score(blend, gaze):
    """Lower = more neutral face."""
    bs_norm = np.abs(blend).sum(axis=1)       # L1 of 52 blendshapes
    # gaze: 6-d (yaw_l, pitch_l, openness_l, yaw_r, pitch_r, openness_r)
    # forward gaze: yaw=0, pitch=0, openness=~1
    forward = np.array([0, 0, 1, 0, 0, 1], dtype=np.float32)
    gaze_dev = np.linalg.norm(gaze - forward[None], axis=1)
    return bs_norm + LAMBDA_GAZE * gaze_dev


def main():
    d = np.load(FACE, allow_pickle=True)
    clip_id = d['clip_id']
    split = d['split']
    subject = d['subject_id']
    engagement = d['engagement']
    blend = d['blendshapes']        # (8571, 52)
    gaze = d['eye_gaze']            # (8571, 6)
    detected = d['detected']        # (8571,)

    print(f"Total clips: {len(clip_id)}  Detected: {detected.sum()}", flush=True)
    print(f"Splits: {dict(zip(*np.unique(split, return_counts=True)))}", flush=True)
    print(f"Subjects total: {len(np.unique(subject))}", flush=True)

    score = compute_neutral_score(blend, gaze)
    # Force non-detected frames to +inf so they cannot be selected as anchors
    score = np.where(detected.astype(bool), score, np.inf).astype(np.float32)
    print(f"Neutral-score stats: min={np.nanmin(score[np.isfinite(score)]):.3f}  "
          f"median={np.nanmedian(score[np.isfinite(score)]):.3f}  "
          f"max={np.nanmax(score[np.isfinite(score)]):.3f}", flush=True)

    out = {"neutral_score": score}
    stats = {"lambda_gaze": LAMBDA_GAZE, "k_list": K_LIST,
             "p_train": {}, "p_zero": {}}

    subjects = np.unique(subject)
    out["subject_ids"] = subjects
    print(f"\n=== Anchor selection per subject ===", flush=True)
    for K in K_LIST:
        p_train_idx = np.full((len(subjects), K), -1, dtype=np.int32)
        p_zero_idx = np.full((len(subjects), K), -1, dtype=np.int32)

        eng_at_train = []
        eng_at_zero = []
        for si, s in enumerate(subjects):
            mask = (subject == s)
            global_idx = np.where(mask)[0]
            if len(global_idx) == 0:
                continue
            s_score = score[global_idx]
            # P-train: pick K with lowest neutral score within this subject
            finite_mask = np.isfinite(s_score)
            usable = global_idx[finite_mask]
            usable_score = s_score[finite_mask]
            if len(usable) == 0:
                continue
            order = np.argsort(usable_score, kind='stable')
            k_take = min(K, len(usable))
            picks = usable[order[:k_take]]
            p_train_idx[si, :k_take] = picks
            eng_at_train.extend(engagement[picks].tolist())

            # P-zero: pick first K in temporal order (clip_id sorted lex)
            # clip_id format e.g. '1100011001' — within a subject, clip suffix encodes ordering
            cids = clip_id[global_idx]
            order_z = np.argsort(cids, kind='stable')
            usable_z = global_idx[order_z]
            # take first K of usable_z
            picks_z = usable_z[:K]
            # filter to detected frames; if too few, pad with finite
            ok = np.array([detected[i] for i in picks_z], dtype=bool)
            if ok.sum() < K:
                # fall back: take first K detected in temporal order
                det_in_order = usable_z[detected[usable_z].astype(bool)]
                picks_z = det_in_order[:K]
            k_take_z = min(K, len(picks_z))
            p_zero_idx[si, :k_take_z] = picks_z[:k_take_z]
            eng_at_zero.extend(engagement[picks_z[:k_take_z]].tolist())

        out[f"p_train_anchor_idx_k{K}"] = p_train_idx
        out[f"p_zero_anchor_idx_k{K}"] = p_zero_idx

        from collections import Counter
        ct_train = Counter(eng_at_train)
        ct_zero = Counter(eng_at_zero)
        ct_all = Counter(engagement.tolist())
        stats["p_train"][str(K)] = {str(k): int(ct_train.get(k, 0)) for k in range(4)}
        stats["p_zero"][str(K)] = {str(k): int(ct_zero.get(k, 0)) for k in range(4)}
        stats["all_distribution"] = {str(k): int(ct_all.get(k, 0)) for k in range(4)}

        print(f"\nK={K}:", flush=True)
        print(f"  P-train anchor engagement dist: {dict(ct_train)}", flush=True)
        print(f"  P-zero  anchor engagement dist: {dict(ct_zero)}", flush=True)

    np.savez_compressed(OUT_NPZ, **out)
    print(f"\nSaved anchors: {OUT_NPZ}", flush=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats:   {OUT_JSON}", flush=True)
    print(f"Overall engagement dist (all 8571 clips): {stats['all_distribution']}", flush=True)


if __name__ == "__main__":
    main()
