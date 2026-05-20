"""
Encode SigLIP-L on t=2 and t=8 frames (frames_full_multi/<split>/<clip>_t{2,8}.jpg).
Combine with existing t=5 features (features/daisee_siglip_l_features.npz) into a
single (N, 3, 1024) array.

Output:
  features/daisee_siglip_l_multiframe_features.npz
    clip_id (N,)
    split (N,)
    subject_id (N,)
    engagement (N,)
    feat (N, 3, 1024)  # t=2, t=5, t=8
    success (N,)        # 1 if all 3 frames encoded
"""
import os, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, SiglipImageProcessor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T5 = os.path.join(BASE, "features", "daisee_siglip_l_features.npz")
FRAMES_MULTI = os.path.join(BASE, "frames_full_multi")
OUT = os.path.join(BASE, "features", "daisee_siglip_l_multiframe_features.npz")
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))


def main():
    print(f"Device: {DEVICE}", flush=True)
    print("Loading t=5 features (existing)...", flush=True)
    d = np.load(T5, allow_pickle=True)
    clip_ids = d['clip_id']; splits = d['split']
    subjects = d['subject_id']; engagement = d['engagement']
    feat_t5 = d['feat']    # (N, 1024)
    N = len(clip_ids)
    print(f"N = {N}, feat_t5 shape = {feat_t5.shape}", flush=True)

    print("Loading SigLIP-L...", flush=True)
    name = "google/siglip-large-patch16-256"
    proc = SiglipImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).vision_model.to(DEVICE).eval()

    feat_t2 = np.zeros((N, 1024), dtype=np.float32)
    feat_t8 = np.zeros((N, 1024), dtype=np.float32)
    succ_t2 = np.zeros(N, dtype=np.int8)
    succ_t8 = np.zeros(N, dtype=np.int8)

    t0 = time.time()
    batch = 16
    pending = {2: [], 8: []}   # list of (idx, img)
    def flush(tag, lst):
        if not lst: return
        imgs = [it[1] for it in lst]
        idxs = [it[0] for it in lst]
        with torch.no_grad():
            inp = proc(images=imgs, return_tensors='pt')
            pv = inp['pixel_values'].to(DEVICE)
            out = model(pixel_values=pv).pooler_output.cpu().numpy()
        for j, idx in enumerate(idxs):
            if tag == 2:
                feat_t2[idx] = out[j]; succ_t2[idx] = 1
            else:
                feat_t8[idx] = out[j]; succ_t8[idx] = 1

    for i in range(N):
        clip = clip_ids[i].replace('.avi', '').replace('.mp4', '')
        split = splits[i]
        for tag in (2, 8):
            p = os.path.join(FRAMES_MULTI, split, f"{clip}_t{tag}.jpg")
            if os.path.exists(p):
                try:
                    img = Image.open(p).convert("RGB")
                    pending[tag].append((i, img))
                    if len(pending[tag]) >= batch:
                        flush(tag, pending[tag]); pending[tag] = []
                except Exception as e:
                    print(f"  ERR loading {p}: {e}", flush=True)
        if (i + 1) % 200 == 0:
            dt = time.time() - t0; rate = (i+1)/dt
            print(f"  {i+1}/{N}  rate={rate:.2f} clip/s  t2_succ={int(succ_t2.sum())} t8_succ={int(succ_t8.sum())}", flush=True)
    flush(2, pending[2]); flush(8, pending[8])

    feat = np.stack([feat_t2, feat_t5, feat_t8], axis=1)  # (N, 3, 1024)
    success = ((succ_t2 == 1) & (succ_t8 == 1)).astype(np.int8)
    print(f"\nSuccess (all 3 frames): {int(success.sum())}/{N}", flush=True)

    np.savez_compressed(
        OUT,
        clip_id=clip_ids, split=splits, subject_id=subjects,
        engagement=engagement, feat=feat, success=success,
    )
    print(f"Saved: {OUT}  shape={feat.shape}", flush=True)


if __name__ == "__main__":
    main()
