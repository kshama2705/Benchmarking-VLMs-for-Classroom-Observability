"""
Ablation study figures for CVPR 2026 workshop paper.

Produces two figures:
  fig5_tsne_features.pdf      -- t-SNE of CLIP image embeddings colored by engagement label
  fig6_similarity_dists.pdf   -- Text-image cosine similarity distributions across prompt variants

Usage:
  cd /Users/amangoyal/Documents/CVPR\ 2026\ Workshop
  source venv/bin/activate
  python scripts/gen_ablations.py
"""

import os
import csv
import numpy as np
import torch
import open_clip
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLED_CSV = os.path.join(BASE_DIR, "sampled_test.csv")
FRAMES_DIR  = os.path.join(BASE_DIR, "sampled_frames")
OUT_DIR     = os.path.join(BASE_DIR, "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Visual style (matches gen_figures.py) ─────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
})

LEVEL_COLORS  = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']   # L0→L3
LEVEL_LABELS  = ['L0: Not Engaged', 'L1: Barely Engaged',
                 'L2: Engaged',     'L3: Highly Engaged']
PROMPT_NAMES  = ['P1 (Minimal)', 'P2 (Behavioral)', 'P3 (Emotional)']

# ── Prompt text templates (identical to 03_clip_inference.py) ─────────────────
PROMPT_VARIANTS = {
    1: {
        0: "a photo of a student who is not engaged at all, completely distracted",
        1: "a photo of a student who is barely engaged, passively present",
        2: "a photo of a student who is engaged and attentive",
        3: "a photo of a student who is highly engaged, very focused and alert",
    },
    2: {
        0: "a student looking away from the screen, yawning or distracted, showing no interest",
        1: "a student passively sitting, occasionally glancing at the screen, low attention",
        2: "a student watching the screen attentively, maintaining eye contact with the content",
        3: "a student leaning forward, deeply focused, actively concentrating on the content",
    },
    3: {
        0: "a bored and disinterested student with a blank or sleepy expression",
        1: "a slightly disengaged student with a neutral, unfocused expression",
        2: "an attentive student with an interested and alert facial expression",
        3: "a highly focused student showing curiosity and active concentration",
    },
}


# ── Load samples ──────────────────────────────────────────────────────────────
def load_samples():
    samples = []
    with open(SAMPLED_CSV) as f:
        for row in csv.DictReader(f):
            frame_path = os.path.join(FRAMES_DIR, os.path.basename(row["frame_path"]))
            if os.path.exists(frame_path):
                samples.append({
                    "clip_id":    row["clip_id"],
                    "engagement": int(row["engagement"]),
                    "frame_path": frame_path,
                })
    print(f"Loaded {len(samples)} samples")
    return samples


# ── Extract CLIP embeddings ───────────────────────────────────────────────────
def extract_embeddings(samples, model, preprocess, device):
    """Returns (N, D) float32 numpy array of L2-normalised image embeddings."""
    feats = []
    model.eval()
    with torch.no_grad():
        for i, s in enumerate(samples):
            img = preprocess(Image.open(s["frame_path"]).convert("RGB")).unsqueeze(0).to(device)
            f = model.encode_image(img)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().float().numpy())
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(samples)} frames encoded")
    return np.vstack(feats)   # (N, 512)


# ── Encode text templates ─────────────────────────────────────────────────────
def encode_texts(model, tokenizer, device):
    """Returns dict {prompt_id: (4, D) tensor of L2-normalised text embeddings}."""
    text_feats = {}
    with torch.no_grad():
        for pid, templates in PROMPT_VARIANTS.items():
            texts  = [templates[lvl] for lvl in range(4)]
            tokens = tokenizer(texts).to(device)
            tf = model.encode_text(tokens)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            text_feats[pid] = tf.cpu().float().numpy()   # (4, 512)
    return text_feats


# ── Figure 1: t-SNE ───────────────────────────────────────────────────────────
def plot_tsne(image_feats, labels, out_path):
    print("Running t-SNE…")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                random_state=42, init='pca', learning_rate='auto')
    emb = tsne.fit_transform(image_feats)

    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    for lvl in range(4):
        mask = np.array(labels) == lvl
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=LEVEL_COLORS[lvl], label=LEVEL_LABELS[lvl],
                   s=18, alpha=0.75, linewidths=0)

    ax.set_title("CLIP Feature Space (t-SNE) — DAiSEE Frames", pad=6)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper right', markerscale=1.4, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 2: similarity distributions ───────────────────────────────────────
def plot_similarity_dists(image_feats, text_feats, labels, out_path):
    """
    3-column figure (one per prompt).
    Each subplot: x = true engagement class, y = cosine similarity.
    4 groups of box plots (one per text template L0-L3), color-coded.
    A well-calibrated model would show highest similarity on the diagonal
    (true class == text class).
    """
    labels = np.array(labels)
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.0), sharey=False)

    for col, (pid, pname) in enumerate(zip([1, 2, 3], PROMPT_NAMES)):
        ax = axes[col]
        tf = text_feats[pid]               # (4, D)
        sims = image_feats @ tf.T          # (N, 4): row i = sims to [L0,L1,L2,L3]

        # For each (true_class, text_class) pair collect similarity values
        n_true, n_text = 4, 4
        x_positions = []
        box_data    = []
        box_colors  = []
        text_class_lines = []   # track x positions per text class for legend

        group_width = 0.8
        gap = 0.5
        bar_w = group_width / n_text

        for true_cls in range(n_true):
            mask = labels == true_cls
            if mask.sum() == 0:
                continue
            grp_center = true_cls * (group_width + gap)
            for text_cls in range(n_text):
                xpos = grp_center + (text_cls - 1.5) * bar_w
                x_positions.append(xpos)
                box_data.append(sims[mask, text_cls])
                box_colors.append(LEVEL_COLORS[text_cls])
                text_class_lines.append((xpos, text_cls))

        bp = ax.boxplot(box_data, positions=x_positions,
                        widths=bar_w * 0.85, patch_artist=True,
                        showfliers=False,
                        medianprops=dict(color='black', linewidth=1.2),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        boxprops=dict(linewidth=0.6))
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        # x-axis: group labels = true class
        grp_centers = [tc * (group_width + gap) + group_width / 2 - bar_w / 2
                       for tc in range(n_true)]
        ax.set_xticks(grp_centers)
        ax.set_xticklabels([f'L{tc}' for tc in range(n_true)])
        ax.set_xlabel("True Engagement Class")
        if col == 0:
            ax.set_ylabel("Cosine Similarity")
        ax.set_title(pname)

        # Diagonal reference: highlight when text_cls == true_cls
        for true_cls in range(n_true):
            mask_count = (labels == true_cls).sum()
            if mask_count == 0:
                continue
            grp_center = true_cls * (group_width + gap)
            diag_x = grp_center + (true_cls - 1.5) * bar_w
            ax.axvspan(diag_x - bar_w * 0.5, diag_x + bar_w * 0.5,
                       alpha=0.08, color='black', zorder=0)

    # Shared legend: text template classes
    patches = [plt.Rectangle((0,0),1,1, fc=LEVEL_COLORS[i], alpha=0.65,
                              label=f'Text: L{i}') for i in range(4)]
    fig.legend(handles=patches, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.05), framealpha=0.9)

    fig.suptitle(
        "Text-Image Cosine Similarity by True Class (shaded box = correct text template)",
        fontsize=8, y=1.01
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = ("mps"  if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available()         else "cpu")
    print(f"Device: {device}")

    print("Loading CLIP ViT-B/32 (laion2b_s34b_b79k)…")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device)

    samples = load_samples()
    labels  = [s["engagement"] for s in samples]

    print("\nExtracting image embeddings…")
    image_feats = extract_embeddings(samples, model, preprocess, device)
    print(f"Image features: {image_feats.shape}")

    print("\nEncoding text templates…")
    text_feats = encode_texts(model, tokenizer, device)

    print("\n── Figure 1: t-SNE ──────────────────────────────────")
    plot_tsne(image_feats, labels,
              os.path.join(OUT_DIR, "fig5_tsne_features.pdf"))

    print("\n── Figure 2: Similarity distributions ──────────────")
    plot_similarity_dists(image_feats, text_feats, labels,
                          os.path.join(OUT_DIR, "fig6_similarity_dists.pdf"))

    # ── Save raw similarity scores to CSV ─────────────────────────────────────
    import pandas as pd
    rows = []
    for i, s in enumerate(samples):
        row = {"clip_id": s["clip_id"], "true_class": labels[i]}
        for pid in [1, 2, 3]:
            tf = text_feats[pid]
            sims = image_feats[i] @ tf.T   # (4,)
            for lvl in range(4):
                row[f"p{pid}_sim_l{lvl}"] = float(sims[lvl])
            row[f"p{pid}_argmax"] = int(np.argmax(sims))
        rows.append(row)
    df = pd.DataFrame(rows)
    sim_csv = os.path.join(BASE_DIR, "results", "ablation_similarities.csv")
    df.to_csv(sim_csv, index=False)
    print(f"Saved: {sim_csv}")

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n── Mean cosine similarity (true class × text template) ──────────────")
    labels_arr = np.array(labels)
    for pid in [1, 2, 3]:
        tf = text_feats[pid]
        sims = image_feats @ tf.T   # (300, 4)
        print(f"\nPrompt {pid}:")
        header = "True\\Text  " + "  ".join(f"  L{j}" for j in range(4))
        print(header)
        for true_cls in range(4):
            mask = labels_arr == true_cls
            if mask.sum() == 0:
                continue
            means = sims[mask].mean(axis=0)
            row_str = f"  L{true_cls} (n={mask.sum():3d})  " + "  ".join(f"{m:.4f}" for m in means)
            print(row_str)

    print("\nDone. Figures written to paper/figures/")


if __name__ == "__main__":
    main()
