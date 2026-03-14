"""
Regenerate fig4_prediction_heatmap.pdf with Qwen2.5-VL rows added.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Raw prediction counts (rows = model-prompt, cols = L0/L1/L2/L3) ──────────
# Each row must sum to 300.
row_labels = [
    "CLIP P1", "CLIP P2", "CLIP P3",
    "BLIP P1", "BLIP P2", "BLIP P3",
    "GPT-4o P1", "GPT-4o P2", "GPT-4o P3",
    "LLaVA P1", "LLaVA P2", "LLaVA P3",
    "Qwen P1", "Qwen P2", "Qwen P3",
    "Ground Truth",
]

counts = np.array([
    [4,   75,  214,   7],   # CLIP P1
    [1,  216,   83,   0],   # CLIP P2
    [0,  117,  167,  16],   # CLIP P3
    [55, 245,    0,   0],   # BLIP P1
    [16,  24,  260,   0],   # BLIP P2
    [0,    0,  300,   0],   # BLIP P3
    [12, 256,   32,   0],   # GPT-4o P1
    [10, 246,   44,   0],   # GPT-4o P2
    [1,    4,  294,   1],   # GPT-4o P3
    [254,  0,   46,   0],   # LLaVA P1
    [116, 182,   1,   1],   # LLaVA P2
    [1,   28,  268,   4],   # LLaVA P3
    [3,  206,   89,   2],   # Qwen P1
    [3,  191,  101,   5],   # Qwen P2
    [0,    0,  293,   7],   # Qwen P3
    [4,   84,  106, 106],   # Ground Truth
], dtype=float)

fractions = counts / 300.0

# ── Plot ──────────────────────────────────────────────────────────────────────
n_rows, n_cols = fractions.shape
fig, ax = plt.subplots(figsize=(7.5, 8.5))

cmap = plt.get_cmap("YlOrRd")
im = ax.imshow(fractions, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

# Colour bar
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Fraction of predictions", fontsize=10)
cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Annotate cells — omit zeros, white text on dark cells
for r in range(n_rows):
    for c in range(n_cols):
        val = int(counts[r, c])
        if val == 0:
            continue
        frac = fractions[r, c]
        text_color = "white" if frac > 0.55 else "black"
        ax.text(c, r, str(val), ha="center", va="center",
                fontsize=8.5, color=text_color, fontweight="normal")

# Axes
ax.set_xticks(range(n_cols))
ax.set_xticklabels(["L0\nNot Eng.", "L1\nBarely", "L2\nEngaged", "L3\nHighly"], fontsize=10)
ax.set_yticks(range(n_rows))
ax.set_yticklabels(row_labels, fontsize=9.5)
ax.set_xlabel("Predicted Class", fontsize=11)

# Separator line before Ground Truth row
ax.axhline(n_rows - 1.5, color="black", linewidth=1.2, linestyle="--")

# Separator lines between model groups (after every 3 rows)
for sep in [2.5, 5.5, 8.5, 11.5, 14.5]:
    if sep < n_rows - 1:
        ax.axhline(sep, color="grey", linewidth=0.6, linestyle="-")

ax.set_title(
    "Prediction Distribution Heatmap\n"
    "(fraction of 300 samples predicted per class)",
    fontsize=11, pad=10
)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "fig4_prediction_heatmap.pdf")
fig.savefig(out_path, bbox_inches="tight", dpi=150)
print("Saved:", out_path)
