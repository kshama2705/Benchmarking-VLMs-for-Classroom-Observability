"""
Reproduce Figure 4: Normalised confusion matrices for best prompt per model on DAiSEE.
Adds Qwen2.5-VL-7B-Instruct panel alongside existing models.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Confusion matrices — raw counts (rows = true, cols = predicted)
# Read from original figure for existing models; computed from CSV for Qwen
# ---------------------------------------------------------------------------
models = {
    "CLIP\nAcc=37.0% $\\kappa$=0.04": np.array([
        [0,  2,  2,  0],
        [1, 25, 56,  2],
        [0, 23, 82,  1],
        [3, 25, 74,  4],
    ]),
    "BLIP-VQA\nAcc=35.3% $\\kappa$=0.00": np.array([
        [0,  0,  4,  0],
        [0,  0, 84,  0],
        [0,  0,106,  0],
        [0,  0,106,  0],
    ]),
    "GPT-4o\nAcc=36.0% $\\kappa$=0.04": np.array([
        [0,  0,  4,  0],
        [0,  4, 80,  0],
        [1,  0,104,  1],
        [0,  0,106,  0],
    ]),
    "LLaVA\nAcc=39.0% $\\kappa$=0.10": np.array([
        [0,  1,  3,  0],
        [0, 17, 64,  3],
        [0,  5,100,  1],
        [0,  5,101,  0],
    ]),
    "Qwen2.5-VL-7B\nAcc=30.0% $\\kappa$=0.10": np.array([
        [ 1,  3,  0,  0],
        [ 2, 61, 20,  1],
        [ 0, 79, 27,  0],
        [ 0, 63, 42,  1],
    ]),
}

# Normalise each row by its sum (true-class total)
def row_normalise(cm):
    row_sums = cm.sum(axis=1, keepdims=True)
    return cm / np.where(row_sums == 0, 1, row_sums)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
n = len(models)
fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6))
fig.suptitle("Normalized Confusion Matrices — Best Prompt per Model (DAiSEE)",
             fontsize=11, y=1.01)

# Light blue colormap matching the original figure
cmap = LinearSegmentedColormap.from_list(
    "lightblue_white", ["#ffffff", "#2171b5"], N=256
)

for ax, (title, cm) in zip(axes, models.items()):
    norm_cm = row_normalise(cm)

    im = ax.imshow(norm_cm, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    # Annotate with raw counts
    for i in range(4):
        for j in range(4):
            val = cm[i, j]
            brightness = norm_cm[i, j]
            color = "white" if brightness > 0.55 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=9, color=color, fontweight="normal")

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(range(4), fontsize=8)
    ax.set_yticklabels(range(4), fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_title(title, fontsize=8.5, pad=6)

axes[0].set_ylabel("True", fontsize=8)

fig.tight_layout()

out = "/home/ubuntu/Benchmarking-VLMs-for-Classroom-Observability/daisee/figures/fig2_confusion_matrices.pdf"
import os; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight", dpi=200)

# Also save PNG for quick preview
png_out = out.replace(".pdf", ".png")
fig.savefig(png_out, bbox_inches="tight", dpi=150)
print(f"Saved: {out}")
print(f"Saved: {png_out}")
