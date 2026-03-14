"""
Generate Figure 3: Best Accuracy and Best kappa — DAiSEE vs SCB cross-dataset comparison.
Includes Qwen2.5-VL-7B-Instruct.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# Best results per model (max over all prompts)
# ---------------------------------------------------------------------------
#                          DAiSEE                SCB
#                     best_acc  best_k      best_acc  best_k
models_data = {
    "CLIP":           (37.0, 0.036,   67.3, 0.599),
    "BLIP-VQA":       (35.3, 0.000,   33.6, 0.000),
    "GPT-4o":         (36.0, 0.039,   67.9, 0.582),
    "LLaVA":          (39.0, 0.101,   49.2, 0.224),
    "Qwen2.5-VL-7B":  (35.3, 0.098,   64.9, 0.508),
}

models   = list(models_data.keys())
x        = np.arange(len(models))
bar_w    = 0.35

daisee_acc = [models_data[m][0] for m in models]
daisee_k   = [models_data[m][1] for m in models]
scb_acc    = [models_data[m][2] for m in models]
scb_k      = [models_data[m][3] for m in models]

blue = "#5B8DB8"   # DAiSEE
red  = "#C97A72"   # SCB

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
fig.subplots_adjust(bottom=0.18, wspace=0.32)

# ---------------------------------------------------------------------------
# (a) Accuracy
# ---------------------------------------------------------------------------
ax1.bar(x - bar_w/2, daisee_acc, width=bar_w, color=blue, label="DAiSEE (individual)", alpha=0.92, zorder=3)
ax1.bar(x + bar_w/2, scb_acc,    width=bar_w, color=red,  label="SCB (scene-level)",   alpha=0.92, zorder=3)

ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=9.5)
ax1.set_ylabel("Accuracy", fontsize=10)
ax1.set_title("(a) Best Accuracy: DAiSEE vs. SCB", fontsize=10)
ax1.set_ylim(0, 85)
ax1.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax1.set_axisbelow(True)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ---------------------------------------------------------------------------
# (b) Kappa
# ---------------------------------------------------------------------------
ax2.bar(x - bar_w/2, daisee_k, width=bar_w, color=blue, alpha=0.92, zorder=3)
ax2.bar(x + bar_w/2, scb_k,    width=bar_w, color=red,  alpha=0.92, zorder=3)

ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=2)

ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=9.5)
ax2.set_ylabel("Cohen's $\\kappa$", fontsize=10)
ax2.set_title("(b) Best $\\kappa$: DAiSEE vs. SCB", fontsize=10)
ax2.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax2.set_axisbelow(True)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# ---------------------------------------------------------------------------
# Shared legend
# ---------------------------------------------------------------------------
handles = [
    plt.Rectangle((0,0),1,1, color=blue, alpha=0.92),
    plt.Rectangle((0,0),1,1, color=red,  alpha=0.92),
]
fig.legend(handles, ["DAiSEE (individual)", "SCB (scene-level)"],
           loc="lower center", ncol=2, fontsize=10, frameon=False,
           bbox_to_anchor=(0.5, 0.01))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = "/home/ubuntu/Benchmarking-VLMs-for-Classroom-Observability/daisee/figures"
os.makedirs(out_dir, exist_ok=True)

for ext in ("pdf", "png"):
    out = os.path.join(out_dir, f"fig3_cross_dataset.{ext}")
    fig.savefig(out, bbox_inches="tight", dpi=150 if ext == "png" else 200)
    print(f"Saved: {out}")
