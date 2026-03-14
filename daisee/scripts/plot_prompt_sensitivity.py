"""
Generate Figure 1: Accuracy and Cohen's kappa by prompt variant on DAiSEE.
Includes Qwen2.5-VL-7B-Instruct alongside existing models.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# Data — (Accuracy %, kappa) per model per prompt
# ---------------------------------------------------------------------------
data = {
    "CLIP":            {"P1": (37.0, 0.036), "P2": (31.3, 0.068), "P3": (35.0, -0.042)},
    "BLIP-VQA":        {"P1": (25.3,-0.018), "P2": (32.7, 0.058), "P3": (35.3,  0.000)},
    "GPT-4o":          {"P1": (27.7, 0.067), "P2": (32.3, 0.075), "P3": (36.0,  0.039)},
    "LLaVA":           {"P1": ( 5.3, 0.037), "P2": (15.7, 0.045), "P3": (37.7,  0.008)},
    "Qwen2.5-VL-7B":   {"P1": (30.0, 0.098), "P2": (30.0, 0.097), "P3": (35.3, -0.041)},
}

prompts = ["P1", "P2", "P3"]
models  = list(data.keys())

# Match existing figure colours; add orange for Qwen
colors = {
    "CLIP":          "#4C72B0",
    "BLIP-VQA":      "#55A868",
    "GPT-4o":        "#C44E52",
    "LLaVA":         "#8172B2",
    "Qwen2.5-VL-7B": "#DD8452",
}

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
n_models  = len(models)
n_prompts = len(prompts)
bar_width = 0.13
group_gap = 0.9          # distance between prompt groups

x = np.arange(n_prompts) * group_gap

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
fig.subplots_adjust(bottom=0.18, wspace=0.32)

# centre offsets for each model within a group
offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_width

# ---------------------------------------------------------------------------
# (a) Accuracy
# ---------------------------------------------------------------------------
for i, model in enumerate(models):
    accs = [data[model][p][0] for p in prompts]
    ax1.bar(x + offsets[i], accs, width=bar_width,
            color=colors[model], label=model, alpha=0.92, zorder=3)

ax1.axhline(25, color="gray", linewidth=1, linestyle="--", zorder=2, label="_nolegend_")
ax1.text(x[-1] + offsets[-1] + bar_width * 1.1, 25.6, "Chance",
         color="gray", fontsize=8, va="bottom")

ax1.set_xticks(x)
ax1.set_xticklabels(prompts, fontsize=10)
ax1.set_ylabel("Accuracy", fontsize=10)
ax1.set_title("(a) Accuracy by Prompt — DAiSEE", fontsize=10)
ax1.set_ylim(0, 55)
ax1.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax1.set_axisbelow(True)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ---------------------------------------------------------------------------
# (b) Kappa
# ---------------------------------------------------------------------------
for i, model in enumerate(models):
    kappas = [data[model][p][1] for p in prompts]
    ax2.bar(x + offsets[i], kappas, width=bar_width,
            color=colors[model], label=model, alpha=0.92, zorder=3)

ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=2)

ax2.set_xticks(x)
ax2.set_xticklabels(prompts, fontsize=10)
ax2.set_ylabel("Cohen's $\\kappa$ (quadratic)", fontsize=10)
ax2.set_title("(b) Cohen's $\\kappa$ by Prompt — DAiSEE", fontsize=10)
ax2.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax2.set_axisbelow(True)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# ---------------------------------------------------------------------------
# Shared legend at bottom
# ---------------------------------------------------------------------------
handles = [plt.Rectangle((0,0),1,1, color=colors[m], alpha=0.92) for m in models]
fig.legend(handles, models, loc="lower center", ncol=n_models,
           fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, 0.01))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = "/home/ubuntu/Benchmarking-VLMs-for-Classroom-Observability/daisee/figures"
os.makedirs(out_dir, exist_ok=True)

pdf_out = os.path.join(out_dir, "fig1_daisee_prompt_sensitivity.pdf")
png_out = os.path.join(out_dir, "fig1_daisee_prompt_sensitivity.png")

fig.savefig(pdf_out, bbox_inches="tight", dpi=200)
fig.savefig(png_out, bbox_inches="tight", dpi=150)
print(f"Saved: {pdf_out}")
print(f"Saved: {png_out}")
