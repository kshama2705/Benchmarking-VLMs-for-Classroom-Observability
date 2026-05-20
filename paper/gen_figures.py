"""Generate all figures for the CVPR 2026 workshop paper."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, cohen_kappa_score, mean_squared_error

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, 'results')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 150,
})

COLORS = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7']
MODEL_COLORS = {'CLIP': '#4878CF', 'BLIP-VQA': '#6ACC65', 'GPT-4o': '#D65F5F', 'LLaVA': '#B47CC7'}
CLASS_NAMES = ['Not Eng.\n(0)', 'Barely\n(1)', 'Engaged\n(2)', 'Highly\n(3)']

# ─────────────────────────────────────────────────────────────
# Figure 1: Accuracy across all model×prompt for DAiSEE
# ─────────────────────────────────────────────────────────────
def load(fname):
    df = pd.read_csv(os.path.join(RESULTS, fname))
    y_t, y_p = df['ground_truth'].tolist(), df['predicted'].tolist()
    return {
        'acc': accuracy_score(y_t, y_p),
        'f1': f1_score(y_t, y_p, average='macro', zero_division=0),
        'kappa': cohen_kappa_score(y_t, y_p, weights='quadratic'),
        'mse': mean_squared_error(y_t, y_p),
        'y_true': y_t,
        'y_pred': y_p,
    }

# Best per model
best = {
    'CLIP':     load('clip_prompt1_run1.csv'),
    'BLIP-VQA': load('blip_vqa_prompt3_run1.csv'),
    'GPT-4o':   load('gpt4o_prompt3_run1.csv'),
    'LLaVA':    load('llava_prompt3_run3.csv'),
}

# Prompt-wise for all models
p_data = {
    'CLIP':     [load('clip_prompt1_run1.csv')['acc'],     load('clip_prompt2_run1.csv')['acc'],     load('clip_prompt3_run1.csv')['acc']],
    'BLIP-VQA': [load('blip_vqa_prompt1_run1.csv')['acc'], load('blip_vqa_prompt2_run1.csv')['acc'], load('blip_vqa_prompt3_run1.csv')['acc']],
    'GPT-4o':   [load('gpt4o_prompt1_run1.csv')['acc'],    load('gpt4o_prompt2_run1.csv')['acc'],    load('gpt4o_prompt3_run1.csv')['acc']],
    'LLaVA':    [load('llava_prompt1_run1.csv')['acc'],    load('llava_prompt2_run1.csv')['acc'],    load('llava_prompt3_run1.csv')['acc']],
}
k_data = {
    'CLIP':     [load('clip_prompt1_run1.csv')['kappa'],     load('clip_prompt2_run1.csv')['kappa'],     load('clip_prompt3_run1.csv')['kappa']],
    'BLIP-VQA': [load('blip_vqa_prompt1_run1.csv')['kappa'], load('blip_vqa_prompt2_run1.csv')['kappa'], load('blip_vqa_prompt3_run1.csv')['kappa']],
    'GPT-4o':   [load('gpt4o_prompt1_run1.csv')['kappa'],    load('gpt4o_prompt2_run1.csv')['kappa'],    load('gpt4o_prompt3_run1.csv')['kappa']],
    'LLaVA':    [load('llava_prompt1_run1.csv')['kappa'],    load('llava_prompt2_run1.csv')['kappa'],    load('llava_prompt3_run1.csv')['kappa']],
}

# ─── Fig 1: Accuracy & Kappa across prompts ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
models = ['CLIP', 'BLIP-VQA', 'GPT-4o', 'LLaVA']
prompts = ['P1', 'P2', 'P3']
x = np.arange(len(prompts))
width = 0.18

for ax, metric_data, ylabel, title in zip(
    axes,
    [p_data, k_data],
    ['Accuracy', "Cohen's $\\kappa$ (quadratic)"],
    ['(a) Accuracy by Prompt — DAiSEE', '(b) Cohen\'s $\\kappa$ by Prompt — DAiSEE'],
):
    for i, (model, color) in enumerate(MODEL_COLORS.items()):
        offset = (i - 1.5) * width
        vals = [v * (100 if ylabel == 'Accuracy' else 1) for v in metric_data[model]]
        bars = ax.bar(x + offset, vals, width, label=model, color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(prompts)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    if ylabel == 'Accuracy':
        ax.set_ylim(0, 55)
        ax.axhline(y=25, color='gray', linewidth=0.8, linestyle=':', alpha=0.6)
        ax.text(2.65, 26.5, 'Chance', fontsize=6, color='gray')
    else:
        ax.set_ylim(-0.15, 0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

handles = [mpatches.Patch(color=c, alpha=0.85, label=m) for m, c in MODEL_COLORS.items()]
fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(os.path.join(OUT, 'fig1_daisee_prompt_sensitivity.pdf'), bbox_inches='tight')
plt.close()
print("Fig 1 saved.")

# ─── Fig 2: Confusion matrices for best models ───────────────
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.0))
class_labels = ['0', '1', '2', '3']
for ax, (model, d) in zip(axes, best.items()):
    cm = confusion_matrix(d['y_true'], d['y_pred'], labels=[0,1,2,3])
    # Normalize by row
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks([0,1,2,3]); ax.set_yticks([0,1,2,3])
    ax.set_xticklabels(class_labels); ax.set_yticklabels(class_labels)
    ax.set_xlabel('Predicted', labelpad=2)
    if model == 'CLIP':
        ax.set_ylabel('True', labelpad=2)
    ax.set_title(f'{model}\nAcc={d["acc"]*100:.1f}%  $\\kappa$={d["kappa"]:.2f}', pad=3)
    for i in range(4):
        for j in range(4):
            val = cm[i, j]
            col = 'white' if cm_norm[i,j] > 0.5 else 'black'
            ax.text(j, i, str(val), ha='center', va='center', fontsize=6, color=col)
    ax.tick_params(length=2)

fig.suptitle('Normalized Confusion Matrices — Best Prompt per Model (DAiSEE)', fontsize=8, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig2_confusion_matrices.pdf'), bbox_inches='tight')
plt.close()
print("Fig 2 saved.")

# ─── Fig 3: SCB vs DAiSEE comparison ─────────────────────────
# SCB results from VLMBench doc
scb_best = {
    'CLIP':     {'acc': 0.6729, 'kappa': 0.5989, 'f1': 0.4715, 'mse': 0.5856},
    'BLIP-VQA': {'acc': 0.3356, 'kappa': 0.0,    'f1': 0.1256, 'mse': 0.7209},
    'GPT-4o':   {'acc': 0.6789, 'kappa': 0.5820, 'f1': 0.3999, 'mse': 0.3724},
    'LLaVA':    {'acc': 0.4923, 'kappa': 0.2239, 'f1': 0.3529, 'mse': 2.6661},
}
daisee_best_acc = {m: best[m]['acc'] for m in models}
daisee_best_kappa = {m: best[m]['kappa'] for m in models}
scb_best_acc = {m: scb_best[m]['acc'] for m in models}
scb_best_kappa = {m: scb_best[m]['kappa'] for m in models}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
x = np.arange(len(models))
width = 0.35

for ax, dai_d, scb_d, ylabel, title in zip(
    axes,
    [daisee_best_acc, daisee_best_kappa],
    [scb_best_acc, scb_best_kappa],
    ['Accuracy', "Cohen's $\\kappa$"],
    ['(a) Best Accuracy: DAiSEE vs. SCB', "(b) Best $\\kappa$: DAiSEE vs. SCB"],
):
    dai_vals = [dai_d[m] * (100 if 'Acc' in ylabel else 1) for m in models]
    scb_vals = [scb_d[m] * (100 if 'Acc' in ylabel else 1) for m in models]
    bars1 = ax.bar(x - width/2, dai_vals, width, label='DAiSEE', color='#4878CF', alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, scb_vals, width, label='SCB',    color='#D65F5F', alpha=0.85, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=10, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if 'Acc' in ylabel:
        ax.set_ylim(0, 85)
    else:
        ax.set_ylim(-0.1, 0.75)
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

handles2 = [mpatches.Patch(color='#4878CF', alpha=0.85, label='DAiSEE (individual)'),
            mpatches.Patch(color='#D65F5F', alpha=0.85, label='SCB (scene-level)')]
fig.legend(handles=handles2, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(os.path.join(OUT, 'fig3_cross_dataset.pdf'), bbox_inches='tight')
plt.close()
print("Fig 3 saved.")

# ─── Fig 4: Prediction distribution heatmap ──────────────────
# Shows class collapse
pred_dists = {
    'CLIP P1':     {0:4,  1:75,  2:214, 3:7},
    'CLIP P2':     {0:1,  1:216, 2:83,  3:0},
    'CLIP P3':     {0:0,  1:117, 2:167, 3:16},
    'BLIP P1':     {0:55, 1:245, 2:0,   3:0},
    'BLIP P2':     {0:16, 1:24,  2:260, 3:0},
    'BLIP P3':     {0:0,  1:0,   2:300, 3:0},
    'GPT-4o P1':   {0:12, 1:256, 2:32,  3:0},
    'GPT-4o P2':   {0:10, 1:246, 2:44,  3:0},
    'GPT-4o P3':   {0:1,  1:4,   2:294, 3:1},
    'LLaVA P1':    {0:254,1:0,   2:46,  3:0},
    'LLaVA P2':    {0:116,1:182, 2:1,   3:1},
    'LLaVA P3':    {0:1,  1:28,  2:268, 3:4},
    'Ground Truth':{0:4,  1:84,  2:106, 3:106},
}
labels_y = list(pred_dists.keys())
matrix = np.array([[pred_dists[k].get(c, 0) for c in range(4)] for k in labels_y], dtype=float)
matrix_norm = matrix / 300.0  # normalize

fig, ax = plt.subplots(figsize=(5.0, 3.8))
im = ax.imshow(matrix_norm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_xticks([0,1,2,3])
ax.set_xticklabels(['L0\nNot Eng.', 'L1\nBarely', 'L2\nEngaged', 'L3\nHighly'])
ax.set_yticks(range(len(labels_y)))
ax.set_yticklabels(labels_y, fontsize=7)
ax.set_xlabel('Predicted Class')
ax.set_title('Prediction Distribution Heatmap\n(fraction of 300 samples predicted per class)', fontsize=8)

# Add text annotations
for i in range(len(labels_y)):
    for j in range(4):
        val = matrix[i, j]
        col = 'white' if matrix_norm[i,j] > 0.6 else 'black'
        if val > 0:
            ax.text(j, i, f'{int(val)}', ha='center', va='center', fontsize=6, color=col)

# Add separator before ground truth
ax.axhline(y=len(labels_y)-1.5, color='white', linewidth=2)
plt.colorbar(im, ax=ax, shrink=0.8, label='Fraction of predictions')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig4_prediction_heatmap.pdf'), bbox_inches='tight')
plt.close()
print("Fig 4 saved.")

print("All figures generated successfully.")
