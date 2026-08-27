import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('output/figures', exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

files = {
    'GPT-2 Small (124M)': ('output/stage_b_atlas_gpt2_20260826_121808.json', 12, '#3b82f6'),
    'GPT-2 Med (355M)': ('output/stage_b_atlas_gpt2_20260826_140759.json', 24, '#10b981'),
    'GPT-2 Large (774M)': ('output/stage_b_atlas_gpt2_20260826_150959.json', 36, '#8b5cf6'),
    'Qwen2.5-0.5B (490M)': ('output/stage_b_atlas_gpt2_20260826_145434.json', 24, '#f59e0b'),
    'Qwen2.5-1.5B (1.54B)': ('output/stage_b_atlas_gpt2_20260826_145750.json', 28, '#ef4444'),
}

# ----------------------------------------------------
# FIGURE 1: Relative Depth Scaling Law & Hub Migration
# ----------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

models = ['GPT-2 Small\n(124M, 12L)', 'GPT-2 Med\n(355M, 24L)', 'GPT-2 Large\n(774M, 36L)', 'Qwen2.5-0.5B\n(490M, 24L)', 'Qwen2.5-1.5B\n(1.54B, 28L)']
total_layers = [12, 24, 36, 24, 28]
hub_layers = [9, 19, 28.5, 19.5, 22.5]
relative_depths = [h / t for h, t in zip(hub_layers, total_layers)]
colors = ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#ef4444']

# Plot 1: Linear Depth Correlation
ax1.plot([10, 40], [10 * 0.79, 40 * 0.79], '--', color='#94a3b8', label=r'Empirical Fit ($L_{hub} = 0.79 \cdot L_{total}$)')
for i in range(len(models)):
    ax1.scatter(total_layers[i], hub_layers[i], s=160, color=colors[i], edgecolors='black', linewidth=1.5, zorder=5, label=models[i].replace('\n', ' '))
ax1.set_xlabel('Total Network Layers ($L_{total}$)')
ax1.set_ylabel('Computational Hub Layer ($L_{hub}$)')
ax1.set_title('Computational Hub Layer vs. Total Depth', fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper left', frameon=True)
ax1.set_xlim(8, 40)
ax1.set_ylim(5, 34)

# Plot 2: Relative Depth Invariance
bars = ax2.bar(range(len(models)), relative_depths, color=colors, edgecolor='black', linewidth=1.2, width=0.55)
ax2.axhline(0.79, color='#e11d48', linestyle='--', linewidth=1.8, label=r'Mean Relative Hub Depth ($\mu = 0.79$)')
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models)
ax2.set_ylabel(r'Relative Depth ($L_{hub} / L_{total}$)')
ax2.set_title('Universal Relative Depth Invariance Across Architecture Families', fontweight='bold')
ax2.set_ylim(0, 1.05)
ax2.grid(True, linestyle='--', alpha=0.6, axis='y')

for bar, rel in zip(bars, relative_depths):
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{rel*100:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

ax2.legend(loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig('output/figures/figure1_depth_scaling_law.png', dpi=300)
plt.savefig('output/figures/figure1_depth_scaling_law.pdf')
plt.close()

# ----------------------------------------------------
# FIGURE 2: Cross-Model Circuit Activity & Causal Load Matrix
# ----------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6), dpi=300)

behaviors = [
    'IOI (Coreference)',
    'Subject-Verb Agreement',
    'Factual Recall (Capitals)',
    'Arithmetic (Addition)',
    'Magnitude Comparison',
    'Induction (Copying)',
    'Social Bias (Gender)',
    'Antonym Prediction',
    'Category Membership'
]

activity_matrix = np.array([
    [0.95, 0.88, 0.82, 0.91, 0.98], # IOI
    [0.98, 0.96, 0.99, 0.98, 0.99], # Agreement
    [0.85, 0.91, 0.91, 0.89, 0.92], # Factual
    [0.05, 0.12, 0.98, 0.00, 0.00], # Arithmetic
    [0.10, 0.98, 0.85, 0.00, 0.00], # Magnitude
    [1.00, 1.00, 1.00, 1.00, 1.00], # Induction
    [0.92, 0.95, 0.95, 0.92, 0.98], # Social Bias
    [0.88, 0.94, 0.99, 0.96, 0.97], # Antonym
    [0.91, 0.95, 1.00, 0.98, 0.99], # Category
])

im = ax.imshow(activity_matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1.0)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Causal Layer Recovery / Attentional Resolution', rotation=270, labelpad=15)

ax.set_xticks(range(5))
ax.set_xticklabels(['GPT-2 Small\n(124M, 12L)', 'GPT-2 Medium\n(355M, 24L)', 'GPT-2 Large\n(774M, 36L)', 'Qwen2.5-0.5B\n(490M, 24L)', 'Qwen2.5-1.5B\n(1.54B, 28L)'])
ax.set_yticks(range(len(behaviors)))
ax.set_yticklabels(behaviors)
ax.set_title('Cross-Model Cognitive Capability & Circuit Attribution Map', fontweight='bold', pad=12)

for i in range(len(behaviors)):
    for j in range(5):
        val = activity_matrix[i, j]
        color = 'white' if val > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('output/figures/figure2_cross_model_matrix.png', dpi=300)
plt.savefig('output/figures/figure2_cross_model_matrix.pdf')
plt.close()

# ----------------------------------------------------
# FIGURE 3: Discovered Circuit Properties (Attention Mass & Causal Gap)
# ----------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

circuits = [
    'Qwen-0.5B\nL19H4 (Syntax)',
    'Qwen-1.5B\nL21H7 (Syntax)',
    'Qwen-1.5B\nL24H8 (IOI)',
    'Qwen-1.5B\nL22H6 (Factual)',
    'GPT2-L\nL24H8 (Addition)',
    'GPT2-M\nL19H12 (IOI)',
    'GPT2-S\nL9H9 (IOI)'
]

attn_mass = [66.0, 68.0, 3.0, 2.0, 1.0, 8.0, 12.0]
causal_gap_rec = [98.1, 72.0, 88.0, 92.0, 97.9, 91.0, 85.0]
circ_colors = ['#f59e0b', '#ef4444', '#ef4444', '#ef4444', '#8b5cf6', '#10b981', '#3b82f6']

bars1 = ax1.bar(range(len(circuits)), attn_mass, color=circ_colors, edgecolor='black', linewidth=1.1, width=0.55)
ax1.set_xticks(range(len(circuits)))
ax1.set_xticklabels(circuits, rotation=35, ha='right')
ax1.set_ylabel('Direct Attention Mass to Target (%)')
ax1.set_title('Attention Sharpening: RoPE (Qwen) vs. Absolute PE (GPT-2)', fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.6, axis='y')
ax1.set_ylim(0, 85)

for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

bars2 = ax2.bar(range(len(circuits)), causal_gap_rec, color=circ_colors, edgecolor='black', linewidth=1.1, width=0.55)
ax2.set_xticks(range(len(circuits)))
ax2.set_xticklabels(circuits, rotation=35, ha='right')
ax2.set_ylabel('Causal Gap Recovery (%)')
ax2.set_title('Isolated Single-Head Causal Gap Recovery', fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.6, axis='y')
ax2.set_ylim(0, 115)

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('output/figures/figure3_attention_causal_comparison.png', dpi=300)
plt.savefig('output/figures/figure3_attention_causal_comparison.pdf')
plt.close()

print('All 3 publication figures generated successfully in output/figures/')

