import matplotlib.pyplot as plt
import pandas as pd

rejection_rates = [0, 5, 10, 15, 20, 25, 30]

data = {
    'Sky130': {
        'Baseline':       [56.00, 57.59, 58.79, 59.85, 60.71, 61.23, 61.98],
        'FiLM':           [54.02, 55.74, 57.36, 58.52, 59.47, 60.32, 61.16],
        'CrossAttention': [59.44, 59.80, 60.17, 60.71, 61.13, 61.39, 62.06],
    },
    'GF180': {
        'Baseline':       [91.00, 91.67, 92.43, 92.63, 93.54, 94.19, 94.17],
        'FiLM':           [85.63, 86.47, 86.56, 86.54, 86.53, 87.39, 87.30],
        'CrossAttention': [87.95, 88.47, 89.86, 90.45, 91.46, 92.35, 93.37],
    },
}

# Обновляем CSV с правильными именами доменов
rows = []
for domain, models_dict in data.items():
    for model, precs in models_dict.items():
        for rate, prec in zip(rejection_rates, precs):
            rows.append({
                'Domain': domain,
                'Model': model,
                'Rejection_Rate': rate,
                'Precision': prec,
            })
pd.DataFrame(rows).to_csv('rejection_curves_evidential.csv', index=False)

# ---- Точная ширина одной колонки elsarticle 5p (1:1 масштаб, как в Fig. 4) ----
FIG_W, FIG_H = 3.45, 1.95

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 7.5,
    'axes.labelsize': 8.0,
    'axes.titlesize': 8.0,
    'xtick.labelsize': 7.0,
    'ytick.labelsize': 7.0,
    'legend.fontsize': 7.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'pdf.fonttype': 42,
})

models = ['Baseline', 'FiLM', 'CrossAttention']
colors = {'Baseline': '#1f77b4', 'FiLM': '#2ca02c', 'CrossAttention': '#ff7f0e'}
markers = {'Baseline': 'o', 'FiLM': 's', 'CrossAttention': '^'}

panels = [
    ('Sky130', 'A) Sky130', (53.5, 62.6), [54, 56, 58, 60, 62]),
    ('GF180',  'B) GF180',  (85.0, 94.8), [86, 88, 90, 92, 94]),
]

fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharex=True, layout='constrained')

for idx, (key, title, ylim, yticks) in enumerate(panels):
    ax = axes[idx]
    for m in models:
        ax.plot(
            rejection_rates,
            data[key][m],
            color=colors[m],
            marker=markers[m],
            ms=3.8,
            lw=1.4,
            label=m if idx == 0 else None,
            zorder=3,
        )

    ax.set_title(title, fontweight='bold', pad=4)
    ax.set_xticks(rejection_rates)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.grid(True, ls=':', lw=0.5, alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel('Confident Precision (%)', fontweight='bold', fontsize=8.0)

fig.get_layout_engine().set(rect=(0, 0, 1, 0.88))
fig.draw_without_rendering()
p_left = axes[0].get_position()
p_right = axes[1].get_position()
x_center = (p_left.x0 + p_right.x1) / 2

fig.supxlabel('Rejection Rate (%)', x=x_center, fontweight='bold', fontsize=8.0)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='upper center',
    bbox_to_anchor=(x_center, 1.0),
    ncol=3,
    frameon=False,
    handlelength=1.8,
    columnspacing=1.2,
    handletextpad=0.4,
)

fig.savefig('fig_rejection_curves.pdf')
fig.savefig('fig_rejection_curves.png', dpi=300)
plt.close()
print("Saved in-domain single-column fig_rejection_curves.pdf and .png")
