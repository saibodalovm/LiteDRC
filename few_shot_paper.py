import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_sky_gf = pd.read_csv('few_shot_results_evidential_sky_gf.csv')
df_gf_sky = pd.read_csv('few_shot_results_evidential_gf_sky.csv')

zero_shot_sky_gf = {'No_FiLM': 0.1992, 'FiLM': 0.2239, 'CrossAttention': 0.2060}
zero_shot_gf_sky = {'No_FiLM': 0.2082, 'FiLM': 0.2110, 'CrossAttention': 0.1925}

# ---- Ширина одной колонки elsarticle 5p ----
FIG_W, FIG_H = 3.45, 3.9

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 7.5,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'pdf.fonttype': 42,
})

models = ['No_FiLM', 'FiLM', 'CrossAttention']
model_labels = {'No_FiLM': 'Baseline', 'FiLM': 'FiLM', 'CrossAttention': 'CrossAttention'}
colors = {'No_FiLM': '#1f77b4', 'FiLM': '#2ca02c', 'CrossAttention': '#ff7f0e'}
markers = {'No_FiLM': 'o', 'FiLM': 's', 'CrossAttention': '^'}

oracle_sky, oracle_gf = 0.5479, 0.7107

# (row, col, df, zero-shot, strategy, oracle, column title, row tag)
configs = [
    (0, 0, df_sky_gf, zero_shot_sky_gf, 'Frozen', oracle_sky,
     'A) Frozen backbone (PEFT)', r'Sky130$\rightarrow$GF180'),
    (0, 1, df_sky_gf, zero_shot_sky_gf, 'Full', oracle_sky,
     'B) Full fine-tuning (FFT)', None),
    (1, 0, df_gf_sky, zero_shot_gf_sky, 'Frozen', oracle_gf,
     None, r'GF180$\rightarrow$Sky130'),
    (1, 1, df_gf_sky, zero_shot_gf_sky, 'Full', oracle_gf,
     None, None),
]

few_shot_fractions = [0.01, 0.05, 0.10, 0.25, 1.00]
x_labels = ['0\n(ZS)', '1', '5', '10', '25', '100']
x_idx = np.arange(len(x_labels))

fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_H), sharex=True, sharey=True,
                         layout='constrained')

for row, col, df, zs, strat, oracle, title, tag in configs:
    ax = axes[row, col]
    sub = df[df['Strategy'] == strat]

    ax.axhline(oracle, color='0.4', ls='--', lw=0.9, zorder=1,
               label='Source oracle' if (row, col) == (0, 0) else None)

    for m in models:
        stats = (sub[sub['Model'] == m]
                 .groupby('Data_Fraction')['PR_AUC'].agg(['mean', 'std'])
                 .reindex(few_shot_fractions))
        means = np.array([zs[m]] + list(stats['mean'].values))
        stds = np.array([0.0] + list(stats['std'].fillna(0).values))

        ax.plot(x_idx, means, color=colors[m], marker=markers[m],
                ms=3.5, lw=1.3, label=model_labels[m] if (row, col) == (0, 0) else None,
                zorder=3)
        ax.fill_between(x_idx, means - stds, means + stds,
                        color=colors[m], alpha=0.15, lw=0, zorder=2)

    if title:
        ax.set_title(title, fontweight='bold', pad=3)
    if tag:
        ax.text(0.04, 0.97, tag, transform=ax.transAxes, ha='left', va='top',
                fontsize=7.5, fontweight='bold', color='0.15')

    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0.15, 0.86)
    ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.grid(True, ls=':', lw=0.5, alpha=0.6)

fig.supylabel('PR-AUC', fontweight='bold', fontsize=8)

# оставляем сверху место под легенду
fig.get_layout_engine().set(rect=(0, 0, 1, 0.94))

# даём layout-у посчитать позиции панелей и берём их горизонтальный центр
fig.draw_without_rendering()
p_left = axes[0, 0].get_position()
p_right = axes[0, 1].get_position()
x_center = (p_left.x0 + p_right.x1) / 2

fig.supxlabel('Target data fraction (%)', x=x_center, fontweight='bold', fontsize=8)

handles, labels = axes[0, 0].get_legend_handles_labels()
order = [labels.index(l) for l in ['Baseline', 'FiLM', 'CrossAttention', 'Source oracle']]
fig.legend([handles[i] for i in order], [labels[i] for i in order],
           loc='upper center', bbox_to_anchor=(x_center, 1.0), ncol=4,
           frameon=False, handlelength=1.6, columnspacing=1.0, handletextpad=0.4)

fig.savefig('fig4_target_domain_pr_auc.pdf')
fig.savefig('fig4_target_domain_pr_auc.png', dpi=300)
plt.close()