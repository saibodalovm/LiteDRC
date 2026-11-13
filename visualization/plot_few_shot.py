import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 1. Загрузка основных данных
    df = pd.read_csv("few_shot_results.csv")

    # 2. Добавление данных Zero-Shot (0%)
    zero_shot_data = [
        {'Model': 'No_FiLM', 'Data_Fraction': 0.0, 'Strategy': 'Frozen', 'PR_AUC': 0.3818},
        {'Model': 'No_FiLM', 'Data_Fraction': 0.0, 'Strategy': 'Full', 'PR_AUC': 0.3818},
        {'Model': 'CrossAttention', 'Data_Fraction': 0.0, 'Strategy': 'Frozen', 'PR_AUC': 0.3796},
        {'Model': 'CrossAttention', 'Data_Fraction': 0.0, 'Strategy': 'Full', 'PR_AUC': 0.3796}
    ]
    df = pd.concat([pd.DataFrame(zero_shot_data), df], ignore_index=True)

    # 3. Преобразование оси X в строковые категории для равномерного шага на графике
    df['Data_Fraction_Str'] = df['Data_Fraction'].apply(
        lambda x: '0%\n(Zero-Shot)' if x == 0 else f"{int(x*100)}%"
    )

    # 4. Настройка стиля графика
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    palette = {"No_FiLM": "#0052cc", "CrossAttention": "#e67300"}

    # --- График А: Frozen ---
    sns.lineplot(
        data=df[df['Strategy'] == 'Frozen'], 
        x='Data_Fraction_Str', y='PR_AUC', hue='Model', 
        marker='o', linewidth=2.5, markersize=8, palette=palette, ax=axes[0]
    )
    axes[0].set_title('A) Frozen Backbone (Parameter-Efficient)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Fraction of Target Domain (Sky130) Data', fontsize=11)
    axes[0].set_ylabel('Global PR-AUC', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # --- График B: Full ---
    sns.lineplot(
        data=df[df['Strategy'] == 'Full'], 
        x='Data_Fraction_Str', y='PR_AUC', hue='Model', 
        marker='X', linewidth=2.5, markersize=9, palette=palette, ax=axes[1]
    )
    axes[1].set_title('B) Full Fine-Tuning', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Fraction of Target Domain (Sky130) Data', fontsize=11)
    axes[1].set_ylabel('')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # 5. Оформление общей легенды
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    fig.legend(
        handles, labels, 
        loc='lower center', bbox_to_anchor=(0.5, -0.05), 
        ncol=2, title="Architecture", fontsize=12, title_fontsize=12,
        frameon=True, shadow=True
    )

    plt.tight_layout()
    plt.savefig('few_shot_split.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('few_shot_split.png', dpi=300, bbox_inches='tight')
    print("✅ Графики сохранены как few_shot_split.pdf и .png")

if __name__ == "__main__":
    main()