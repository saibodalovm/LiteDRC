import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_zero_shot_metrics():
    """
    Вычисляет (или загружает) метрики Zero-Shot для левого края графика (0% данных).
    """
    metrics = {
        'Baseline': 0.2082,
        'FiLM': 0.2110,        # <- Замени на реальную цифру после прогона
        'CrossAttention': 0.1925
    }

    '''metrics = {
        'Baseline': 0.1992,
        'FiLM': 0.2239,        # <- Замени на реальную цифру после прогона
        'CrossAttention': 0.2060
    }'''
    
    print("✅ Zero-Shot результаты получены!")
    return metrics


def main():
    # 1. Загрузка данных Few-Shot
    df = pd.read_csv(r"C:\Users\pcuser\DRCpred\few_shot_results_evidential_gf_sky.csv")

    # Переименовываем No_FiLM в Baseline
    df['Model'] = df['Model'].replace({'No_FiLM': 'Baseline'})

    # 2. Добавляем Zero-Shot (0%)
    zero_shot_metrics = get_zero_shot_metrics()
    zero_shot_data = []
    
    seeds = [42, 43, 44]
    strategies = ['Frozen', 'Full']
    
    for seed in seeds:
        for model_name, pr_auc_value in zero_shot_metrics.items():
            for strategy in strategies:
                zero_shot_data.append({
                    'Seed': seed, 
                    'Model': model_name, 
                    'Data_Fraction': 0.0, 
                    'Strategy': strategy, 
                    'PR_AUC': pr_auc_value
                })
                
    df = pd.concat([pd.DataFrame(zero_shot_data), df], ignore_index=True)

    # 3. Сортируем по числовому значению
    df = df.sort_values(by='Data_Fraction')

    # Формируем читаемые подписи
    unique_fractions = sorted(df['Data_Fraction'].unique())
    labels_map = {x: ('0%\n(Zero-Shot)' if x == 0 else f"{int(x*100)}%") for x in unique_fractions}
    df['Data_Fraction_Str'] = df['Data_Fraction'].map(labels_map)
    
    # 🔥 ФИКС: Задаем категориальный тип данных с жестким порядком категорий
    categories_order = [labels_map[x] for x in unique_fractions]
    df['Data_Fraction_Str'] = pd.Categorical(df['Data_Fraction_Str'], categories=categories_order, ordered=True)

    # 4. Настройка стиля и палитры
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    palette = {
        "Baseline": "#0052cc",        # Синий
        "FiLM": "#2ca02c",            # Зеленый
        "CrossAttention": "#e67300"   # Оранжевый
    }

    # --- График A: Frozen Backbone (PEFT) ---
    sns.lineplot(
        data=df[df['Strategy'] == 'Frozen'], 
        x='Data_Fraction_Str', y='PR_AUC', hue='Model', 
        errorbar='sd', 
        marker='o', linewidth=2.5, markersize=8, palette=palette, ax=axes[0]
    )
    axes[0].set_title('A) Frozen Backbone (PEFT)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Target Data Fraction (sky130)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('PR-AUC', fontsize=12, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # --- График B: Full Fine-Tuning (FFT) ---
    sns.lineplot(
        data=df[df['Strategy'] == 'Full'], 
        x='Data_Fraction_Str', y='PR_AUC', hue='Model', 
        errorbar='sd', 
        marker='o', linewidth=2.5, markersize=8, palette=palette, ax=axes[1]
    )
    axes[1].set_title('B) Full Fine-Tuning (FFT)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Target Data Fraction (sky130)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # 5. Настройка лимитов и Оракула
    HOME_DOMAIN_PR_AUC = 0.7107 #0.7107 0.5479

    max_pr_auc = max(df['PR_AUC'].max(), HOME_DOMAIN_PR_AUC)
    min_pr_auc = df['PR_AUC'].min()

    y_min = max(0.0, min_pr_auc - 0.03)
    y_max = min(1.0, max_pr_auc + 0.08)

    for ax in axes:
        ax.set_ylim(y_min, y_max)
        
        # Пунктир Оракула
        ax.axhline(y=HOME_DOMAIN_PR_AUC, color='gray', linestyle='--', linewidth=2, alpha=0.8, zorder=2)
        
        # Подпись Оракула
        ax.text(
            x=0.0, y=HOME_DOMAIN_PR_AUC + 0.008, 
            s=f'Source Domain Oracle (GF180: {HOME_DOMAIN_PR_AUC:.4f})', 
            color='gray', fontsize=9.5, fontweight='bold', 
            ha='left', va='bottom', zorder=4
        )

    # 6. Единая легенда
    handles, labels = axes[0].get_legend_handles_labels()
    
    if axes[0].get_legend(): axes[0].get_legend().remove()
    if axes[1].get_legend(): axes[1].get_legend().remove()
    
    axes[1].legend(
        handles, labels, 
        loc='lower right', 
        title="Architecture", fontsize=10.5, title_fontsize=11.5,
        frameon=True, shadow=True
    )

    plt.tight_layout()
    plt.savefig('few_shot_split_std.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('few_shot_split_std.png', dpi=300, bbox_inches='tight')
    print("✅ Идеально чистая версия графиков сохранена!")

if __name__ == "__main__":
    main()