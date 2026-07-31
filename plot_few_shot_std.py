import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_zero_shot_metrics():
    """
    Вычисляет (или загружает) метрики Zero-Shot для левого края графика (0% данных).
    """
    metrics = {
        'No_FiLM': 0.2506,
        'FiLM': 0.2527,        # <- Замени на реальную цифру после прогона
        'CrossAttention': 0.1889
    }
    
    print("✅ Zero-Shot результаты получены!")
    return metrics


def main():
    # 1. Загрузка данных Few-Shot (уже 3 модели и 3 сида)
    df = pd.read_csv("few_shot_results_3models.csv")

    # 2. Добавляем Zero-Shot (0%). 
    # Динамически генерируем строки для всех сидов, моделей и стратегий
    zero_shot_metrics = get_zero_shot_metrics()
    zero_shot_data = []
    
    seeds = [42, 43, 44] # Сиды, которые ты использовал
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

    # 3. Преобразование оси X в строковые категории для равномерного шага
    df['Data_Fraction_Str'] = df['Data_Fraction'].apply(
        lambda x: '0%\n(Zero-Shot)' if x == 0 else f"{int(x*100)}%"
    )

    # 4. Настройка стиля и палитры для ТРЕХ моделей
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    # Добавили классический зеленый цвет для FiLM, чтобы график был контрастным
    palette = {
        "No_FiLM": "#0052cc",        # Синий
        "FiLM": "#2ca02c",           # Зеленый
        "CrossAttention": "#e67300"  # Оранжевый
    }

    # --- График A: Frozen Backbone ---
    sns.lineplot(
        data=df[df['Strategy'] == 'Frozen'], 
        x='Data_Fraction_Str', y='PR_AUC', hue='Model', 
        errorbar='sd', 
        marker='o', linewidth=2.5, markersize=8, palette=palette, ax=axes[0]
    )
    axes[0].set_title('A) Frozen Backbone (PEFT)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Sky130 Training Data (%)', fontsize=11)
    axes[0].set_ylabel('PR-AUC', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # --- График B: Full Fine-Tuning ---
    sns.lineplot(
        data=df[df['Strategy'] == 'Full'], 
        x='Data_Fraction_Str', y='PR_AUC', hue='Model', 
        errorbar='sd', 
        marker='X', linewidth=2.5, markersize=9, palette=palette, ax=axes[1]
    )
    axes[1].set_title('B) Full Fine-Tuning', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Sky130 Training Data (%)', fontsize=11)
    axes[1].set_ylabel('')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    HOME_DOMAIN_PR_AUC = 0.6986 # Твой результат на GF180
    
    for ax in axes:
        # Рисуем горизонтальную линию
        ax.axhline(y=HOME_DOMAIN_PR_AUC, color='gray', linestyle='--', linewidth=2, alpha=0.8, zorder=0)
        
        # Легкая заливка от линии до верха графика, чтобы показать "идеальную зону"
        ax.fill_between(x=[-0.5, 5.5], y1=HOME_DOMAIN_PR_AUC, y2=1.0, color='gray', alpha=0.1, zorder=0)
        
        # Добавляем текстовую подпись прямо над линией
        ax.text(x=0.0, y=HOME_DOMAIN_PR_AUC + 0.005, s=f'Source Domain Oracle (GF180: {HOME_DOMAIN_PR_AUC})', 
                color='gray', fontsize=10, fontweight='bold', ha='left', va='bottom')
        
    # Немного расширим лимиты по Y, чтобы линия и текст не прилипли к краю
    axes[0].set_ylim(0.15, HOME_DOMAIN_PR_AUC + 0.05)
    
    
    # 5. Оформление общей легенды
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    
    # Легенда с тремя архитектурами
    axes[1].legend(
        handles, labels, 
        loc='lower right', 
        title="Architecture", fontsize=11, title_fontsize=12,
        frameon=True, shadow=True
    )

    plt.tight_layout()
    plt.savefig('few_shot_split_std.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('few_shot_split_std.png', dpi=300, bbox_inches='tight')
    print("✅ Графики для 3-х моделей со стандартным отклонением успешно сохранены!")

if __name__ == "__main__":
    main()
