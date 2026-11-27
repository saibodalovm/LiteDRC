#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot few-shot results for LiteDRC")
    parser.add_argument("--input-csv", type=str, default="few_shot_results.csv", help="Input CSV file with few-shot results")
    parser.add_argument("--out-pdf", type=str, default="few_shot_split_std.pdf", help="Output PDF figure path")
    parser.add_argument("--out-png", type=str, default="few_shot_split_std.png", help="Output PNG figure path")
    parser.add_argument("--oracle", type=float, default=0.7107, help="Source domain oracle PR-AUC value")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_csv):
        print(f"[Warning] Input CSV {args.input_csv} not found. Creating dummy/empty plot or skipping.")
        return
        
    df = pd.read_csv(args.input_csv)
    df['Model'] = df['Model'].replace({'No_FiLM': 'Baseline'})
    
    zero_shot_metrics = {'Baseline': 0.2082, 'FiLM': 0.2110, 'CrossAttention': 0.1925}
    zero_shot_data = []
    seeds = [42, 43, 44]
    strategies = ['Frozen', 'Full']
    
    for seed in seeds:
        for model_name, pr_val in zero_shot_metrics.items():
            for strategy in strategies:
                zero_shot_data.append({
                    'Seed': seed, 'Model': model_name, 'Data_Fraction': 0.0,
                    'Strategy': strategy, 'PR_AUC': pr_val
                })
                
    df = pd.concat([pd.DataFrame(zero_shot_data), df], ignore_index=True)
    df = df.sort_values(by='Data_Fraction')
    
    unique_fractions = sorted(df['Data_Fraction'].unique())
    labels_map = {x: ('0%\n(Zero-Shot)' if x == 0 else f"{int(x*100)}%") for x in unique_fractions}
    df['Data_Fraction_Str'] = df['Data_Fraction'].map(labels_map)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    colors = {
        "Baseline": "#0052cc",
        "FiLM": "#2ca02c",
        "CrossAttention": "#e67300"
    }
    
    for ax_idx, strategy in enumerate(['Frozen', 'Full']):
        ax = axes[ax_idx]
        subset_strat = df[df['Strategy'] == strategy]
        for model_name, color in colors.items():
            subset_model = subset_strat[subset_strat['Model'] == model_name]
            if len(subset_model) > 0:
                grouped = subset_model.groupby('Data_Fraction_Str')['PR_AUC'].agg(['mean', 'std']).reindex(labels_map.values())
                ax.plot(grouped.index, grouped['mean'], marker='o', linewidth=2.5, markersize=8, color=color, label=model_name)
                ax.fill_between(grouped.index, grouped['mean'] - grouped['std'], grouped['mean'] + grouped['std'], color=color, alpha=0.2)
                
        title = 'A) Frozen Backbone (PEFT)' if strategy == 'Frozen' else 'B) Full Fine-Tuning (FFT)'
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Target Data Fraction', fontsize=11, fontweight='bold')
        if ax_idx == 0:
            ax.set_ylabel('PR-AUC', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        
    max_pr_auc = max(df['PR_AUC'].max(), args.oracle)
    min_pr_auc = df['PR_AUC'].min()
    y_min = max(0.0, min_pr_auc - 0.03)
    y_max = min(1.0, max_pr_auc + 0.08)
    
    for ax in axes:
        ax.set_ylim(y_min, y_max)
        ax.axhline(y=args.oracle, color='gray', linestyle='--', linewidth=2, alpha=0.8, zorder=2)
        ax.text(
            x=0.0, y=args.oracle + 0.008,
            s=f'Source Domain Oracle ({args.oracle:.4f})',
            color='gray', fontsize=9.5, fontweight='bold',
            ha='left', va='bottom', zorder=4
        )
        
    axes[1].legend(loc='lower right', title="Architecture", fontsize=10.5, title_fontsize=11.5, frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(args.out_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(args.out_png, dpi=300, bbox_inches='tight')
    print(f"✅ Saved few-shot plots to {args.out_pdf} and {args.out_png}")

if __name__ == "__main__":
    main()
