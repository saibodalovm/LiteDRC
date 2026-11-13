import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
from tqdm import tqdm

# Импортируем наши новые модули
from models.Encoder import LightweightRegionNet
from data_processing.patch_dataset import EDAChipDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство для инференса: {device}")
    
    TEST_DIR = r"J:\processed_dataset\val"
    STATS_PATH = "J:/processed_dataset/train/global_zscore_stats.json"
    PATCH_SIZE = 64
    BATCH_SIZE = 128
    MODEL_PATH = "best_lightweight_region_net_NOFILM.pth" # Имя новых весов
    
    # ВАЖНО: Включаем mode='test', чтобы сработал детерминированный тайлинг (без рандома)
    test_dataset = EDAChipDataset(
        data_dir=TEST_DIR, 
        stats_path=STATS_PATH, 
        patch_size=PATCH_SIZE, 
        mode='test' # Сетка патчей
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    model = LightweightRegionNet(in_channels=3, num_classes=3, use_film=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"Веса модели загружены из {MODEL_PATH}\n")
    
    all_preds = []
    all_targets = []
    
    print("Прогоняем инференс по детерминированной сетке чипов...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            features = batch["features"].to(device)
            pdk_meta = batch["pdk_meta"].to(device)
            pdk_id = batch["pdk_id"].to(device)
            targets = batch["target_regions"].cpu().numpy() # (B, 3, 8, 8)
            
            preds_logits = model(features, pdk_meta, pdk_id)["pred_regions"]
            preds_probs = torch.sigmoid(preds_logits).cpu().numpy() # (B, 3, 8, 8)
            
            all_preds.append(preds_probs)
            all_targets.append(targets)

    # Склеиваем всё в единые массивы
    all_preds = np.concatenate(all_preds, axis=0)     # (N_patches, 3, 8, 8)
    all_targets = np.concatenate(all_targets, axis=0) # (N_patches, 3, 8, 8)
    
    # Для построения глобальных кривых "схлопываем" пространственные измерения
    # Теперь у нас просто два огромных массива: реальные дефекты и предсказанные вероятности
    y_true_flat = all_targets.flatten()
    y_scores_flat = all_preds.flatten()
    
    # Также выделим отдельно класс Shorts (канал 0)
    y_true_shorts = all_targets[:, 0, :, :].flatten()
    y_scores_shorts = all_preds[:, 0, :, :].flatten()

    print("\nВычисляем метрики через sklearn...")
    # ================= РАСЧЕТ КРИВЫХ (GLOBAL) =================
    # PR Curve
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true_flat, y_scores_flat)
    pr_auc = average_precision_score(y_true_flat, y_scores_flat)
    
    # Ищем лучший F1 глобально
    f1_scores = np.divide(2 * precisions * recalls, precisions + recalls, out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_thresh = pr_thresholds[best_idx] if best_idx < len(pr_thresholds) else 1.0
    
    # ROC Curve
    fprs, tprs, roc_thresholds = roc_curve(y_true_flat, y_scores_flat)
    roc_auc_val = auc(fprs, tprs)
    
    # ================= РАСЧЕТ КРИВЫХ (SHORTS ONLY) =================
    precisions_s, recalls_s, _ = precision_recall_curve(y_true_shorts, y_scores_shorts)
    pr_auc_shorts = average_precision_score(y_true_shorts, y_scores_shorts)

    print(f"\n--- ИТОГИ (GLOBAL) ---")
    print(f"ROC-AUC: {roc_auc_val:.4f}")
    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")
    print(f"Максимальный Global F1: {best_f1*100:.2f}% (при пороге {best_thresh:.3f})")
    print(f"PR-AUC специально для Shorts: {pr_auc_shorts:.4f}")

    # ================= РИСУЕМ ГРАФИКИ =================
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # График 1: Precision-Recall Curve
    ax1.plot(recalls, precisions, color='blue', lw=2, label=f'Global PR (AUC = {pr_auc:.3f})')
    ax1.plot(recalls_s, precisions_s, color='purple', lw=2, linestyle=':', label=f'Shorts PR (AUC = {pr_auc_shorts:.3f})')
    ax1.scatter(recalls[best_idx], precisions[best_idx], color='red', s=100, zorder=5, 
                label=f'Best F1={best_f1:.2f} (Thr={best_thresh:.2f})')
    ax1.set_xlabel('Recall (Полнота)')
    ax1.set_ylabel('Precision (Точность)')
    ax1.set_title('Precision-Recall Curve (Regional 8x8)')
    ax1.legend(loc='upper right')
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    
    # График 2: ROC Curve
    ax2.plot(fprs, tprs, color='green', lw=2, label=f'Global ROC (AUC = {roc_auc_val:.3f})')
    ax2.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax2.set_xlabel('False Positive Rate (Доля ложных тревог)')
    ax2.set_ylabel('True Positive Rate (Recall)')
    ax2.set_title('ROC Curve')
    ax2.legend(loc='lower right')
    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    plt.savefig('Region_AUC_curves.png', dpi=300)
    print("\nГрафики сохранены в файл 'Region_AUC_curves.png'")
    plt.show()

if __name__ == "__main__":
    main()