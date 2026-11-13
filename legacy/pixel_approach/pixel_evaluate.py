import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from tqdm import tqdm  # Для красивого прогресс-бара

# Импортируем твои модули
from models.modelFILM import CenterNetEDAUnet
from pixel_approach.pixel_patch_dataset import EDAChipDataset
from pixel_approach.pixel_metrics import evaluate_batch
import random

def evaluate_model(model, dataloader, device, obj_threshold):
    model.eval()
    
    total_metrics = dict(TP=0, FP=0, FN=0, cls_correct=0, n_true=0, n_pred=0)
    total_pixels = 0  # Для подсчета TN
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(batch["features"], batch["pdk_meta"], batch["pdk_id"])
            
            # ВАЖНО: Переводим логиты в вероятности перед оценкой
            preds['obj'] = torch.sigmoid(preds['obj'])
            
            # Считаем Hungarian метрики для батча
            batch_res = evaluate_batch(
                preds, batch, 
                obj_threshold=obj_threshold, 
                dist_threshold_gcells=2.0
            )
            
            for k in total_metrics.keys():
                total_metrics[k] += batch_res[k]
                
            # Считаем площадь патчей в пикселях для FPR
            total_pixels += batch["features"].shape[0] * batch["features"].shape[2] * batch["features"].shape[3]
                
    precision = total_metrics['TP'] / (total_metrics['TP'] + total_metrics['FP'] + 1e-6)
    recall = total_metrics['TP'] / (total_metrics['TP'] + total_metrics['FN'] + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    # Считаем FPR (False Positive Rate) для ROC кривой
    actual_negatives = total_pixels - total_metrics['n_true']
    fpr = total_metrics['FP'] / (actual_negatives + 1e-6)
    
    return precision, recall, fpr, f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство для инференса: {device}")
    
    TEST_DIR = r"J:\processed_dataset\val"
    PATCH_SIZE = 64
    BATCH_SIZE = 128  # Можно побольше для быстрого инференса
    MODEL_PATH = "best_model_f1_file_sampler_FILM.pth"
    
    test_dataset = EDAChipDataset(data_dir=TEST_DIR, stats_path="J:/processed_dataset/train/global_zscore_stats.json", patch_size=PATCH_SIZE, samples_per_file=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    model = CenterNetEDAUnet(in_channels=3, num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    print(f"Веса модели загружены из {MODEL_PATH}\n")
    
    # Создаем сетку порогов (от самых мягких к самым жестким)
    thresholds = np.linspace(0.01, 0.70, 30)
    
    precisions, recalls, fprs, f1_scores = [], [], [], []
    
    print("Вычисляем точки для кривых...")
    for thresh in tqdm(thresholds, desc="Evaluating thresholds"):
        # Фиксируем сиды, чтобы инференс был 100% детерминированным
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        
        prec, rec, fpr, f1 = evaluate_model(model, test_loader, device, obj_threshold=thresh)
        
        precisions.append(prec)
        recalls.append(rec)
        fprs.append(fpr)
        f1_scores.append(f1)

    # ================= ИСПРАВЛЕНИЕ МАТЕМАТИКИ AUC =================
    # Для PR-кривой добавляем краевые точки:
    # (Recall=0, Precision=1) - идеальный жесткий порог
    # (Recall=1, Precision=0) - самый мягкий порог (все пиксели = дефекты)
    recalls_pr = np.array(recalls + [0.0, 1.0])
    precisions_pr = np.array(precisions + [1.0, 0.0])
    pr_sort_idx = np.argsort(recalls_pr)
    pr_auc = auc(recalls_pr[pr_sort_idx], precisions_pr[pr_sort_idx])

    # Для ROC-кривой добавляем краевые точки:
    # (FPR=0, TPR=0) - жесткий порог
    # (FPR=1, TPR=1) - самый мягкий порог
    recalls_roc = np.array(recalls + [0.0, 1.0])
    fprs_roc = np.array(fprs + [0.0, 1.0])
    roc_sort_idx = np.argsort(fprs_roc)
    roc_auc = auc(fprs_roc[roc_sort_idx], recalls_roc[roc_sort_idx])  # <-- ЗДЕСЬ БЫЛ БАГ

    best_f1_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_f1_idx]
    
    print(f"\n--- ИТОГИ ---")
    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Максимальный F1-score: {f1_scores[best_f1_idx]*100:.2f}% (при пороге {best_thresh:.2f})")

    # ================= РИСУЕМ ГРАФИКИ =================
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # График 1: Precision-Recall Curve (используем отсортированные массивы!)
    ax1.plot(recalls_pr[pr_sort_idx], precisions_pr[pr_sort_idx], color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    ax1.scatter(recalls[best_f1_idx], precisions[best_f1_idx], color='red', s=100, zorder=5, 
                label=f'Best F1 (Thr={best_thresh:.2f})')
    ax1.set_xlabel('Recall (True Positive Rate)')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve for DRC Prediction')
    ax1.legend(loc='upper right')
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    
    # График 2: ROC Curve (используем отсортированные массивы!)
    ax2.plot(fprs_roc[roc_sort_idx], recalls_roc[roc_sort_idx], color='green', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate (Recall)')
    ax2.set_title('ROC Curve (Note: highly imbalanced background)')
    ax2.legend(loc='lower right')
    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    plt.savefig('AUC_curves.png', dpi=300)
    print("\nГрафики сохранены в файл 'AUC_curves.png'")
    plt.show()

if __name__ == "__main__":
    main()