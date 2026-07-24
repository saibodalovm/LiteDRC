import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from tqdm import tqdm

from models.modelFILM import CenterNetEDAUnet
from pixel_approach.pixel_patch_dataset import EDAChipDataset
import random

def evaluate_hotspots_detailed(preds, targets, obj_threshold, window_size=8):
    num_classes = preds['cls'].shape[1]
    
    # 1. Предсказания
    pred_obj = torch.sigmoid(preds['obj']) # (B, 1, H, W)
    pred_mask = (pred_obj > obj_threshold).float()
    pred_cls_probs = F.softmax(preds['cls'], dim=1) # (B, C, H, W)
    pred_hotspots_raw = pred_cls_probs * pred_mask
    
    # 2. Таргеты (Ground Truth)
    target_mask = (targets['mask'] > 0).float()
    target_cls = targets['cls'].long()
    target_one_hot = F.one_hot(target_cls, num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_hotspots_raw = target_one_hot * target_mask
    
    # 3. Региональный Pooling (8x8)
    pred_regions = F.max_pool2d(pred_hotspots_raw, kernel_size=window_size, stride=window_size) > 0.1
    target_regions = F.max_pool2d(target_hotspots_raw, kernel_size=window_size, stride=window_size) > 0
    
    # --- А. БИНАРНЫЕ МЕТРИКИ (Есть ли вообще дефект в окне) ---
    pred_any = pred_regions[:, 1:, :, :].any(dim=1, keepdim=True) # Любой класс > 0
    target_any = target_regions[:, 1:, :, :].any(dim=1, keepdim=True)
    
    bin_TP = (pred_any & target_any).sum().item()
    bin_FP = (pred_any & ~target_any).sum().item()
    bin_FN = (~pred_any & target_any).sum().item()
    bin_TN = (~pred_any & ~target_any).sum().item()
    
    # --- Б. ПОКЛАССОВЫЕ МЕТРИКИ (Для каждого класса отдельно) ---
    class_stats = {}
    for c in range(1, num_classes): # Пропускаем 0 (фон)
        p_c = pred_regions[:, c:c+1, :, :]
        t_c = target_regions[:, c:c+1, :, :]
        
        tp = (p_c & t_c).sum().item()
        fp = (p_c & ~t_c).sum().item()
        fn = (~p_c & t_c).sum().item()
        class_stats[c] = (tp, fp, fn)
        
    return (bin_TP, bin_FP, bin_FN, bin_TN), class_stats


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")
    
    TEST_DIR = r"J:\processed_dataset\val"
    PATCH_SIZE = 64
    BATCH_SIZE = 128
    MODEL_PATH = "best_model_f1_file_sampler_FILM.pth"
    WINDOW_SIZE = 8
    
    test_dataset = EDAChipDataset(data_dir=TEST_DIR, stats_path="J:/processed_dataset/train/global_zscore_stats.json", patch_size=PATCH_SIZE, samples_per_file=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    model = CenterNetEDAUnet(in_channels=3, num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Веса загружены. Запуск детального анализа (Бинарный + Поклассовый)...\n")
    
    thresholds = np.linspace(0.01, 0.70, 30)
    bin_f1s, bin_recs, bin_precs, bin_fprs = [], [], [], []
    
    best_results = None
    best_f1_val = -1
    best_thresh = 0
    
    with torch.no_grad():
        for thresh in tqdm(thresholds, desc="Проверка порогов"):
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            
            tot_tp, tot_fp, tot_fn, tot_tn = 0, 0, 0, 0
            agg_class_stats = {1: [0,0,0], 2: [0,0,0], 3: [0,0,0]} # Для классов 1, 2, 3
            
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                preds = model(batch["features"], batch["pdk_meta"], batch["pdk_id"])
                
                (btp, bfp, bfn, btn), c_stats = evaluate_hotspots_detailed(preds, batch, obj_threshold=thresh, window_size=WINDOW_SIZE)
                tot_tp += btp
                tot_fp += bfp
                tot_fn += bfn
                tot_tn += btn
                
                for c in c_stats:
                    agg_class_stats[c][0] += c_stats[c][0]
                    agg_class_stats[c][1] += c_stats[c][1]
                    agg_class_stats[c][2] += c_stats[c][2]
                
            prec = tot_tp / (tot_tp + tot_fp + 1e-6)
            rec = tot_tp / (tot_tp + tot_fn + 1e-6)
            f1 = 2 * prec * rec / (prec + rec + 1e-6)
            fpr = tot_fp / (tot_fp + tot_tn + 1e-6)
            
            bin_precs.append(prec)
            bin_recs.append(rec)
            bin_fprs.append(fpr)
            bin_f1s.append(f1)
            
            if f1 > best_f1_val:
                best_f1_val = f1
                best_thresh = thresh
                best_results = (tot_tp, tot_fp, tot_fn, tot_tn, agg_class_stats)

    b_tp, b_fp, b_fn, b_tn, best_classes = best_results
    
    print(f"\n==================================================")
    print(f"       ИТОГИ БИНАРНОЙ ОЦЕНКИ ХОТСПОТОВ (Окно {WINDOW_SIZE}x{WINDOW_SIZE})")
    print(f"==================================================")
    print(f"Оптимальный порог (Best F1 Threshold): {best_thresh:.2f}")
    print(f"Бинарный F1-score: {best_f1_val*100:.2f}%")
    print(f"Бинарная Precision: {b_tp / (b_tp + b_fp + 1e-6)*100:.2f}%")
    print(f"Бинарный Recall (TPR): {b_tp / (b_tp + b_fn + 1e-6)*100:.2f}%")
    print(f"True Positives: {b_tp} | False Positives: {b_fp} | False Negatives: {b_fn}")
    
    print(f"\n--------------------------------------------------")
    print(f"       ПОКЛАССОВЫЙ РАЗРЕЗ (БЕРИ В СТАТЬЮ)")
    print(f"--------------------------------------------------")
    class_names = {1: "Short (Замыкания)", 2: "Spacing (Зазоры)", 3: "MinArea / Other"}
    for c in best_classes:
        tp, fp, fn = best_classes[c]
        p = tp / (tp + fp + 1e-6)
        r = tp / (tp + fn + 1e-6)
        f1_c = 2 * p * r / (p + r + 1e-6)
        print(f" Класс {c} [{class_names.get(c, 'Unknown')}]:")
        print(f"   -> F1: {f1_c*100:.2f}% | Prec: {p*100:.2f}% | Recall: {r*100:.2f}% (TP={tp}, FP={fp}, FN={fn})")
    print(f"==================================================")

if __name__ == "__main__":
    main()