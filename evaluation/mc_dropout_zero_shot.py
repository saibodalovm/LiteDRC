import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve
from torch.utils.data import DataLoader, Subset
import sys
import os

# Получаем абсолютный путь к папке на уровень выше (корень проекта)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Добавляем его в пути Питона
sys.path.append(PROJECT_ROOT)

# Импортируем архитектуры
from models.Encoder import LightweightRegionNet
from models.EncoderCrossAtt import LightweightRegionNet as LightweightRegionNetAttn 
import random


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def enable_mc_dropout(model):
    """
    Включает режим обучения ТОЛЬКО для слоев Dropout,
    оставляя BatchNorm в замороженном режиме (eval).
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def evaluate_model(model, dataloader, device, num_passes=15, rejection_rate=0.15):
    """
    num_passes: количество прогонов Монте-Карло для оценки неопределенности
    rejection_rate: доля предсказаний с наивысшей дисперсией, которые мы отбрасываем (0.15 = 15%)
    """
    model.eval()
    enable_mc_dropout(model) # Включаем MC Dropout
    
    all_mean_preds = []
    all_variances = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            pdk_meta = batch["pdk_meta"].to(device)
            pdk_id = batch["pdk_id"].to(device)
            targets = batch["target_regions"].to(device)
            
            # --- БЛОК MONTE-CARLO DROPOUT ---
            batch_logits = []
            batch_probs = []
            
            for _ in range(num_passes):
                preds_logits = model(features, pdk_meta, pdk_id)["pred_regions"]
                
                batch_logits.append(preds_logits.unsqueeze(0)) 
                batch_probs.append(torch.sigmoid(preds_logits).unsqueeze(0))
                
            # Конкатенируем
            batch_logits = torch.cat(batch_logits, dim=0)
            batch_probs = torch.cat(batch_probs, dim=0)
            
            # Среднее предсказание (для метрик PR-AUC) считаем по вероятностям
            mean_probs = batch_probs.mean(dim=0).cpu().numpy()
            
            # А вот ДИСПЕРСИЮ (неопределенность) считаем по СЫРЫМ ЛОГИТАМ!
            # Это спасет нас от сатурации (схлопывания в 0 или 1)
            variance = batch_logits.var(dim=0).cpu().numpy()
            
            all_mean_preds.append(mean_probs)
            all_variances.append(variance)
            all_targets.append(targets.cpu().numpy())
            
    # Объединяем батчи
    all_preds = np.concatenate(all_mean_preds)
    all_targets = np.concatenate(all_targets)
    all_vars = np.concatenate(all_variances)
    
    # --- ГЛОБАЛЬНЫЕ МЕТРИКИ НА ВСЕХ ДАННЫХ ---
    preds_flat = all_preds.flatten()
    targets_flat = all_targets.flatten()
    vars_flat = all_vars.flatten()
    
    if len(np.unique(targets_flat)) > 1:
        roc_auc = roc_auc_score(targets_flat, preds_flat)
        global_pr_auc = average_precision_score(targets_flat, preds_flat)
        
        # Динамический поиск лучшего F1
        precisions, recalls, thresholds = precision_recall_curve(targets_flat, preds_flat)
        f1_scores = np.divide(2 * precisions * recalls, precisions + recalls, 
                              out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
        best_idx = np.argmax(f1_scores)
        global_f1 = f1_scores[best_idx] 
    else:
        roc_auc, global_pr_auc, global_f1 = 0.0, 0.0, 0.0
        
    # --- МЕТРИКИ С УЧЕТОМ ОТКАЗА ОТ НЕУВЕРЕННЫХ (REJECTION) ---
    # Находим порог дисперсии, отсекающий верхние `rejection_rate` процентов
    preds_binary = preds_flat > 0.5
    
    tp_mask = (preds_binary == 1) & (targets_flat == 1) # True Positives
    fp_mask = (preds_binary == 1) & (targets_flat == 0) # False Positives
    fn_mask = (preds_binary == 0) & (targets_flat == 1) # False Negatives
    
    mean_var_tp = vars_flat[tp_mask].mean() if tp_mask.sum() > 0 else 0.0
    mean_var_fp = vars_flat[fp_mask].mean() if fp_mask.sum() > 0 else 0.0
    mean_var_fn = vars_flat[fn_mask].mean() if fn_mask.sum() > 0 else 0.0
    
    print(f"\n[Дебаг Uncertainty - {model.__class__.__name__}]")
    print(f"  Средняя дисперсия на True Positives (Угадали дефект):  {mean_var_tp:.5f}")
    print(f"  Средняя дисперсия на False Positives (Ложная тревога): {mean_var_fp:.5f}")
    print(f"  Средняя дисперсия на False Negatives (Пропустили):     {mean_var_fn:.5f}")
    
    conf_pr_auc = 0.0 # Пока заглушка, чтобы не ломать return
        
    # --- ПОКЛАССОВЫЕ МЕТРИКИ PR-AUC ---
    class_pr_aucs = []
    for c in range(3):
        t_c = all_targets[:, c, :, :].flatten()
        p_c = all_preds[:, c, :, :].flatten()
        
        if len(np.unique(t_c)) > 1:
            class_pr_aucs.append(average_precision_score(t_c, p_c))
        else:
            class_pr_aucs.append(0.0)
            
    # Возвращаем расширенный набор метрик
    return roc_auc, global_pr_auc, conf_pr_auc, global_f1, class_pr_aucs[1], class_pr_aucs[0], class_pr_aucs[2]


def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}\n")
    
    TEST_DIR = r"J:\processed_dataset\val"
    TEST_PDK = "sky130" 
    PATCH_SIZE = 64
    BATCH_SIZE = 512
    # Импорт перенесен сюда, чтобы не ломался путь, если скрипт запущен не из той директории
    from data_processing.patch_dataset import EDAChipDataset
    
    WEIGHTS = {
        "No_FiLM": "ckpt/train_sky_true/best_model_zeroshot_train_sky130_test_gf180.pth",
        "FiLM": "ckpt/train_sky_true/best_model_zeroshot_train_sky130_test_gf180_FiLM.pth",
        "CrossAttention": "ckpt/train_sky_true/best_model_zeroshot_train_sky130_test_gf180_attn.pth"
    }
    
    print("Загрузка и фильтрация датасета...")
    dataset = EDAChipDataset(data_dir=TEST_DIR, stats_path="J:/processed_dataset/train/global_zscore_stats.json", patch_size=PATCH_SIZE, samples_per_file=2, mode='test')
    
    # Оставляем только файлы целевого завода
    test_indices = [i for i in range(len(dataset)) if TEST_PDK in str(dataset.files[dataset.pdk_grid[i][0]])]
    test_subset = Subset(dataset, test_indices)
    
    dataloader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Тестовых патчей {TEST_PDK.upper()}: {len(test_indices)}\n")
    
    results = []
    
    for model_name, weight_path in WEIGHTS.items():
        print(f"Оценка архитектуры: {model_name}...")
        
        # Инициализация нужной архитектуры
        if model_name == "CrossAttention":
            model = LightweightRegionNetAttn(in_channels=3, num_classes=3, use_film=True).to(device)
        elif model_name == "FiLM":
            model = LightweightRegionNet(in_channels=3, num_classes=3, use_film=True).to(device)
        else: # No_FiLM
            model = LightweightRegionNet(in_channels=3, num_classes=3, use_film=False).to(device)
            
        # Загружаем веса
        try:
            model.load_state_dict(torch.load(weight_path, map_location=device))
        except FileNotFoundError:
            print(f"  [Ошибка] Файл {weight_path} не найден. Пропуск.")
            continue
            
        roc_auc, pr_auc, conf_pr_auc, f1, m_spac, short_pr, others_pr = evaluate_model(model, dataloader, device, num_passes=15, rejection_rate=0.15)
        results.append((model_name, TEST_PDK, roc_auc, pr_auc, conf_pr_auc, f1, m_spac, short_pr, others_pr))
        
    # --- ВЫВОД КРАСИВОЙ ТАБЛИЦЫ ---
    print("\n" + "="*118 + " СВОДНАЯ ТАБЛИЦА МЕТРИК ZERO-SHOT (MC DROPOUT) " + "="*118)
    print(f"{'Model':>16} {'PDK / Scope':>13} {'ROC-AUC':>10} {'PR-AUC (All)':>14} {'PR-AUC (Top 85%)':>18} {'Global F1':>11} {'Metal Spacing PR':>20} {'Short PR':>12} {'Others PR':>12}")
    for r in results:
        print(f"{r[0]:>16} {r[1]:>13} {r[2]:10.4f} {r[3]:14.4f} {r[4]:18.4f} {r[5]*100:10.2f}% {r[6]:20.4f} {r[7]:12.4f} {r[8]:12.4f}")
    print("="*280 + "\n")

if __name__ == "__main__":
    main()