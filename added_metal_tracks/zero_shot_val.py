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
from added_metal_tracks.patch_dataset import EDAChipDataset
import random


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            pdk_meta = batch["pdk_meta"].to(device)
            pdk_id = batch["pdk_id"].to(device)
            targets = batch["target_regions"].to(device)
            
            # Forward
            preds_logits = model(features, pdk_meta, pdk_id)["pred_regions"]
            preds_probs = torch.sigmoid(preds_logits).cpu().numpy()
            
            all_preds.append(preds_probs)
            all_targets.append(targets.cpu().numpy())
            
    # Объединяем батчи (B, 3, 8, 8)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # --- ГЛОБАЛЬНЫЕ МЕТРИКИ ---
    preds_flat = all_preds.flatten()
    targets_flat = all_targets.flatten()
    
    if len(np.unique(targets_flat)) > 1:
        roc_auc = roc_auc_score(targets_flat, preds_flat)
        global_pr_auc = average_precision_score(targets_flat, preds_flat)
        
        # --- ДИНАМИЧЕСКИЙ ПОИСК ЛУЧШЕГО F1 ---
        precisions, recalls, thresholds = precision_recall_curve(targets_flat, preds_flat)
        f1_scores = np.divide(2 * precisions * recalls, precisions + recalls, 
                              out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
        best_idx = np.argmax(f1_scores)
        global_f1 = f1_scores[best_idx] 
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        # ---------------------------------------
    else:
        roc_auc, global_pr_auc, global_f1, best_threshold = 0.0, 0.0, 0.0, 0.5
        
    
    # --- ПОКЛАССОВЫЕ МЕТРИКИ PR-AUC ---
    # Индекс 0: Short, Индекс 1: Metal Spacing, Индекс 2: Others
    class_pr_aucs = []
    for c in range(3):
        t_c = all_targets[:, c, :, :].flatten()
        p_c = all_preds[:, c, :, :].flatten()
        
        if len(np.unique(t_c)) > 1:
            class_pr_aucs.append(average_precision_score(t_c, p_c))
        else:
            class_pr_aucs.append(0.0)
            
    # Возвращаем: ROC-AUC, Global PR-AUC, Global F1, Spacing PR, Short PR, Others PR
    return roc_auc, global_pr_auc, global_f1, class_pr_aucs[1], class_pr_aucs[0], class_pr_aucs[2]


def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}\n")
    
    TEST_DIR = r"J:\processed_dataset\val"
    TEST_PDK = "sky130" 
    PATCH_SIZE = 64
    BATCH_SIZE = 512
    
    WEIGHTS = {
        "No_FiLM": "ckpt/train_sky_true_DISTANCE_MAPS/best_model_zeroshot_train_sky130_test_gf180.pth",
        "FiLM": "ckpt/train_sky_true_DISTANCE_MAPS/best_model_zeroshot_train_sky130_test_gf180_FiLM.pth",
        "CrossAttention": "ckpt/train_sky_true_DISTANCE_MAPS/best_model_zeroshot_train_sky130_test_gf180_attn.pth"
    }
    
    print("Загрузка и фильтрация датасета...")
    dataset = EDAChipDataset(data_dir=TEST_DIR, stats_path="J:/processed_dataset/train/global_zscore_stats.json", patch_size=PATCH_SIZE, samples_per_file=2, mode='test', use_distances=True)
    
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
            model = LightweightRegionNetAttn(in_channels=5, num_classes=3, use_film=True).to(device)
        elif model_name == "FiLM":
            model = LightweightRegionNet(in_channels=5, num_classes=3, use_film=True).to(device)
        else: # No_FiLM
            model = LightweightRegionNet(in_channels=5, num_classes=3, use_film=False).to(device)
            
        # Загружаем веса
        try:
            model.load_state_dict(torch.load(weight_path, map_location=device))
        except FileNotFoundError:
            print(f"  [Ошибка] Файл {weight_path} не найден. Пропуск.")
            continue
            
        roc_auc, pr_auc, f1, m_spac, short_pr, others_pr = evaluate_model(model, dataloader, device)
        results.append((model_name, TEST_PDK, roc_auc, pr_auc, f1, m_spac, short_pr, others_pr))
        
    # --- ВЫВОД КРАСИВОЙ ТАБЛИЦЫ ---
    print("\n" + "="*102 + " СВОДНАЯ ТАБЛИЦА МЕТРИК ZERO-SHOT " + "="*102)
    print(f"{'Model':>16} {'PDK / Scope':>13} {'ROC-AUC':>10} {'Global PR-AUC':>15} {'Global F1':>11} {'Metal Spacing PR-AUC':>22} {'Short PR-AUC':>14} {'Others PR-AUC':>15}")
    for r in results:
        print(f"{r[0]:>16} {r[1]:>13} {r[2]:10.4f} {r[3]:15.4f} {r[4]*100:10.2f}% {r[5]:22.4f} {r[6]:14.4f} {r[7]:15.4f}")
    print("="*238 + "\n")

if __name__ == "__main__":
    main()