import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import average_precision_score

# Импортируем твои модули
from models.EncoderCrossAtt import LightweightRegionNet as LightweightRegionNetAttn 
from models.Encoder import LightweightRegionNet
from patch_dataset import EDAChipDataset

def get_pr_auc(model, dataloader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in dataloader:
            preds = model(batch["features"].to(device), batch["pdk_meta"].to(device), batch["pdk_id"].to(device))["pred_regions"]
            all_preds.append(torch.sigmoid(preds).cpu().numpy())
            all_targets.append(batch["target_regions"].numpy())
            
    preds_flat = np.concatenate(all_preds).flatten()
    targets_flat = np.concatenate(all_targets).flatten()
    
    if len(np.unique(targets_flat)) > 1:
        return average_precision_score(targets_flat, preds_flat)
    return 0.0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")
    
    # --- НАСТРОЙКИ ---
    TARGET_PDK = "gf180" # ВАЖНО: Тестируем на GF180!
    TEST_DIR = r"J:\processed_dataset\val"
    STATS_PATH = r"J:\processed_dataset\train\global_zscore_stats.json"
    
    # Путь к твоей лучшей модели CrossAttention, обученной на GF180
    WEIGHT_PATH = "ckpt/train_gf_true/best_model_zeroshot_train_gf180_test_sky130_FiLM.pth"
    
    print(f"Загрузка тестового датасета из {TEST_DIR}...")
    # Обязательно mode='test' для честного скользящего окна по всему чипу
    test_ds = EDAChipDataset(data_dir=TEST_DIR, stats_path=STATS_PATH, patch_size=64, mode='test')
    
    # Отфильтровываем только файлы GF180
    test_indices = [i for i in range(len(test_ds)) if TARGET_PDK in str(test_ds.files[test_ds.pdk_grid[i][0]])]
    
    if len(test_indices) == 0:
        print(f"❌ ОШИБКА: В папке {TEST_DIR} не найдено файлов для PDK {TARGET_PDK}!")
        return
        
    val_loader = DataLoader(Subset(test_ds, test_indices), batch_size=512, shuffle=False, num_workers=2)
    print(f"Найдено {len(test_indices)} патчей макетов GF180 для тестирования.\n")
    
    # Инициализация и загрузка модели
    print("Загрузка весов модели CrossAttention...")
    #model = LightweightRegionNetAttn(in_channels=3, num_classes=3, use_film=True).to(device)
    model = LightweightRegionNet(in_channels=3, num_classes=3, use_film=True).to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    
    # Запуск оценки
    print("Запуск инференса (Oracle Evaluation)...")
    oracle_pr_auc = get_pr_auc(model, val_loader, device)
    
    print("\n" + "="*50)
    print(f"🏆 ORACLE PR-AUC (GF180 -> GF180): {oracle_pr_auc:.4f}")
    print("="*50)
    print("Теперь вставь эту цифру в переменную HOME_DOMAIN_PR_AUC в скрипте графиков!")

if __name__ == "__main__":
    main()