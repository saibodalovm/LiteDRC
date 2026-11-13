import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import random
import os
from sklearn.metrics import average_precision_score

# Импортируем архитектуры и датасет
from models.Encoder import LightweightRegionNet
from models.EncoderCrossAtt import LightweightRegionNet as LightweightRegionNetAttn 
from data_processing.patch_dataset import EDAChipDataset

# ============================================================================
# 1. ЛОСС И МЕТРИКИ
# ============================================================================
class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()

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

# ============================================================================
# 2. УПРАВЛЕНИЕ ВЕСАМИ
# ============================================================================
def set_parameter_requires_grad(model, strategy):
    if strategy == "Full":
        for param in model.parameters(): param.requires_grad = True
    elif strategy == "Frozen":
        unfreeze_keys = ['attn', 'attention', 'film', 'head', 'classifier', 'fc']
        for name, param in model.named_parameters():
            param.requires_grad = any(key in name.lower() for key in unfreeze_keys)

# ============================================================================
# 3. ОСНОВНОЙ ЦИКЛ
# ============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}\n")
    
    # --- НАСТРОЙКИ ---
    TARGET_PDK = "gf180"
    TRAIN_DIR = r"J:\processed_dataset\train"
    TEST_DIR = r"J:\processed_dataset\val"
    STATS_PATH = r"J:\processed_dataset\train\global_zscore_stats.json"
    
    FRACTIONS = [0.01, 0.05, 0.10, 0.25, 1.0]
    SEEDS = [42, 43, 44]        # 3 прогона для честности!
    MAX_STEPS = 200             # Фиксированное число обновлений весов
    EVAL_EVERY = 50             # Оценка каждые 50 шагов
    BATCH_SIZE = 128
    
    WEIGHTS = {
        "No_FiLM": "ckpt/train_sky_true/best_model_zeroshot_train_sky130_test_gf180.pth",
        "FiLM": "ckpt/train_sky_true/best_model_zeroshot_train_sky130_test_gf180_FiLM.pth",
        "CrossAttention": "ckpt/train_sky_true/best_model_zeroshot_train_sky130_test_gf180_attn.pth"
    }
    
    print("Загрузка датасетов...")
    train_ds = EDAChipDataset(data_dir=TRAIN_DIR, stats_path=STATS_PATH, patch_size=64, samples_per_file=5)
    
    # ВАЖНО: Включен режим 'test' для честной оценки по всему чипу
    test_ds = EDAChipDataset(data_dir=TEST_DIR, stats_path=STATS_PATH, patch_size=64, mode='test')
    
    # Индексы полного трейна (samples_per_file=5)
    base_train_indices = [i for i in range(len(train_ds)) if TARGET_PDK in str(train_ds.files[i // 5])]
    # Индексы теста (mode='test', берем по сетке)
    test_indices = [i for i in range(len(test_ds)) if TARGET_PDK in str(test_ds.files[test_ds.pdk_grid[i][0]])]
    
    val_loader = DataLoader(Subset(test_ds, test_indices), batch_size=512, shuffle=False, num_workers=2)
    results_data = []

    for seed in SEEDS:
        print("\n" + "="*70)
        print(f"🌱 ЗАПУСК ДЛЯ SEED = {seed}")
        print("="*70)
        
        # Фиксируем рандом для воспроизводимости и честной нарезки
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Перемешиваем трейн для текущего сида
        full_train_indices = base_train_indices.copy()
        random.shuffle(full_train_indices)
        
        for model_name, weight_path in WEIGHTS.items():
            print(f"\n🔥 АРХИТЕКТУРА: {model_name.upper()}")
            
            for frac in FRACTIONS:
                subset_size = max(1, int(len(full_train_indices) * frac))
                current_train_indices = full_train_indices[:subset_size]
                few_shot_loader = DataLoader(Subset(train_ds, current_train_indices), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
                
                print(f"  --- Данные: {frac*100}% ({subset_size} патчей) ---")
                
                for strategy in ["Frozen", "Full"]:
                    
                    # === ИСПРАВЛЕННАЯ ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ ===
                    if model_name == "CrossAttention":
                        model = LightweightRegionNetAttn(in_channels=3, num_classes=3, use_film=True).to(device)
                    elif model_name == "FiLM":
                        model = LightweightRegionNet(in_channels=3, num_classes=3, use_film=True).to(device)
                    else: # No_FiLM
                        model = LightweightRegionNet(in_channels=3, num_classes=3, use_film=False).to(device)
                    # ==========================================
                        
                    model.load_state_dict(torch.load(weight_path, map_location=device))
                    set_parameter_requires_grad(model, strategy)
                    
                    lr = 2e-4 if strategy == "Frozen" else 1e-5
                    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-2)
                    criterion = BinaryFocalLossWithLogits()
                    
                    best_pr_auc = 0.0
                    step = 0
                    train_iter = iter(few_shot_loader)
                    
                    # ОБУЧЕНИЕ ПО ШАГАМ (STEPS), А НЕ ЭПОХАМ
                    model.train()
                    while step < MAX_STEPS:
                        try:
                            batch = next(train_iter)
                        except StopIteration:
                            # Данные закончились - перезапускаем итератор (эмуляция следующей эпохи)
                            train_iter = iter(few_shot_loader)
                            batch = next(train_iter)
                            
                        preds = model(batch["features"].to(device), batch["pdk_meta"].to(device), batch["pdk_id"].to(device))["pred_regions"]
                        loss = criterion(preds, batch["target_regions"].to(device))
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        step += 1
                        
                        # ВАЛИДАЦИЯ
                        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
                            current_pr = get_pr_auc(model, val_loader, device)
                            if current_pr > best_pr_auc:
                                best_pr_auc = current_pr
                            model.train() # Возвращаем в train mode после eval
                            
                    print(f"      [{strategy}] Лучший PR-AUC (за {MAX_STEPS} шагов): {best_pr_auc:.4f}")
                    results_data.append({
                        "Seed": seed, "Model": model_name, "Data_Fraction": frac, 
                        "Strategy": strategy, "PR_AUC": best_pr_auc
                    })

    # Сохраняем финальную таблицу
    df = pd.DataFrame(results_data)
    df.to_csv("few_shot_results_3models_sky_gf.csv", index=False)
    print("\n✅ Все сиды завершены. Результаты сохранены в few_shot_results_3models.csv")

if __name__ == "__main__":
    main()