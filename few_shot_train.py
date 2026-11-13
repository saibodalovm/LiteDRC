import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import random
import os
import sys
from sklearn.metrics import average_precision_score

# Добавляем корень проекта в путь
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# Импортируем Evidential архитектуры и датасет
from models.EncoderDerichle import EvidentialLightweightRegionNet
from models.EncoderCrossAttDerichle import EvidentialLightweightRegionNet as EvidentialLightweightRegionNetAttn 
from data_processing.patch_dataset import EDAChipDataset

# ============================================================================
# 1. ТВОЙ ТОЧНЫЙ EVIDENTIAL LOSS (BETA-EDL)
# ============================================================================
class BetaEDLLoss(nn.Module):
    def __init__(self, annealing_steps=100):
        super().__init__()
        self.annealing_steps = annealing_steps

    def forward(self, alpha, beta, targets, current_step):
        # alpha, beta, targets: shape (B, 3, 8, 8)
        S = alpha + beta
        probs = alpha / S
        
        # 1. Ошибка предсказания (MSE + Дисперсия)
        loss_err = (targets - probs) ** 2 + (alpha * beta) / (S ** 2 * (S + 1.0))
        
        # 2. KL-дивергенция (Штраф за ложную уверенность)
        alpha_tilde = targets + (1 - targets) * alpha
        beta_tilde = (1 - targets) + targets * beta
        S_tilde = alpha_tilde + beta_tilde
        
        kl_loss = torch.lgamma(S_tilde) - torch.lgamma(alpha_tilde) - torch.lgamma(beta_tilde) \
                  + (alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)) \
                  + (beta_tilde - 1) * (torch.digamma(beta_tilde) - torch.digamma(S_tilde))
                  
        # Коэффициент включения штрафа по шагам обучения
        annealing_coef = min(1.0, current_step / self.annealing_steps)
        
        # Суммируем лосс по классам (dim=1) и усредняем по батчу/пикселям
        loss = loss_err + annealing_coef * kl_loss
        return loss.sum(dim=1).mean()

# ============================================================================
# 2. ПОДСЧЕТ PR-AUC ДЛЯ EVIDENTIAL МОДЕЛЕЙ
# ============================================================================
def get_pr_auc(model, dataloader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            pdk_meta = batch["pdk_meta"].to(device)
            pdk_id = batch["pdk_id"].to(device)
            
            preds = model(features, pdk_meta, pdk_id)
            probs = preds["probs"].cpu().numpy()
            
            all_preds.append(probs)
            all_targets.append(batch["target_regions"].numpy())
            
    preds_flat = np.concatenate(all_preds).flatten()
    targets_flat = np.concatenate(all_targets).flatten()
    
    if len(np.unique(targets_flat)) > 1:
        return average_precision_score(targets_flat, preds_flat)
    return 0.0

# ============================================================================
# 3. УПРАВЛЕНИЕ ЗАМОРОЗКОЙ СЛОЕВ
# ============================================================================
def set_parameter_requires_grad(model, strategy):
    if strategy == "Full":
        for param in model.parameters(): 
            param.requires_grad = True
    elif strategy == "Frozen":
        unfreeze_keys = ['attn', 'attention', 'film', 'head', 'classifier', 'fc']
        for name, param in model.named_parameters():
            param.requires_grad = any(key in name.lower() for key in unfreeze_keys)

# ============================================================================
# 4. ОСНОВНОЙ ЦИКЛ FEW-SHOT ОБУЧЕНИЯ
# ============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}\n")
    
    # --- НАСТРОЙКИ ---
    TARGET_PDK = "sky130"
    TRAIN_DIR = r"J:\processed_dataset\train"
    TEST_DIR = r"J:\processed_dataset\val"
    STATS_PATH = r"J:\processed_dataset\train\global_zscore_stats.json"
    
    FRACTIONS = [0.01, 0.05, 0.10, 0.25, 1.0]
    SEEDS = [42, 43, 44]        # 3 прогона
    MAX_STEPS = 200             # Число обновлений весов
    EVAL_EVERY = 50             # Валидация каждые 50 шагов
    BATCH_SIZE = 128
    
    WEIGHTS = {
        "No_FiLM": "ckpt/train_gf_true_DERICHLE/best_model_zeroshot_train_gf180_test_sky130_evidential.pth",
        "FiLM": "ckpt/train_gf_true_DERICHLE/best_model_zeroshot_train_gf180_test_sky130_evidential_FiLM.pth",
        "CrossAttention": "ckpt/train_gf_true_DERICHLE/best_model_zeroshot_train_gf180_test_sky130_evidential_attn.pth"
    }
    
    print("Загрузка датасетов...")
    train_ds = EDAChipDataset(data_dir=TRAIN_DIR, stats_path=STATS_PATH, patch_size=64, samples_per_file=5)
    test_ds = EDAChipDataset(data_dir=TEST_DIR, stats_path=STATS_PATH, patch_size=64, mode='test')
    
    base_train_indices = [i for i in range(len(train_ds)) if TARGET_PDK in str(train_ds.files[i // 5])]
    test_indices = [i for i in range(len(test_ds)) if TARGET_PDK in str(test_ds.files[test_ds.pdk_grid[i][0]])]
    
    val_loader = DataLoader(Subset(test_ds, test_indices), batch_size=512, shuffle=False, num_workers=2)
    results_data = []

    for seed in SEEDS:
        print("\n" + "="*70)
        print(f"🌱 ЗАПУСК ДЛЯ SEED = {seed}")
        print("="*70)
        
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        full_train_indices = base_train_indices.copy()
        random.shuffle(full_train_indices)
        
        for model_name, weight_path in WEIGHTS.items():
            print(f"\n🔥 АРХИТЕКТУРА: {model_name.upper()}")
            
            for frac in FRACTIONS:
                subset_size = max(1, int(len(full_train_indices) * frac))
                current_train_indices = full_train_indices[:subset_size]
                few_shot_loader = DataLoader(Subset(train_ds, current_train_indices), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
                
                print(f"   --- Данные: {frac*100}% ({subset_size} патчей) ---")
                
                for strategy in ["Frozen", "Full"]:
                    
                    if model_name == "CrossAttention":
                        model = EvidentialLightweightRegionNetAttn(in_channels=3, num_classes=3, use_film=True).to(device)
                    elif model_name == "FiLM":
                        model = EvidentialLightweightRegionNet(in_channels=3, num_classes=3, use_film=True).to(device)
                    else: # No_FiLM
                        model = EvidentialLightweightRegionNet(in_channels=3, num_classes=3, use_film=False).to(device)
                        
                    try:
                        model.load_state_dict(torch.load(weight_path, map_location=device))
                    except FileNotFoundError:
                        print(f"  [Ошибка] Веса {weight_path} не найдены. Пропуск.")
                        continue
                        
                    set_parameter_requires_grad(model, strategy)
                    
                    lr = 2e-4 if strategy == "Frozen" else 1e-5
                    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-2)
                    criterion = BetaEDLLoss(annealing_steps=100)
                    
                    best_pr_auc = 0.0
                    step = 0
                    train_iter = iter(few_shot_loader)
                    
                    model.train()
                    while step < MAX_STEPS:
                        try:
                            batch = next(train_iter)
                        except StopIteration:
                            train_iter = iter(few_shot_loader)
                            batch = next(train_iter)
                            
                        features = batch["features"].to(device)
                        pdk_meta = batch["pdk_meta"].to(device)
                        pdk_id = batch["pdk_id"].to(device)
                        targets = batch["target_regions"].to(device)
                        
                        optimizer.zero_grad()
                        
                        # Forward pass
                        preds = model(features, pdk_meta, pdk_id)
                        alpha = preds["alpha"]
                        beta = preds["beta"]
                        
                        # Расчет твоего лосса
                        loss = criterion(alpha, beta, targets, current_step=step)
                        loss.backward()
                        
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        step += 1
                        
                        # Валидация
                        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
                            current_pr = get_pr_auc(model, val_loader, device)
                            if current_pr > best_pr_auc:
                                best_pr_auc = current_pr
                            model.train()
                            
                    print(f"      [{strategy}] Лучший PR-AUC (за {MAX_STEPS} шагов): {best_pr_auc:.4f}")
                    results_data.append({
                        "Seed": seed, "Model": model_name, "Data_Fraction": frac, 
                        "Strategy": strategy, "PR_AUC": best_pr_auc
                    })

    # Сохранение результатов
    df = pd.DataFrame(results_data)
    df.to_csv("few_shot_results_evidential_gf_sky.csv", index=False)
    print("\n✅ Все 3 сида завершены! Результаты сохранены в few_shot_results_evidential_sky_gf.csv")

if __name__ == "__main__":
    main()