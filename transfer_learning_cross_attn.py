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

from models.EncoderCrossAttDerichle import EvidentialLightweightRegionNet as EvidentialLightweightRegionNetAttn 
from data_processing.patch_dataset import EDAChipDataset

# ============================================================================
# 1. TВОЙ EVIDENTIAL LOSS (BETA-EDL)
# ============================================================================
class BetaEDLLoss(nn.Module):
    def __init__(self, annealing_steps=100):
        super().__init__()
        self.annealing_steps = annealing_steps

    def forward(self, alpha, beta, targets, current_step):
        S = alpha + beta
        probs = alpha / S
        
        loss_err = (targets - probs) ** 2 + (alpha * beta) / (S ** 2 * (S + 1.0))
        
        alpha_tilde = targets + (1 - targets) * alpha
        beta_tilde = (1 - targets) + targets * beta
        S_tilde = alpha_tilde + beta_tilde
        
        kl_loss = torch.lgamma(S_tilde) - torch.lgamma(alpha_tilde) - torch.lgamma(beta_tilde) \
                  + (alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)) \
                  + (beta_tilde - 1) * (torch.digamma(beta_tilde) - torch.digamma(S_tilde))
                  
        annealing_coef = min(1.0, current_step / self.annealing_steps)
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

# ============================================================================
# 4. ОСНОВНОЙ ЦИКЛ FEW-SHOT (ТОЛЬКО CROSS-ATTENTION + FULL FINE-TUNING)
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
    SEED = 42                   # Оставляем только один сид
    MAX_STEPS = 200             # Число обновлений весов
    EVAL_EVERY = 50             # Валидация каждые 50 шагов
    BATCH_SIZE = 128
    
    WEIGHT_PATH = "ckpt/train_gf_true_DERICHLE/best_model_zeroshot_train_gf180_test_sky130_evidential.pth"
    
    # Директория для сохранения новых дообученных весов
    SAVE_CKPT_DIR = "ckpt/traingf180130_DERICHLE_finetuned_sky130"
    os.makedirs(SAVE_CKPT_DIR, exist_ok=True)
    
    print("Загрузка датасетов...")
    train_ds = EDAChipDataset(data_dir=TRAIN_DIR, stats_path=STATS_PATH, patch_size=64, samples_per_file=5)
    test_ds = EDAChipDataset(data_dir=TEST_DIR, stats_path=STATS_PATH, patch_size=64, mode='test')
    
    base_train_indices = [i for i in range(len(train_ds)) if TARGET_PDK in str(train_ds.files[i // 5])]
    test_indices = [i for i in range(len(test_ds)) if TARGET_PDK in str(test_ds.files[test_ds.pdk_grid[i][0]])]
    
    val_loader = DataLoader(Subset(test_ds, test_indices), batch_size=512, shuffle=False, num_workers=2)
    results_data = []

    print("\n" + "="*70)
    print(f"🌱 ЗАПУСК ДООБУЧЕНИЯ ДЛЯ SEED = {SEED} (CrossAttention - Full FT)")
    print("="*70)
    
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    full_train_indices = base_train_indices.copy()
    random.shuffle(full_train_indices)
    
    for frac in FRACTIONS:
        subset_size = max(1, int(len(full_train_indices) * frac))
        current_train_indices = full_train_indices[:subset_size]
        few_shot_loader = DataLoader(Subset(train_ds, current_train_indices), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        
        print(f"\n --- Данные: {frac*100}% ({subset_size} патчей GF180) ---")
        
        model = EvidentialLightweightRegionNetAttn(in_channels=3, num_classes=3, use_film=True).to(device)
            
        try:
            model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
        except FileNotFoundError:
            print(f"  [Ошибка] Базовые веса {WEIGHT_PATH} не найдены. Остановка.")
            return
            
        set_parameter_requires_grad(model, "Full")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
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
            
            preds = model(features, pdk_meta, pdk_id)
            alpha = preds["alpha"]
            beta = preds["beta"]
            
            loss = criterion(alpha, beta, targets, current_step=step)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            step += 1
            
            if step % EVAL_EVERY == 0 or step == MAX_STEPS:
                current_pr = get_pr_auc(model, val_loader, device)
                
                # --- ВАЖНО: СОХРАНЕНИЕ ЛУЧШИХ ВЕСОВ ---
                if current_pr > best_pr_auc:
                    best_pr_auc = current_pr
                    save_path = os.path.join(SAVE_CKPT_DIR, f"best_fft_gf180_frac_{frac}.pth")
                    torch.save(model.state_dict(), save_path)
                    print(f"      [Шаг {step}] Новый лучший PR-AUC: {best_pr_auc:.4f} -> Веса сохранены!")
                
                model.train()
                
        print(f"   Итог для {frac*100}%: Лучший PR-AUC = {best_pr_auc:.4f}")
        results_data.append({
            "Seed": SEED, "Model": "CrossAttention", "Data_Fraction": frac, 
            "Strategy": "Full", "PR_AUC": best_pr_auc
        })

    df = pd.DataFrame(results_data)
    df.to_csv("few_shot_results_Baseline_sky130.csv", index=False)
    print("\n✅ Дообучение завершено! Веса сохранены в папку ckpt/finetuned_gf180/")

if __name__ == "__main__":
    main()