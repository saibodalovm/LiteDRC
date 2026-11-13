import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.data import Subset
from sklearn.metrics import average_precision_score

# Убедись, что импортируешь именно новые Evidential-модели!
from models.EncoderDerichle import EvidentialLightweightRegionNet
from models.EncoderCrossAttDerichle import EvidentialLightweightRegionNet as EvidentialLightweightRegionNetAttn
from data_processing.patch_dataset import EDAChipDataset

# ============================================================================
# 1. ЛОСС-ФУНКЦИЯ (Evidential Deep Learning Loss)
# ============================================================================
class BetaEDLLoss(nn.Module):
    def __init__(self, annealing_epochs=50):
        super().__init__()
        self.annealing_epochs = annealing_epochs

    def forward(self, alpha, beta, targets, current_epoch):
        # alpha, beta, targets: shape (B, 3, 8, 8)
        
        S = alpha + beta
        probs = alpha / S
        
        # 1. Ошибка предсказания (MSE + Дисперсия)
        loss_err = (targets - probs) ** 2 + (alpha * beta) / (S ** 2 * (S + 1.0))
        
        # 2. KL-дивергенция (Штраф за ложную уверенность)
        # Если target=1, штрафуем beta (ложные доказательства отсутствия). alpha_tilde = 1
        # Если target=0, штрафуем alpha (ложные доказательства наличия). beta_tilde = 1
        alpha_tilde = targets + (1 - targets) * alpha
        beta_tilde = (1 - targets) + targets * beta
        S_tilde = alpha_tilde + beta_tilde
        
        kl_loss = torch.lgamma(S_tilde) - torch.lgamma(alpha_tilde) - torch.lgamma(beta_tilde) \
                  + (alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)) \
                  + (beta_tilde - 1) * (torch.digamma(beta_tilde) - torch.digamma(S_tilde))
                  
        # Коэффициент включения штрафа
        annealing_coef = min(1.0, current_epoch / self.annealing_epochs)
        
        # Суммируем лосс по классам (dim=1) и усредняем по батчу/пикселям
        loss = loss_err + annealing_coef * kl_loss
        return loss.sum(dim=1).mean()


# ============================================================================
# 2. МЕТРИКИ (Чистая региональная сетка 8x8)
# ============================================================================
def compute_region_metrics(preds_probs, targets, threshold=0.5):
    # ВНИМАНИЕ: sigmoid больше не нужен, на входе уже готовые вероятности Дирихле!
    preds_bin = (preds_probs > threshold).float()
    targets_bin = targets.float()
    
    # --- ГЛОБАЛЬНЫЕ МЕТРИКИ (Есть ли ЛЮБОЙ дефект в регионе) ---
    pred_any = preds_bin.any(dim=1)    # (B, 8, 8)
    target_any = targets_bin.any(dim=1) # (B, 8, 8)
    
    tp = (pred_any & target_any).sum().item()
    fp = (pred_any & ~target_any).sum().item()
    fn = (~pred_any & target_any).sum().item()
    
    # --- ПОКЛАССОВЫЕ МЕТРИКИ ---
    cls_tp = (preds_bin * targets_bin).sum(dim=(0, 2, 3)).cpu().numpy()
    cls_fp = (preds_bin * (1 - targets_bin)).sum(dim=(0, 2, 3)).cpu().numpy()
    cls_fn = ((1 - preds_bin) * targets_bin).sum(dim=(0, 2, 3)).cpu().numpy()
    
    return tp, fp, fn, cls_tp, cls_fp, cls_fn


# ============================================================================
# 3. ЦИКЛЫ ОБУЧЕНИЯ И ВАЛИДАЦИИ
# ============================================================================
def train_epoch(model, dataloader, optimizer, criterion, device, current_epoch):
    model.train()
    running_loss = 0.0
    
    # Аккумуляторы
    epoch_tp, epoch_fp, epoch_fn = 0, 0, 0
    epoch_cls_tp = np.zeros(3)
    epoch_cls_fp = np.zeros(3)
    epoch_cls_fn = np.zeros(3)
    
    for batch in dataloader:
        features = batch["features"].to(device)
        pdk_meta = batch["pdk_meta"].to(device)
        pdk_id = batch["pdk_id"].to(device)
        targets = batch["target_regions"].to(device) 
        
        optimizer.zero_grad()
        
        # Forward (Модель теперь возвращает словарь с alpha, probs и uncertainty)
        preds = model(features, pdk_meta, pdk_id)
        alpha = preds["alpha"]
        beta = preds["beta"]  # <--- ДОБАВИЛИ BETA
        probs = preds["probs"]
        
        # Loss (Передаем alpha, beta, таргеты и текущую эпоху)
        loss = criterion(alpha, beta, targets, current_epoch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        
        # Metrics (Порог 0.3 можно подстроить при необходимости)
        with torch.no_grad():
            tp, fp, fn, c_tp, c_fp, c_fn = compute_region_metrics(probs, targets, threshold=0.3)
            epoch_tp += tp
            epoch_fp += fp
            epoch_fn += fn
            epoch_cls_tp += c_tp
            epoch_cls_fp += c_fp
            epoch_cls_fn += c_fn
            
    # Глобальный F1
    precision = epoch_tp / (epoch_tp + epoch_fp + 1e-6)
    recall = epoch_tp / (epoch_tp + epoch_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    # Класс 1 (Shorts) F1 - индекс 0
    short_p = epoch_cls_tp[0] / (epoch_cls_tp[0] + epoch_cls_fp[0] + 1e-6)
    short_r = epoch_cls_tp[0] / (epoch_cls_tp[0] + epoch_cls_fn[0] + 1e-6)
    short_f1 = 2 * short_p * short_r / (short_p + short_r + 1e-6)
        
    return running_loss / len(dataloader), f1, recall, precision, short_f1


def validate(model, dataloader, criterion, device, max_epoch=200):
    model.eval()
    running_loss = 0.0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            pdk_meta = batch["pdk_meta"].to(device)
            pdk_id = batch["pdk_id"].to(device)
            targets = batch["target_regions"].to(device)
            
            preds = model(features, pdk_meta, pdk_id)
            alpha = preds["alpha"]
            beta = preds["beta"]  # <--- ДОБАВИЛИ BETA
            probs = preds["probs"]
            
            # На валидации всегда считаем лосс с максимальным KL-штрафом
            loss = criterion(alpha, beta, targets, max_epoch)
            running_loss += loss.item()
            
            # Собираем вероятности для честного PR-AUC
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    # Считаем глобальный PR-AUC
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    
    if len(np.unique(all_targets)) > 1:
        val_pr_auc = average_precision_score(all_targets, all_preds)
    else:
        val_pr_auc = 0.0

    return running_loss / len(dataloader), val_pr_auc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")
    
    TRAIN_DIR = r"J:\processed_dataset\train"
    TEST_DIR = r"J:\processed_dataset\val"
    
    BATCH_SIZE = 512 
    PATCH_SIZE = 64
    EPOCHS = 200
    LR = 0.0002  
    
    # --- НАСТРОЙКИ ZERO-SHOT ЭКСПЕРИМЕНТА ---
    TRAIN_PDK = "sky130"   
    TEST_PDK = "gf180"   
    # ----------------------------------------
    
    train_dataset = EDAChipDataset(data_dir=TRAIN_DIR, stats_path="J:/processed_dataset/train/global_zscore_stats.json", patch_size=PATCH_SIZE, samples_per_file=5)
    test_dataset = EDAChipDataset(data_dir=TEST_DIR, stats_path="J:/processed_dataset/train/global_zscore_stats.json", patch_size=PATCH_SIZE, samples_per_file=2)
    
    train_indices = [i for i in range(len(train_dataset)) if TRAIN_PDK in str(train_dataset.files[i // 5])]
    test_indices = [i for i in range(len(test_dataset)) if TRAIN_PDK in str(test_dataset.files[i // 2])]
    val_target_indices = [i for i in range(len(test_dataset)) if TEST_PDK in str(test_dataset.files[i // 2])]
    
    print(f"\n🚀 ЧЕСТНЫЙ ZERO-SHOT (EVIDENTIAL): Учим на {TRAIN_PDK.upper()} -> Тестируем на {TEST_PDK.upper()}")
    print(f"Трейн патчей: {len(train_indices)} | Валидация Source: {len(test_indices)} | Тест Target: {len(val_target_indices)}\n")

    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    val_target_subset = Subset(test_dataset, val_target_indices)

    all_train_weights = train_dataset.sample_weights(power=0.2, floor=1.0)
    filtered_train_weights = [all_train_weights[i] for i in train_indices]
    
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=filtered_train_weights, num_samples=len(filtered_train_weights), replacement=True
    )
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2, pin_memory=True) 
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    val_target_loader = DataLoader(val_target_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # ----------------------------------------------------
    # ИНИЦИАЛИЗАЦИЯ EVIDENTIAL-МОДЕЛИ И ЛОССА
    # ----------------------------------------------------
    model = EvidentialLightweightRegionNetAttn(in_channels=3, num_classes=3, use_film=True).to(device) 
    #model = EvidentialLightweightRegionNet(in_channels=3, num_classes=3, use_film=True).to(device) 
    
    criterion = BetaEDLLoss(annealing_epochs=50)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    best_val_pr_auc = 0.0
    save_path = f"ckpt/train_sky_true_DERICHLE/best_model_zeroshot_train_{TRAIN_PDK}_test_{TEST_PDK}_evidential_attn.pth"
    
    for epoch in range(EPOCHS):
        t_loss, t_f1, t_rec, t_prec, t_short = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        v_loss_source, v_pr_source = validate(model, test_loader, criterion, device, max_epoch=EPOCHS)
        v_loss_target, v_pr_target = validate(model, val_target_loader, criterion, device, max_epoch=EPOCHS)

        scheduler.step()
        
        print(f"Эпоха {epoch+1:03d}/{EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  [Train] Loss: {t_loss:.4f} | Global F1: {t_f1*100:.1f}%")
        print(f"  [Val SOURCE ({TRAIN_PDK})] PR-AUC: {v_pr_source:.4f}  <-- ORACLE")
        print(f"  [Val TARGET ({TEST_PDK})] PR-AUC: {v_pr_target:.4f}  <-- ZERO-SHOT")
        
        if v_pr_source > best_val_pr_auc:
            best_val_pr_auc = v_pr_source
            torch.save(model.state_dict(), save_path)
            print(f"  --> [СОХРАНЕНО] Новый лучший Oracle PR-AUC: {v_pr_source:.4f} (в этот момент Zero-Shot = {v_pr_target:.4f})")
        print("-" * 60)

if __name__ == "__main__":
    main()