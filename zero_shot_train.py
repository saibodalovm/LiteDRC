import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.data import Subset
from sklearn.metrics import average_precision_score
# Импортируем наши новые модули
from models.Encoder import LightweightRegionNet
from models.EncoderCrossAtt import LightweightRegionNet as LightweightRegionNetAttn 
from patch_dataset import EDAChipDataset

# ============================================================================
# 1. ЛОСС-ФУНКЦИЯ (Multi-Label Focal Loss)
# ============================================================================
class BinaryFocalLossWithLogits(nn.Module):
    """
    Focal Loss для Multi-Label сегментации.
    Справляется с диким дисбалансом классов (когда 95% макро-окон пустые).
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: сырые логиты (B, 3, 8, 8)
        # targets: бинарные маски (B, 3, 8, 8)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) # Вероятность правильного предсказания
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============================================================================
# 2. МЕТРИКИ (Чистая региональная сетка 8x8)
# ============================================================================
def compute_region_metrics(preds_logits, targets, threshold=0.5):
    # Переводим логиты в вероятности и бинаризуем
    preds_probs = torch.sigmoid(preds_logits)
    preds_bin = (preds_probs > threshold).float()
    targets_bin = targets.float()
    
    # --- ГЛОБАЛЬНЫЕ МЕТРИКИ (Есть ли ЛЮБОЙ дефект в регионе) ---
    pred_any = preds_bin.any(dim=1)    # (B, 8, 8)
    target_any = targets_bin.any(dim=1) # (B, 8, 8)
    
    tp = (pred_any & target_any).sum().item()
    fp = (pred_any & ~target_any).sum().item()
    fn = (~pred_any & target_any).sum().item()
    
    # --- ПОКЛАССОВЫЕ МЕТРИКИ ---
    # Суммируем совпадения по каждому из 3 классов отдельно
    # Размерность результата: (3,)
    cls_tp = (preds_bin * targets_bin).sum(dim=(0, 2, 3)).cpu().numpy()
    cls_fp = (preds_bin * (1 - targets_bin)).sum(dim=(0, 2, 3)).cpu().numpy()
    cls_fn = ((1 - preds_bin) * targets_bin).sum(dim=(0, 2, 3)).cpu().numpy()
    
    return tp, fp, fn, cls_tp, cls_fp, cls_fn


# ============================================================================
# 3. ЦИКЛЫ ОБУЧЕНИЯ И ВАЛИДАЦИИ
# ============================================================================
def train_epoch(model, dataloader, optimizer, criterion, device):
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
        targets = batch["target_regions"].to(device) # (B, 3, 8, 8)
        
        optimizer.zero_grad()
        
        # Forward
        preds = model(features, pdk_meta, pdk_id)
        pred_regions = preds["pred_regions"]
        
        # Loss
        loss = criterion(pred_regions, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        
        # Metrics
        with torch.no_grad():
            tp, fp, fn, c_tp, c_fp, c_fn = compute_region_metrics(pred_regions, targets, threshold=0.3)
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
    
    # Класс 1 (Shorts) F1 - забираем индекс 0 (так как мы отрезали фон)
    short_p = epoch_cls_tp[0] / (epoch_cls_tp[0] + epoch_cls_fp[0] + 1e-6)
    short_r = epoch_cls_tp[0] / (epoch_cls_tp[0] + epoch_cls_fn[0] + 1e-6)
    short_f1 = 2 * short_p * short_r / (short_p + short_r + 1e-6)
        
    return running_loss / len(dataloader), f1, recall, precision, short_f1


def validate(model, dataloader, criterion, device):
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
            pred_regions = preds["pred_regions"]
            
            loss = criterion(pred_regions, targets)
            running_loss += loss.item()
            
            # Собираем вероятности для честного PR-AUC
            preds_probs = torch.sigmoid(pred_regions).cpu().numpy()
            all_preds.append(preds_probs)
            all_targets.append(targets.cpu().numpy())
            
    # Считаем глобальный PR-AUC по всем батчам
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    
    # Защита от пустого батча/класса
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
    # Меняй эти переменные местами для Эксперимента 1 и Эксперимента 2
    TRAIN_PDK = "gf180"   # Оставляем в обучающей выборке ТОЛЬКО этот завод
    TEST_PDK = "sky130"   # Тестируем ТОЛЬКО на этом заводе
    # ----------------------------------------
    
    train_dataset = EDAChipDataset(data_dir=TRAIN_DIR, stats_path="J:/processed_dataset/train/global_zscore_stats.json", patch_size=PATCH_SIZE, samples_per_file=5)
    test_dataset = EDAChipDataset(data_dir=TEST_DIR, stats_path="J:/processed_dataset/train/global_zscore_stats.json", patch_size=PATCH_SIZE, samples_per_file=2)
    
    # Фильтруем списки файлов (предполагаем, что имя завода есть в названии файла/пути)
    train_indices = [
        i for i in range(len(train_dataset)) 
        if TRAIN_PDK in str(train_dataset.files[i // 5])
    ]
    test_indices = [
        i for i in range(len(test_dataset)) 
        if TRAIN_PDK in str(test_dataset.files[i // 2])
    ]

    val_target_indices = [
        i for i in range(len(test_dataset)) 
        if TEST_PDK in str(test_dataset.files[i // 2])
    ]
    
    print(f"\n🚀 ЧЕСТНЫЙ ZERO-SHOT: Учим на {TRAIN_PDK.upper()} -> Тестируем на {TEST_PDK.upper()}")
    print(f"Трейн патчей: {len(train_indices)} | Валидация Source: {len(val_target_indices)} | Тест Target: {len(val_target_indices)}\n")

    # 3. Оборачиваем в Subset
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    val_target_subset = Subset(test_dataset, val_target_indices)

    # 4. Генерируем веса для ВСЕГО датасета, а потом фильтруем только нужные
    all_train_weights = train_dataset.sample_weights(power=0.2, floor=1.0)
    filtered_train_weights = [all_train_weights[i] for i in train_indices]
    
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=filtered_train_weights, num_samples=len(filtered_train_weights), replacement=True
    )
    
    # 5. Передаем в лоадеры наши Subset-ы
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2, pin_memory=True) 
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    val_target_loader = DataLoader(val_target_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # ----------------------------------------------------
    # ИНИЦИАЛИЗАЦИЯ НОВОЙ МОДЕЛИ И ЛОССА
    # ----------------------------------------------------
    # ВНИМАНИЕ: Для честного теста сначала обучи No_FiLM (use_film=False), 
    # а потом CrossAttention (use_film=True), чтобы было с чем сравнивать!
    model = LightweightRegionNetAttn(in_channels=3, num_classes=3, use_film=True).to(device) 
    #model = LightweightRegionNet(in_channels=3, num_classes=3, use_film=True).to(device)
    
    criterion = BinaryFocalLossWithLogits(alpha=0.75, gamma=2.0, reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    best_val_pr_auc = 0.0
    # Динамическое имя для сохранения весов
    save_path = f"ckpt/train_gf_true/best_model_zeroshot_train_{TRAIN_PDK}_test_{TEST_PDK}_attn.pth"
    
    for epoch in range(EPOCHS):
        t_loss, t_f1, t_rec, t_prec, t_short = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss_source, v_pr_source = validate(model, test_loader, criterion, device)
        v_loss_target, v_pr_target = validate(model, val_target_loader, criterion, device)

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