import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Импортируем твои модули
from models.modelFILM import CenterNetEDAUnet
from pixel_approach.pixel_patch_dataset import EDAChipDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

# ============================================================================
# 1. ЛОСС-ФУНКЦИИ
# ============================================================================
def centernet_focal_loss(pred, target, alpha=2.0, beta=4.0):
    """
    Честный Penalty-Reduced Focal Loss из статьи CenterNet.
    Специально разработан для работы с непрерывными Гауссовыми таргетами.
    """
    # pred - это сырые логиты, переводим в вероятности
    pred = torch.sigmoid(pred)
    
    # Зажимаем вероятности, чтобы избежать log(0)
    pred = torch.clamp(pred, min=1e-5, max=1.0 - 1e-5)
    
    # ИСПРАВЛЕНО: Маски для центров (смягчили до 0.95 из-за float32)
    pos_mask = target.ge(0.95).float()
    neg_mask = target.lt(0.95).float()
    
    # Лосс для точных центров
    pos_loss = -torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos_mask
    
    # Лосс для фона и Гауссовых склонов
    neg_loss = -torch.log(1.0 - pred) * torch.pow(pred, alpha) * \
               torch.pow(1.0 - target, beta) * neg_mask
    
    num_pos = pos_mask.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    if num_pos == 0:
        return neg_loss
    return (pos_loss + neg_loss) / num_pos


def compute_composite_loss(preds, targets):
    loss_obj = centernet_focal_loss(preds['obj'], targets['obj'], alpha=2.0, beta=4.0)
    
    pos_mask = targets['mask'].squeeze(1) > 0  # (B, H, W)
    num_pos = pos_mask.sum()

    loss_cls = torch.tensor(0.0, device=preds['obj'].device)
    loss_offset = torch.tensor(0.0, device=preds['obj'].device)

    if num_pos > 0:
        class_weights = torch.tensor([1.0, 1.0, 1.0, 2.0], device=preds['obj'].device)
        
        pred_cls_flat = preds['cls'].permute(0, 2, 3, 1)[pos_mask]
        target_cls_flat = targets['cls'][pos_mask].long()
        loss_cls = F.cross_entropy(pred_cls_flat, target_cls_flat, weight=class_weights)
        
        pred_offset_flat = preds['offset'].permute(0, 2, 3, 1)[pos_mask]
        target_offset_flat = targets['offset'].permute(0, 2, 3, 1)[pos_mask]
        loss_offset = F.l1_loss(pred_offset_flat, target_offset_flat)

    # ИСПРАВЛЕНО: Вернули 0.1 для классификации, чтобы не ломать градиенты детекции
    total_loss = loss_obj * 1.0 + loss_cls * 0.1 + loss_offset * 1.0
    return total_loss, loss_obj, loss_cls, loss_offset



def compute_metrics(preds, targets, obj_threshold=0.05):
    # 1. Достаем вероятности (Sigmoid)
    pred_obj = torch.sigmoid(preds['obj'])
    
    # 2. ПРЕДСКАЗАНИЯ (Peak Extraction): оставляем только пики ("иголки")
    hmax = F.max_pool2d(pred_obj, kernel_size=3, padding=1, stride=1)
    keep_mask = (pred_obj == hmax)
    pred_mask = ((pred_obj > obj_threshold) & keep_mask).squeeze(1) # (B, H, W)
    
    # 3. ПОДСЧЕТ РЕАЛЬНЫХ ОБЪЕКТОВ
    # Считаем количество дефектов как количество пиков в target_objectness
    target_obj = targets['obj']
    t_hmax = F.max_pool2d(target_obj, kernel_size=3, padding=1, stride=1)
    target_peaks = ((target_obj == t_hmax) & (target_obj > 0.5)).squeeze(1)
    total_actual_defects = target_peaks.sum().item()
    
    # 4. МИШЕНИ (Dilation): расширяем маску целей на 1 пиксель (допуск)
    # Если иголка попадет в радиус 3x3 от центра, это засчитается
    target_mask_float = targets['mask'].float()
    target_dilated = F.max_pool2d(target_mask_float, kernel_size=3, padding=1, stride=1).squeeze(1) > 0
    
    # 5. СЧИТАЕМ ГЕОМЕТРИЮ (Попала ли иголка в расширенную мишень?)
    TP = (pred_mask & target_dilated).sum().item()
    FP = (pred_mask & ~target_dilated).sum().item()
    # FN = Все реальные дефекты минус те, что мы нашли
    FN = max(0, total_actual_defects - TP)
    
    # 6. КЛАССИФИКАЦИЯ (Проверка только для успешных попаданий)
    correct_cls = 0
    if TP > 0:
        # Также расширяем классы, чтобы при смещении на 1 пиксель мы считали правильный класс, а не фон (0)
        target_cls_float = targets['cls'].float().unsqueeze(1)
        target_cls_dilated = F.max_pool2d(target_cls_float, kernel_size=3, padding=1, stride=1).squeeze(1).long()
        
        tp_mask = pred_mask & target_dilated
        pred_cls = preds['cls'].argmax(dim=1)
        correct_cls = (pred_cls[tp_mask] == target_cls_dilated[tp_mask]).sum().item()
        
    return TP, FP, FN, correct_cls

# ============================================================================
# 2. ЦИКЛЫ ОБУЧЕНИЯ И ВАЛИДАЦИИ
# ============================================================================
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    
    # Аккумуляторы для абсолютных значений по всей эпохе
    epoch_totals = dict(TP=0, FP=0, FN=0, cls_correct=0)
    
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        preds = model(batch["features"], batch["pdk_meta"], batch["pdk_id"])
        total_loss, _, _, _ = compute_composite_loss(preds, batch)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += total_loss.item()
        
        # Считаем метрики батча через ТВОЮ функцию compute_metrics
        detached_preds = {k: v.detach() for k, v in preds.items()}
        tp, fp, fn, correct_cls = compute_metrics(detached_preds, batch, obj_threshold=0.1)
        
        # Копим абсолютные значения (без старого цикла for)
        epoch_totals['TP'] += tp
        epoch_totals['FP'] += fp
        epoch_totals['FN'] += fn
        epoch_totals['cls_correct'] += correct_cls
            
    # Считаем итоговые метрики за эпоху
    precision = epoch_totals['TP'] / (epoch_totals['TP'] + epoch_totals['FP'] + 1e-6)
    recall = epoch_totals['TP'] / (epoch_totals['TP'] + epoch_totals['FN'] + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    cls_acc = epoch_totals['cls_correct'] / (epoch_totals['TP'] + 1e-6) if epoch_totals['TP'] > 0 else 0.0
        
    return running_loss / len(dataloader), f1, recall, precision, cls_acc


def validate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    epoch_totals = dict(TP=0, FP=0, FN=0, cls_correct=0)
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(batch["features"], batch["pdk_meta"], batch["pdk_id"])
            
            total_loss, _, _, _ = compute_composite_loss(preds, batch)
            running_loss += total_loss.item()
            
            # Считаем метрики вал-батча
            tp, fp, fn, correct_cls = compute_metrics(preds, batch, obj_threshold=0.1)
            
            # Копим абсолютные значения
            epoch_totals['TP'] += tp
            epoch_totals['FP'] += fp
            epoch_totals['FN'] += fn
            epoch_totals['cls_correct'] += correct_cls
                
    precision = epoch_totals['TP'] / (epoch_totals['TP'] + epoch_totals['FP'] + 1e-6)
    recall = epoch_totals['TP'] / (epoch_totals['TP'] + epoch_totals['FN'] + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    cls_acc = epoch_totals['cls_correct'] / (epoch_totals['TP'] + 1e-6) if epoch_totals['TP'] > 0 else 0.0
    
    return running_loss / len(dataloader), f1, recall, precision, cls_acc


def validate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    epoch_totals = dict(TP=0, FP=0, FN=0, cls_correct=0)
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(batch["features"], batch["pdk_meta"], batch["pdk_id"])
            
            total_loss, _, _, _ = compute_composite_loss(preds, batch)
            running_loss += total_loss.item()
            
            # Считаем метрики вал-батча
            tp, fp, fn, correct_cls = compute_metrics(preds, batch, obj_threshold=0.1)
            
            # Копим абсолютные значения
            epoch_totals['TP'] += tp
            epoch_totals['FP'] += fp
            epoch_totals['FN'] += fn
            epoch_totals['cls_correct'] += correct_cls
                
    precision = epoch_totals['TP'] / (epoch_totals['TP'] + epoch_totals['FP'] + 1e-6)
    recall = epoch_totals['TP'] / (epoch_totals['TP'] + epoch_totals['FN'] + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    cls_acc = epoch_totals['cls_correct'] / (epoch_totals['TP'] + 1e-6) if epoch_totals['TP'] > 0 else 0.0
    
    return running_loss / len(dataloader), f1, recall, precision, cls_acc

# ============================================================================
# 3. ЗАПУСК
# ============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")
    
    # ---------------- НАСТРОЙКИ ----------------
    TRAIN_DIR = r"J:\processed_dataset\train"
    TEST_DIR = r"J:\processed_dataset\val"
    
    BATCH_SIZE = 512  # Раз патчи 64x64, можно ставить побольше
    PATCH_SIZE = 64
    EPOCHS = 200
    LR = 0.0003
    # -------------------------------------------
    
    # Инициализация датасетов
    train_dataset = EDAChipDataset(data_dir=TRAIN_DIR, stats_path="J:/processed_dataset/train/global_zscore_stats.json", patch_size=PATCH_SIZE, samples_per_file=5)
    test_dataset = EDAChipDataset(data_dir=TEST_DIR, stats_path="J:/processed_dataset/train/global_zscore_stats.json", patch_size=PATCH_SIZE, samples_per_file=2)
    
    print(f"Тренировочных файлов: {len(train_dataset.files)} | Тестовых файлов: {len(test_dataset.files)}")

    train_weights = train_dataset.sample_weights(power=0.2, floor=1.0)
    
    # Dataloaders
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler,  # Вот она, магия балансировки!
        num_workers=2, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    # Инициализация модели (оставляем дефолтную конфигурацию из твоего класса)
    model = CenterNetEDAUnet(in_channels=3, num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    best_val_f1 = 0.0  # Теперь логично сохраняться по лучшему F1-score!
    
    for epoch in range(EPOCHS):
        # Вызов возвращает 5 значений
        t_loss, t_f1, t_rec, t_prec, t_acc = train_epoch(model, train_loader, optimizer, device)
        v_loss, v_f1, v_rec, v_prec, v_acc = validate(model, test_loader, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Эпоха {epoch:03d}/{EPOCHS} | LR: {current_lr:.6f} | "
              f"Train F1: {t_f1*100:.1f}% | Val F1: {v_f1*100:.1f}%")
        # Информативный принт
        print(f"Эпоха {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {t_loss:.4f} [F1: {t_f1*100:.1f}%] | "
              f"Val Loss: {v_loss:.4f} | "
              f"Val F1: {v_f1*100:.1f}% (Rec: {v_rec*100:.1f}%, Prec: {v_prec*100:.1f}%) | "
              f"Val Cls: {v_acc*100:.1f}%")
        
        # Сохраняем по F1, так как лосс из-за дисбаланса может обманывать
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            torch.save(model.state_dict(), "best_model_f1_file_sampler_FILM.pth")
            print("  --> Модель сохранена по лучшему F1")

if __name__ == "__main__":
    main()