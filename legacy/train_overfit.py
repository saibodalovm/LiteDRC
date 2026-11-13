import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# ============================================================================
# 1. АРХИТЕКТУРА МОДЕЛИ (CenterNetEDAUnet)
# ============================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class CenterNetEDAUnet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, base_filters=32):
        """
        num_classes: Количество классов включая background (0=bg, 1=Spacing, 2=Short)
        """
        super().__init__()
        self.in_conv = DoubleConv(in_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        
        self.up1 = Up(base_filters * 4 + base_filters * 2, base_filters * 2)
        self.up2 = Up(base_filters * 2 + base_filters, base_filters)
        
        # Головы CenterNet
        self.obj_head = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, 1, kernel_size=1)
        )
        self.class_head = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, num_classes, kernel_size=1)
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, 2, kernel_size=1)
        )

    def forward(self, x):
        x_in = self.in_conv(x)
        x_d1 = self.down1(x_in)
        x_d2 = self.down2(x_d1)
        
        x_u1 = self.up1(x_d2, x_d1)
        x_u2 = self.up2(x_u1, x_in)
        
        return {
            "obj": self.obj_head(x_u2),
            "cls": self.class_head(x_u2),
            "offset": self.offset_head(x_u2)
        }

# ============================================================================
# 2. ФУНКЦИЯ ПОТЕРЬ (COMPOSITE LOSS)
# ============================================================================
import torch.nn.functional as F

def stable_focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    pred: logits (batch, 1, H, W)
    target: mask (batch, 1, H, W)
    """
    # 1. Используем встроенную функцию, которая сама считает BCE стабильно
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    
    # 2. Получаем вероятности
    p = torch.sigmoid(pred)
    
    # 3. Считаем факторы фокуса
    # Если target=1 (позитив), weight = (1 - p)^gamma
    # Если target=0 (негатив), weight = p^gamma
    p_t = p * target + (1 - p) * (1 - target)
    alpha_factor = target * alpha + (1 - target) * (1 - alpha)
    modulating_factor = torch.pow(1.0 - p_t, gamma)
    
    # 4. Итоговый лосс
    loss = alpha_factor * modulating_factor * bce_loss
    
    return loss.mean()

# 2. Обновленная основная функция потерь
def compute_composite_loss(preds, targets):
    # Теперь здесь используем Focal Loss для objectness
    loss_obj = stable_focal_loss(preds['obj'], targets['target_objectness'])
    
    pos_mask = targets['target_offset_mask'] > 0
    num_pos = pos_mask.sum()

    loss_cls = torch.tensor(0.0, device=preds['obj'].device)
    loss_offset = torch.tensor(0.0, device=preds['obj'].device)

    if num_pos > 0:
        pred_cls_flat = preds['cls'].permute(0, 2, 3, 1)[pos_mask]
        target_cls_flat = targets['target_class_map'][pos_mask].long()
        loss_cls = F.cross_entropy(pred_cls_flat, target_cls_flat)
        
        pred_offset_flat = preds['offset'].permute(0, 2, 3, 1)[pos_mask]
        target_offset_flat = targets['target_offset'].permute(0, 2, 3, 1)[pos_mask]
        loss_offset = F.l1_loss(pred_offset_flat, target_offset_flat)

    # Веса оставляем те же
    total_loss = loss_obj * 1.0 + loss_cls * 1.0 + loss_offset * 0.5
    return total_loss, loss_obj, loss_cls, loss_offset



# ============================================================================
# 3. ПОДГОТОВКА ДАННЫХ (ОДИН СЭМПЛ)
# ============================================================================
def load_single_sample(npz_path: Path, device: torch.device):
    data = np.load(npz_path)
    print("Уникальные классы в файле:", np.unique(data["target_class_map"]))
    
    # Собираем твои 3 фичи: "золотое трио"
    features = np.stack([
        data["rudy"],
        data["pin_density"],
        data["placement_density"]
    ], axis=0).astype(np.float32) # (3, H, W)
    
    # Переносим таргеты
    target_objectness = data["target_objectness"].astype(np.float32)   # (H, W)
    target_class_map = data["target_class_map"].astype(np.int64)       # (H, W)
    target_offset = data["target_offset"].astype(np.float32)           # (2, H, W)
    target_offset_mask = data["target_offset_mask"].astype(np.float32) # (H, W)
    
    # Оборачиваем в батч размером 1 и кидаем на GPU/CPU
    batch = {
        "features": torch.from_numpy(features).unsqueeze(0).to(device),              # (1, 3, H, W)
        "target_objectness": torch.from_numpy(target_objectness).unsqueeze(0).unsqueeze(0).to(device), # (1, 1, H, W)
        "target_class_map": torch.from_numpy(target_class_map).unsqueeze(0).to(device), # (1, H, W)
        "target_offset": torch.from_numpy(target_offset).unsqueeze(0).to(device),     # (1, 2, H, W)
        "target_offset_mask": torch.from_numpy(target_offset_mask).unsqueeze(0).to(device) # (1, H, W)
    }
    return batch

# ============================================================================
# 4. ЦИКЛ ОБУЧЕНИЯ (OVERFITTING)
# ============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")
    
    # ❗❗❗ УКАЖИ СВОЙ ПУТЬ К ЛЮБОМУ СГЕНЕРИРОВАННОМУ .NPZ ФАЙЛУ С ДЕФЕКТАМИ ❗❗❗
    sample_path = Path(r"J:\processed_dataset\train\sky130_wb_dma_top_CLK_20.7_IO_0.00_CU_20_AR_0.5_HW_4_HS_4_HP_32_VW_4_VS_4_VP_32.npz") 
    
    if not sample_path.exists():
        print(f"Файл {sample_path} не найден! Для демонстрации генерируем случайный тестовый файл...")
        # Генерируем фейковый сэмпл, чтобы код запустился "из коробки" в любом случае
        H, W = 64, 64
        dummy_rudy = np.random.rand(H, W)
        dummy_pin = np.random.rand(H, W)
        dummy_place = np.random.rand(H, W)
        
        target_obj = np.zeros((H, W))
        # Нарисуем пару фейковых DRC
        target_obj[15, 20] = 1.0
        target_obj[40, 45] = 1.0
        
        target_cls = np.zeros((H, W))
        target_cls[15, 20] = 1 # Класс 1
        target_cls[40, 45] = 2 # Класс 2
        
        target_offset = np.zeros((2, H, W))
        target_offset[0, 15, 20] = 0.3
        target_offset[1, 15, 20] = 0.5
        
        target_mask = np.zeros((H, W))
        target_mask[15, 20] = 1
        target_mask[40, 45] = 1
        
        np.savez(
            "dummy_sample.npz",
            rudy=dummy_rudy,
            pin_density=dummy_pin,
            placement_density=dummy_place,
            target_objectness=target_obj,
            target_class_map=target_cls,
            target_offset=target_offset,
            target_offset_mask=target_mask
        )
        sample_path = Path("dummy_sample.npz")

    # Загружаем
    batch = load_single_sample(sample_path, device)
    
    # Классы: background (0) + 2 наших основных класса (Metal Spacing, Short) + others (3)
    # Итого num_classes = 4
    num_classes = 4 
    in_channels = batch["features"].shape[1] # Должно быть 3 (RUDY, Pin, Placement)
    
    model = CenterNetEDAUnet(in_channels=in_channels, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print("\nНачинаем жесткий оверфит на одном дизайне...")
    print("Цель: суммарный лосс должен упасть практически до 0 за 200 эпох.\n")
    
    model.train()
    for epoch in range(1, 201):
        optimizer.zero_grad()
        
        # Forward pass
        preds = model(batch["features"])
        
        # Считаем лосс
        total_loss, l_obj, l_cls, l_offset = compute_composite_loss(preds, batch)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"Эпоха {epoch:3d}/200 | "
                  f"Loss Total: {total_loss.item():.5f} | "
                  f"Obj: {l_obj.item():.5f} | "
                  f"Cls: {l_cls.item():.5f} | "
                  f"Offset: {l_offset.item():.5f}")

    # ============================================================================
    # 5. ПРОВЕРКА РЕЗУЛЬТАТОВ (Inference на том же сэмпле)
    # ============================================================================
    model.eval()
    with torch.no_grad():
        preds = model(batch["features"])
        # Применяем сигмоиду к логам objectness
        pred_obj_probs = torch.sigmoid(preds["obj"]).squeeze().cpu().numpy()
        true_obj = batch["target_objectness"].squeeze().cpu().numpy()
        
        # Извлекаем классы
        pred_cls_ids = torch.argmax(preds["cls"], dim=1).squeeze().cpu().numpy()
        true_cls_ids = batch["target_class_map"].squeeze().cpu().numpy()

    # Находим координаты, где реально были дефекты
    true_y, true_x = np.where(true_obj > 0.5)
    
    print("\n=== РЕЗУЛЬТАТЫ СХОДИМОСТИ ===")
    print(f"Найдено истинных DRC центров в файле: {len(true_y)}")
    for ty, tx in zip(true_y, true_x):
        print(f"Точка ({ty}, {tx}):")
        print(f"  Реальный класс: {true_cls_ids[ty, tx]} | Предсказанный класс: {pred_cls_ids[ty, tx]}")
        print(f"  Реальная вер-ть: {true_obj[ty, tx]:.4f} | Предсказанная вер-ть: {pred_obj_probs[ty, tx]:.4f}")
        
        # Смещения
        true_off = batch["target_offset"].squeeze().cpu().numpy()[:, ty, tx]
        pred_off = preds["offset"].squeeze().cpu().numpy()[:, ty, tx]
        print(f"  Реальный offset: {true_off} | Предсказанный offset: {pred_off}")
        print("-" * 40)
        
    # Удаляем временный dummy файл, если создавали
    if sample_path.name == "dummy_sample.npz" and os.path.exists(sample_path):
        os.remove(sample_path)

if __name__ == "__main__":
    main()