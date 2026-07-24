import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
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

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with skip connection"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Защита от нечетных размеров H и W при конкатенации
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CenterNetEDAUnet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, base_filters=32):
        """
        CenterNet архитектура на базе U-Net для EDA/DRC задач.
        
        Args:
            in_channels: 3 (RUDY, Pin Density, Placement Density)
            num_classes: Количество классов нарушений (включая background)
            base_filters: Базовое число каналов на первом слое
        """
        super().__init__()
        self.in_conv = DoubleConv(in_channels, base_filters)
        
        # Encoder (Сжатие пространственного разрешения)
        self.down1 = Down(base_filters, base_filters * 2)      # 32 -> 64
        self.down2 = Down(base_filters * 2, base_filters * 4)  # 64 -> 128
        self.down3 = Down(base_filters * 4, base_filters * 8)  # 128 -> 256
        
        # Decoder (Восстановление разрешения до исходного)
        self.up1 = Up(base_filters * 8 + base_filters * 4, base_filters * 4) # 256 + 128 -> 128
        self.up2 = Up(base_filters * 4 + base_filters * 2, base_filters * 2) # 128 + 64 -> 64
        self.up3 = Up(base_filters * 2 + base_filters, base_filters)         # 64 + 32 -> 32
        
        # --- ВЫХОДНЫЕ ГОЛОВЫ (CenterNet Heads) ---
        
        # 1. Heatmap (Objectness) Head: предсказание наличия центра дефекта
        self.obj_head = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, 1, kernel_size=1)
        )
        
        # 2. Class Head: классификация типа DRC нарушения
        self.class_head = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, num_classes, kernel_size=1)
        )
        
        # 3. Offset Head: регрессия субпиксельного смещения (dx, dy)
        self.offset_head = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, 2, kernel_size=1)
        )

    def forward(self, x):
        # Проход по энкодеру со сбором skip-connections
        x_in = self.in_conv(x)
        x_d1 = self.down1(x_in)
        x_d2 = self.down2(x_d1)
        x_d3 = self.down3(x_d2)
        
        # Декодер
        x_u1 = self.up1(x_d3, x_d2)
        x_u2 = self.up2(x_u1, x_d1)
        x_u3 = self.up3(x_u2, x_in)
        
        # Генерация предсказаний
        out_obj = self.obj_head(x_u3)
        out_cls = self.class_head(x_u3)
        out_offset = self.offset_head(x_u3)
        
        return {
            "obj": out_obj,       # (B, 1, H, W)
            "cls": out_cls,       # (B, num_classes, H, W)
            "offset": out_offset  # (B, 2, H, W)
        }

if __name__ == "__main__":
    # Проверка на патче 64x64
    dummy_input = torch.randn(2, 3, 64, 64)
    model = CenterNetEDAUnet(in_channels=3, num_classes=4)
    outputs = model(dummy_input)
    
    print("Проверка размерностей для патча 64x64:")
    print("Вход:", dummy_input.shape)
    print("Objectness head:", outputs["obj"].shape)   # [2, 1, 64, 64]
    print("Class head:", outputs["cls"].shape)       # [2, 4, 64, 64]
    # Офсеты должны строго сохранять пространственное разрешение для точной локализации
    print("Offset head:", outputs["offset"].shape)   # [2, 2, 64, 64] 
    print("Модель полностью готова к интеграции с train.py!")