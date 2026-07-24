import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    """
    Модуль Feature-wise Linear Modulation.
    Генерирует масштаб (gamma) и сдвиг (beta) из вектора PDK.
    """
    def __init__(self, condition_dim, feature_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, feature_dim * 2)
        )
        
        # Трюк: инициализируем нулями последний слой, чтобы стартовать с identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, condition):
        film_params = self.mlp(condition) # (B, 2*C)
        gamma, beta = torch.chunk(film_params, 2, dim=1) # Каждая по (B, C)
        
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        
        return x * (1 + gamma) + beta


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU => [Dropout2d]) => (convolution => [BN] => ReLU)"""
    def __init__(self, in_channels, out_channels, drop_prob=0.1): # ---> DROPOUT (добавили drop_prob)
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_prob), # ---> DROPOUT: Spatial Dropout между свертками
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, drop_prob=0.1): # ---> DROPOUT
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, drop_prob=drop_prob) # ---> DROPOUT
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with skip connection"""
    def __init__(self, in_channels, out_channels, drop_prob=0.1): # ---> DROPOUT
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, drop_prob=drop_prob) # ---> DROPOUT

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CenterNetEDAUnet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, base_filters=32, 
                 pdk_dim=9, num_pdks=4, pdk_embed_dim=8, drop_prob=0.15): # ---> DROPOUT (параметр drop_prob)
        super().__init__()
        
        self.drop_prob = drop_prob
        self.in_conv = DoubleConv(in_channels, base_filters, drop_prob=0.0) # Вход лучше не зашумлять
        
        # Encoder
        self.down1 = Down(base_filters, base_filters * 2, drop_prob=drop_prob)      
        self.down2 = Down(base_filters * 2, base_filters * 4, drop_prob=drop_prob)  
        self.down3 = Down(base_filters * 4, base_filters * 8, drop_prob=drop_prob)  
        
        # --- СЛОЙ ЭМБЕДДИНГОВ ДЛЯ PDK_ID ---
        self.pdk_embed = nn.Embedding(num_embeddings=num_pdks, embedding_dim=pdk_embed_dim)
        total_cond_dim = pdk_dim + pdk_embed_dim
        
        # --- FiLM НА ВЫХОДЕ ИЗ БОТТЛНЕКА ---
        self.bottleneck_film = FiLM(condition_dim=total_cond_dim, feature_dim=base_filters * 8)
        
        # ---> DROPOUT: Боттлнек-дропаут (самый важный щит от переобучения!)
        self.bottleneck_drop = nn.Dropout2d(p=0.2) 
        
        # Decoder 
        self.up1 = Up(base_filters * 8 + base_filters * 4, base_filters * 4, drop_prob=drop_prob) 
        self.up2 = Up(base_filters * 4 + base_filters * 2, base_filters * 2, drop_prob=drop_prob) 
        self.up3 = Up(base_filters * 2 + base_filters, base_filters, drop_prob=drop_prob)         
        
        # --- FiLM СЛОИ В ГОЛОВАХ ---
        self.obj_film = FiLM(condition_dim=total_cond_dim, feature_dim=base_filters)
        self.cls_film = FiLM(condition_dim=total_cond_dim, feature_dim=base_filters)
        
        # ---> DROPOUT: Дропаут перед предсказанием голов
        self.head_drop = nn.Dropout2d(p=drop_prob)
        
        # Выходные головы
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

        self.obj_head[-1].bias.data.fill_(-2.19)

    def forward(self, x, pdk_meta=None, pdk_id=None):
        x_in = self.in_conv(x)
        x_d1 = self.down1(x_in)
        x_d2 = self.down2(x_d1)
        x_d3 = self.down3(x_d2)
        
        # Собираем итоговый вектор контекста
        cond = pdk_meta
        if pdk_id is not None and pdk_meta is not None:
            emb = self.pdk_embed(pdk_id)
            cond = torch.cat([pdk_meta, emb], dim=1)
        
        # 1. Модулируем глубокие фичи в боттлнеке и ПРИМЕНЯЕМ ДРОПАУТ
        if cond is not None:
            x_d3 = self.bottleneck_film(x_d3, cond)
        x_d3 = self.bottleneck_drop(x_d3) # ---> DROPOUT
        
        x_u1 = self.up1(x_d3, x_d2)
        x_u2 = self.up2(x_u1, x_d1)
        x_u3 = self.up3(x_u2, x_in)
        
        # 2. Модулируем финальные фичи прямо перед триггером лосса
        x_obj = self.obj_film(x_u3, cond) if cond is not None else x_u3
        x_cls = self.cls_film(x_u3, cond) if cond is not None else x_u3
        
        # ---> DROPOUT: Защищаем головы от переобучения
        x_obj = self.head_drop(x_obj)
        x_cls = self.head_drop(x_cls)
        x_off = self.head_drop(x_u3)
        
        # Генерация предсказаний
        out_obj = self.obj_head(x_obj)
        out_cls = self.class_head(x_cls)
        out_offset = self.offset_head(x_off) 
        
        return {
            "obj": out_obj,
            "cls": out_cls,
            "offset": out_offset
        }

if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 64, 64)
    dummy_pdk_meta = torch.randn(2, 9)
    dummy_pdk_id = torch.tensor([0, 3])  # asap7, sky130

    model = CenterNetEDAUnet(in_channels=3, num_classes=4, pdk_dim=9, num_pdks=4, pdk_embed_dim=8)
    outputs = model(dummy_input, dummy_pdk_meta, dummy_pdk_id)

    print("Проверка размерностей с FiLM + PDK embedding:")
    print("Вход карт:", dummy_input.shape)
    print("Вход PDK continuous:", dummy_pdk_meta.shape)
    print("Вход PDK id:", dummy_pdk_id.shape)
    print("Objectness head:", outputs["obj"].shape)
    print("Class head:", outputs["cls"].shape)
    print("Offset head:", outputs["offset"].shape)