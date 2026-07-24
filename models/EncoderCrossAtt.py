import torch
import torch.nn as nn
from models.CrossAtt import PDKCrossAttention


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU => [Dropout2d]) => (convolution => [BN] => ReLU)"""
    def __init__(self, in_channels, out_channels, drop_prob=0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, drop_prob=0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, drop_prob=drop_prob)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class LightweightRegionNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, base_filters=32, 
                 pdk_dim=9, num_pdks=4, pdk_embed_dim=8, drop_prob=0.15, use_film=True): # <-- ДОБАВИЛИ ФЛАГ
        super().__init__()
        
        self.drop_prob = drop_prob
        self.use_film = use_film # Сохраняем флаг
        
        self.in_conv = DoubleConv(in_channels, base_filters, drop_prob=0.0) 
        
        self.down1 = Down(base_filters, base_filters * 2, drop_prob=drop_prob)      
        self.down2 = Down(base_filters * 2, base_filters * 4, drop_prob=drop_prob)  
        self.down3 = Down(base_filters * 4, base_filters * 8, drop_prob=drop_prob)  
        
        # --- ИНИЦИАЛИЗИРУЕМ FiLM ТОЛЬКО ЕСЛИ ФЛАГ TRUE ---
        if self.use_film:
            self.pdk_embed = nn.Embedding(num_embeddings=num_pdks, embedding_dim=pdk_embed_dim)
            total_cond_dim = pdk_dim + pdk_embed_dim
            self.attn = PDKCrossAttention(feature_dim=base_filters * 8, cond_dim=total_cond_dim, num_heads=4)
        
        self.bottleneck_drop = nn.Dropout2d(p=0.2)
        
        self.regional_head = nn.Sequential(
            nn.Conv2d(base_filters * 8, base_filters * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_prob),
            nn.Conv2d(base_filters * 4, num_classes, kernel_size=1) 
        )

    def forward(self, x, pdk_meta=None, pdk_id=None):
        x_in = self.in_conv(x)
        x_d1 = self.down1(x_in)
        x_d2 = self.down2(x_d1)
        x_d3 = self.down3(x_d2) 
        
        # --- ПРИМЕНЯЕМ FiLM ТОЛЬКО ЕСЛИ ОН ВКЛЮЧЕН ---
        if self.use_film and pdk_id is not None and pdk_meta is not None:
            emb = self.pdk_embed(pdk_id)
            cond = torch.cat([pdk_meta, emb], dim=1)
            x_d3 = self.attn(x_d3, cond)
            
        x_d3 = self.bottleneck_drop(x_d3)
        
        out_regions = self.regional_head(x_d3)
        
        return {
            "pred_regions": out_regions 
        }

if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 64, 64)
    dummy_pdk_meta = torch.randn(2, 9)
    dummy_pdk_id = torch.tensor([0, 3])  # asap7, sky130

    # Обрати внимание: num_classes=3, так как мы выкинули фон в датасете
    model = LightweightRegionNet(in_channels=3, num_classes=3, pdk_dim=9, num_pdks=4, pdk_embed_dim=8)
    outputs = model(dummy_input, dummy_pdk_meta, dummy_pdk_id)

    print("Проверка размерностей (Encoder-Only + FiLM):")
    print("Вход признаков:", dummy_input.shape)
    print("Вход PDK continuous:", dummy_pdk_meta.shape)
    print("Вход PDK id:", dummy_pdk_id.shape)
    print("Regional Head Output:", outputs["pred_regions"].shape) # Ожидаем (2, 3, 8, 8)