import torch
import torch.nn as nn

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
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, condition):
        film_params = self.mlp(condition) # (B, 2*C)
        gamma, beta = torch.chunk(film_params, 2, dim=1) # Каждая по (B, C)
        
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        
        return x * (1 + gamma) + beta


class DoubleConv(nn.Module):
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
    def __init__(self, in_channels, out_channels, drop_prob=0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, drop_prob=drop_prob)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class EvidentialLightweightRegionNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, base_filters=32, 
                 pdk_dim=9, num_pdks=4, pdk_embed_dim=8, drop_prob=0.15, use_film=True):
        super().__init__()
        
        self.drop_prob = drop_prob
        self.use_film = use_film
        self.num_classes = num_classes
        
        self.in_conv = DoubleConv(in_channels, base_filters, drop_prob=0.0) 
        self.down1 = Down(base_filters, base_filters * 2, drop_prob=drop_prob)      
        self.down2 = Down(base_filters * 2, base_filters * 4, drop_prob=drop_prob)  
        self.down3 = Down(base_filters * 4, base_filters * 8, drop_prob=drop_prob)  
        
        if self.use_film:
            self.pdk_embed = nn.Embedding(num_embeddings=num_pdks, embedding_dim=pdk_embed_dim)
            total_cond_dim = pdk_dim + pdk_embed_dim
            self.bottleneck_film = FiLM(condition_dim=total_cond_dim, feature_dim=base_filters * 8)
        
        self.bottleneck_drop = nn.Dropout2d(p=0.2)
        
        # --- Evidential Regional Head ---
        # Выдаем свидетельства (evidence). Softplus гарантирует, что e >= 0.
        self.regional_head = nn.Sequential(
            nn.Conv2d(base_filters * 8, base_filters * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_prob),
            # Выдаем 6 каналов (по 2 на каждый класс: evidence_pos и evidence_neg)
            nn.Conv2d(base_filters * 4, num_classes * 2, kernel_size=1),
            nn.Softplus() # Гарантируем e >= 0
        )

    def forward(self, x, pdk_meta=None, pdk_id=None):
        x_in = self.in_conv(x)
        x_d1 = self.down1(x_in)
        x_d2 = self.down2(x_d1)
        x_d3 = self.down3(x_d2) 
        
        if self.use_film and pdk_id is not None and pdk_meta is not None:
            emb = self.pdk_embed(pdk_id)
            cond = torch.cat([pdk_meta, emb], dim=1)
            x_d3 = self.bottleneck_film(x_d3, cond)
            
        x_d3 = self.bottleneck_drop(x_d3)
        
        # Свидетельства (Evidence): shape (B, num_classes, H, W)
        evidence = self.regional_head(x_d3)
        
        # Разбиваем 6 каналов на 3 позитивных и 3 негативных
        evidence_pos, evidence_neg = torch.chunk(evidence, 2, dim=1) # оба (B, 3, H, W)
        
        # 2. Параметры Бета-распределения (alpha и beta)
        alpha = evidence_pos + 1.0
        beta = evidence_neg + 1.0
        
        # 3. Общая масса свидетельств S (для каждого класса независимо)
        S = alpha + beta # (B, 3, H, W)
        
        # 4. Ожидаемая вероятность наличия дефекта
        probs = alpha / S # (B, 3, H, W)
        
        # 5. Неопределенность (K=2, так как бинарный выбор: есть дефект / нет дефекта)
        uncertainty = 2.0 / S # (B, 3, H, W)
        
        return {
            "evidence_pos": evidence_pos,
            "evidence_neg": evidence_neg,
            "alpha": alpha,
            "beta": beta,
            "probs": probs,             
            "uncertainty": uncertainty  
        }

if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 64, 64)
    dummy_pdk_meta = torch.randn(2, 9)
    dummy_pdk_id = torch.tensor([0, 3])  # asap7, sky130

    model = EvidentialLightweightRegionNet(in_channels=3, num_classes=3)
    outputs = model(dummy_input, dummy_pdk_meta, dummy_pdk_id)

    print("Проверка размерностей (Evidential Deep Learning):")
    print("Вероятности классов (probs):", outputs["probs"].shape)           # Ожидаем (2, 3, 8, 8)
    print("Карта неопределенности (uncertainty):", outputs["uncertainty"].shape) # Ожидаем (2, 1, 8, 8)