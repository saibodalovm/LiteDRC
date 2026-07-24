import torch
import torch.nn as nn
import torch.nn.functional as F

class PDKCrossAttention(nn.Module):
    """
    Модуль Cross-Attention: пространственные фичи чипа (Query)
    делают запрос к вектору параметров PDK (Key / Value).
    """
    def __init__(self, feature_dim: int, cond_dim: int, num_heads: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Проекция вектора условий PDK в пространство каналов признаков
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Стандартный Multi-Head Attention
        self.mha = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        
        # Нормализация и LayerScale для стабильности градиентов
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.SiLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        self.gamma = nn.Parameter(torch.zeros(1)) # Инициализация нулем для остаточного подключения

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) — карту признаков из боттлнека (например, 128, 8, 8)
        cond: (B, cond_dim) — вектор PDK (pdk_meta + pdk_embed)
        """
        B, C, H, W = x.shape
        
        # 1. Готовим Query: превращаем карту (B, C, H, W) в последовательность токенов (B, H*W, C)
        x_flat = x.flatten(2).permute(0, 2, 1) # (B, 64, C)
        
        # 2. Готовим Key и Value из вектора PDK: (B, 1, C)
        cond_emb = self.cond_proj(cond).unsqueeze(1) # (B, 1, C)
        
        # 3. Cross-Attention: Query = пространственные ячейки, Key/Value = вектор PDK
        norm_x = self.norm1(x_flat)
        attn_out, _ = self.mha(query=norm_x, key=cond_emb, value=cond_emb)
        
        # Residual connection #1
        x_flat = x_flat + self.gamma * attn_out
        
        # Residual connection #2 (FFN)
        x_flat = x_flat + self.ffn(self.norm2(x_flat))
        
        # 4. Собираем обратно в spatial tensor (B, C, H, W)
        out = x_flat.permute(0, 2, 1).reshape(B, C, H, W)
        return out