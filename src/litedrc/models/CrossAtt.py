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
        
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.mha = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.SiLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)
        cond_emb = self.cond_proj(cond).unsqueeze(1)
        
        norm_x = self.norm1(x_flat)
        attn_out, _ = self.mha(query=norm_x, key=cond_emb, value=cond_emb)
        
        x_flat = x_flat + self.gamma * attn_out
        x_flat = x_flat + self.ffn(self.norm2(x_flat))
        
        out = x_flat.permute(0, 2, 1).reshape(B, C, H, W)
        return out
