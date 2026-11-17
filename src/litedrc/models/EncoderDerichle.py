import torch
import torch.nn as nn

class FiLM(nn.Module):
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
        film_params = self.mlp(condition)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        return x * (1 + gamma) + beta


class ConcatConditioning(nn.Module):
    def __init__(self, condition_dim, feature_dim):
        super().__init__()
        self.proj = nn.Conv2d(feature_dim + condition_dim, feature_dim, kernel_size=1)

    def forward(self, x, condition):
        b, _, h, w = x.shape
        cond_map = condition.unsqueeze(2).unsqueeze(3).expand(b, -1, h, w)
        return self.proj(torch.cat([x, cond_map], dim=1))

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
                 pdk_dim=9, num_pdks=4, pdk_embed_dim=8, drop_prob=0.15, 
                 use_film=True, cond_type=None):
        super().__init__()
        
        self.drop_prob = drop_prob
        self.cond_type = cond_type if cond_type is not None else ("film" if use_film else "none")
        self.use_film = (self.cond_type == "film")
        self.num_classes = num_classes
        
        self.in_conv = DoubleConv(in_channels, base_filters, drop_prob=0.0) 
        self.down1 = Down(base_filters, base_filters * 2, drop_prob=drop_prob)      
        self.down2 = Down(base_filters * 2, base_filters * 4, drop_prob=drop_prob)  
        self.down3 = Down(base_filters * 4, base_filters * 8, drop_prob=drop_prob)  
        
        if self.cond_type in ("film", "concat"):
            self.pdk_embed = nn.Embedding(num_embeddings=num_pdks, embedding_dim=pdk_embed_dim)
            total_cond_dim = pdk_dim + pdk_embed_dim
            if self.cond_type == "film":
                self.bottleneck_film = FiLM(condition_dim=total_cond_dim, feature_dim=base_filters * 8)
            elif self.cond_type == "concat":
                self.bottleneck_concat = ConcatConditioning(condition_dim=total_cond_dim, feature_dim=base_filters * 8)
        
        self.bottleneck_drop = nn.Dropout2d(p=0.2)
        
        self.regional_head = nn.Sequential(
            nn.Conv2d(base_filters * 8, base_filters * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_prob),
            nn.Conv2d(base_filters * 4, num_classes * 2, kernel_size=1),
            nn.Softplus()
        )

    def forward(self, x, pdk_meta=None, pdk_id=None):
        x_in = self.in_conv(x)
        x_d1 = self.down1(x_in)
        x_d2 = self.down2(x_d1)
        x_d3 = self.down3(x_d2) 
        
        if self.cond_type != "none" and pdk_id is not None and pdk_meta is not None:
            emb = self.pdk_embed(pdk_id)
            cond = torch.cat([pdk_meta, emb], dim=1)
            if self.cond_type == "film":
                x_d3 = self.bottleneck_film(x_d3, cond)
            elif self.cond_type == "concat":
                x_d3 = self.bottleneck_concat(x_d3, cond)
            
        x_d3 = self.bottleneck_drop(x_d3)
        
        evidence = self.regional_head(x_d3)
        evidence_pos, evidence_neg = torch.chunk(evidence, 2, dim=1)
        
        alpha = evidence_pos + 1.0
        beta = evidence_neg + 1.0
        S = alpha + beta
        probs = alpha / S
        uncertainty = 2.0 / S
        
        return {
            "evidence_pos": evidence_pos,
            "evidence_neg": evidence_neg,
            "alpha": alpha,
            "beta": beta,
            "probs": probs,             
            "uncertainty": uncertainty  
        }
