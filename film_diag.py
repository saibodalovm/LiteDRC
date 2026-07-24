import torch
from models.Encoder import LightweightRegionNet
 
def film_diagnostics(model):
    if not model.use_film:
        print("FiLM выключен в этой модели")
        return
    last_w = model.bottleneck_film.mlp[-1].weight
    last_b = model.bottleneck_film.mlp[-1].bias
    emb = model.pdk_embed.weight
 
    print(f"FiLM last layer |weight|_2 = {last_w.norm().item():.6f}  |bias|_2 = {last_b.norm().item():.6f}")
    print(f"PDK embedding per-PDK norms: {emb.norm(dim=1).detach().cpu().numpy()}")
    # Если PDK-эмбеддинги почти идентичны друг другу (низкая межклассовая
    # дисперсия) - модель не научилась их различать, даже если норма ненулевая
    emb_np = emb.detach().cpu().numpy()
    pairwise_dist = ((emb_np[:, None, :] - emb_np[None, :, :]) ** 2).sum(-1) ** 0.5
    print("Попарные расстояния между PDK-эмбеддингами:\n", pairwise_dist)

device = 'cuda'
model = LightweightRegionNet(in_channels=3, num_classes=3, use_film=True).to(device)
model.load_state_dict(torch.load('best_lightweight_region_net.pth', weights_only=True))
model.eval()
film_diagnostics(model=model)