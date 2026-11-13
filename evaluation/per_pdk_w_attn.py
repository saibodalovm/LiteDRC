import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc
from tqdm import tqdm

# --- 1. ИМПОРТЫ МОДЕЛЕЙ ---
# Базовая модель (No_FiLM / With_FiLM)
from models.Encoder import LightweightRegionNet

# МОДЕЛЬ С CROSS-ATTENTION: укажи здесь твой файл и имя класса, например:
# from models.Encoder_Attn import LightweightRegionNet as LightweightRegionNetAttn
from models.EncoderCrossAtt import LightweightRegionNet as LightweightRegionNetAttn 

from data_processing.patch_dataset import EDAChipDataset

IDX_TO_PDK = {0: "asap7", 1: "freepdk45", 2: "gf180", 3: "sky130"}

# Маппинг каналов (после отрезания фона)
CHANNEL_TO_CLASS = {
    0: "Metal Spacing",
    1: "Short",
    2: "Others"
}

def compute_metrics(y_true, y_scores):
    """Вычисляет ROC-AUC, PR-AUC и лучший F1 для плоских массивов."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.0, 0.0, 0.0

    # PR Curve & Best F1
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    
    f1_scores = np.divide(2 * precisions * recalls, precisions + recalls, 
                          out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx] * 100 # В проценты
    
    # ROC AUC
    fprs, tprs, _ = roc_curve(y_true, y_scores)
    roc_auc_val = auc(fprs, tprs)
    
    return roc_auc_val, pr_auc, best_f1

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_pdk_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference", leave=False):
            features = batch["features"].to(device)
            pdk_meta = batch["pdk_meta"].to(device)
            pdk_id = batch["pdk_id"].to(device)
            targets = batch["target_regions"].cpu().numpy() # (B, 3, 8, 8)
            
            preds_logits = model(features, pdk_meta, pdk_id)["pred_regions"]
            preds_probs = torch.sigmoid(preds_logits).cpu().numpy() # (B, 3, 8, 8)
            
            all_preds.append(preds_probs)
            all_targets.append(targets)
            all_pdk_ids.append(batch["pdk_id"].cpu().numpy()) # (B,)

    all_preds = np.concatenate(all_preds, axis=0)     # (N, 3, 8, 8)
    all_targets = np.concatenate(all_targets, axis=0) # (N, 3, 8, 8)
    all_pdk_ids = np.concatenate(all_pdk_ids, axis=0) # (N,)
    
    results = {}
    
    # --- 1. ГЛОБАЛЬНЫЕ МЕТРИКИ ---
    roc_g, pr_g, f1_g = compute_metrics(all_targets.flatten(), all_preds.flatten())
    
    class_metrics_g = {}
    for ch, cls_name in CHANNEL_TO_CLASS.items():
        _, pr_cls, _ = compute_metrics(all_targets[:, ch, :, :].flatten(), all_preds[:, ch, :, :].flatten())
        class_metrics_g[f"{cls_name} PR-AUC"] = pr_cls
        
    results['GLOBAL'] = {
        'ROC-AUC': roc_g, 'Global PR-AUC': pr_g, 'Global F1': f1_g,
        **class_metrics_g
    }
    
    # --- 2. МЕТРИКИ С РАЗБИВКОЙ ПО PDK ---
    for idx, pdk_name in IDX_TO_PDK.items():
        mask = (all_pdk_ids == idx)
        
        pdk_targets = all_targets[mask]
        pdk_preds = all_preds[mask]
        
        if len(pdk_targets) == 0:
            continue
            
        roc, pr, f1 = compute_metrics(pdk_targets.flatten(), pdk_preds.flatten())
        
        class_metrics_pdk = {}
        for ch, cls_name in CHANNEL_TO_CLASS.items():
            _, pr_cls, _ = compute_metrics(pdk_targets[:, ch, :, :].flatten(), pdk_preds[:, ch, :, :].flatten())
            class_metrics_pdk[f"{cls_name} PR-AUC"] = pr_cls
            
        results[pdk_name] = {
            'ROC-AUC': roc, 'Global PR-AUC': pr, 'Global F1': f1,
            **class_metrics_pdk
        }
        
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}\n")
    
    TEST_DIR = r"J:\processed_dataset\val"
    STATS_PATH = "J:/processed_dataset/train/global_zscore_stats.json"
    PATCH_SIZE = 64
    BATCH_SIZE = 128
    
    # ======= КОНФИГУРАЦИЯ ВСЕХ ТРЕХ МОДЕЛЕЙ =======
    MODELS_CONFIG = {
        "No_FiLM": {
            "class": LightweightRegionNet,
            "kwargs": {"in_channels": 3, "num_classes": 3, "use_film": False},
            "path": "ckpt/FILM_sky_gf/best_lightweight_region_net_NOFILM_only_sky_and_gf.pth"
        },
        "With_FiLM": {
            "class": LightweightRegionNet,
            "kwargs": {"in_channels": 3, "num_classes": 3, "use_film": True},
            "path": "ckpt/FILM_sky_gf/best_lightweight_region_net_FILM_only_sky_and_gf.pth"
        },
        "CrossAttention": {
            "class": LightweightRegionNetAttn, # Класс модели с Cross-Attention
            "kwargs": {"in_channels": 3, "num_classes": 3, "use_film": True},
            "path": "best_lightweight_region_net_crossattn_sky_and_gf.pth" # Укажи путь к весам с Attention
        }
    }
    # ===============================================
    
    test_dataset = EDAChipDataset(
        data_dir=TEST_DIR, stats_path=STATS_PATH, patch_size=PATCH_SIZE, mode='test'
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    all_results = []
    
    for model_name, config in MODELS_CONFIG.items():
        print(f"--- Тестирование модели: {model_name} ---")
        
        # Динамически создаем модель из указанного класса и параметров
        model_cls = config["class"]
        model = model_cls(**config["kwargs"])
        
        model.load_state_dict(torch.load(config["path"], map_location=device))
        model.to(device)
        
        res = evaluate_model(model, test_loader, device)
        
        for pdk_type, metrics in res.items():
            row = {'Model': model_name, 'PDK / Scope': pdk_type}
            row.update(metrics)
            all_results.append(row)
            
    df = pd.DataFrame(all_results)
    df = df.sort_values(by=['PDK / Scope', 'Model']).reset_index(drop=True)
    
    # Сохраняем статистику в CSV
    #df.to_csv('per_pdk_stat.csv', index=False)
    
    print("\n======================================= СВОДНАЯ ТАБЛИЦА МЕТРИК =======================================")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1500)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if x <= 1.0 else f"{x:.2f}%"))
    print("======================================================================================================")

if __name__ == "__main__":
    main()