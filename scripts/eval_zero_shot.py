#!/usr/bin/env python3
import argparse
import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

from litedrc.data import EDAChipDataset
from litedrc.models import (
    EvidentialLightweightRegionNet,
    EvidentialLightweightRegionNetAttn
)

def load_config(config_path="configs/default.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

def main():
    parser = argparse.ArgumentParser(description="Evaluate zero-shot performance for LiteDRC")
    parser.add_argument("--test-pdk", type=str, default="sky130", help="PDK to evaluate on")
    parser.add_argument("--weights", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--cond", type=str, choices=["none", "film", "cross_attn"], default="cross_attn", help="Conditioning type")
    parser.add_argument("--data-root", type=str, default="data/processed", help="Path to processed dataset root")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    test_dir = os.path.join(args.data_root, "val")
    stats_path = os.path.join(args.data_root, "train", "global_zscore_stats.json")
    if not os.path.exists(stats_path):
        stats_path = "global_zscore_stats.json" if os.path.exists("global_zscore_stats.json") else stats_path
        
    dataset = EDAChipDataset(
        data_dir=test_dir if os.path.exists(test_dir) else args.data_root,
        stats_path=stats_path,
        patch_size=config.get("data", {}).get("patch_size", 64),
        samples_per_file=2,
        mode='test'
    )
    
    test_indices = [i for i in range(len(dataset)) if args.test_pdk in str(dataset.files[dataset.pdk_grid[i][0]])]
    subset = Subset(dataset, test_indices)
    dataloader = DataLoader(subset, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    
    model_cfg = config.get("model", {})
    if args.cond == "none":
        model = EvidentialLightweightRegionNet(in_channels=3, num_classes=3, use_film=False).to(device)
    elif args.cond == "film":
        model = EvidentialLightweightRegionNet(in_channels=3, num_classes=3, use_film=True).to(device)
    else:
        model = EvidentialLightweightRegionNetAttn(in_channels=3, num_classes=3, use_film=True).to(device)
        
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights from {args.weights}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")
        
    model.eval()
    all_preds, all_targets, all_uncertainties = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            pdk_meta = batch["pdk_meta"].to(device)
            pdk_id = batch["pdk_id"].to(device)
            targets = batch["target_regions"].to(device)
            
            preds = model(features, pdk_meta, pdk_id)
            all_preds.append(preds["probs"].cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_uncertainties.append(preds["uncertainty"].cpu().numpy())
            
    preds_arr = np.concatenate(all_preds)
    targets_arr = np.concatenate(all_targets)
    
    preds_flat = preds_arr.flatten()
    targets_flat = targets_arr.flatten()
    
    pr_auc = average_precision_score(targets_flat, preds_flat) if len(np.unique(targets_flat)) > 1 else 0.0
    roc_auc = roc_auc_score(targets_flat, preds_flat) if len(np.unique(targets_flat)) > 1 else 0.0
    
    print(f"\n--- Evaluation Results on {args.test_pdk.upper()} ---")
    print(f"Global PR-AUC: {pr_auc:.4f}")
    print(f"Global ROC-AUC: {roc_auc:.4f}")
    
    for c, c_name in enumerate(["Short", "Spacing", "MinArea"]):
        t_c = targets_arr[:, c, :, :].flatten()
        p_c = preds_arr[:, c, :, :].flatten()
        c_prauc = average_precision_score(t_c, p_c) if len(np.unique(t_c)) > 1 else 0.0
        print(f"  Class {c_name} PR-AUC: {c_prauc:.4f}")

if __name__ == "__main__":
    main()
