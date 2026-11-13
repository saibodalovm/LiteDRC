#!/usr/bin/env python3
import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import random
from sklearn.metrics import average_precision_score

from litedrc.data import EDAChipDataset
from litedrc.models import (
    EvidentialLightweightRegionNet,
    EvidentialLightweightRegionNetAttn
)
from litedrc.losses import BetaEDLLoss

def load_config(config_path="configs/default.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

def get_pr_auc(model, dataloader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            pdk_meta = batch["pdk_meta"].to(device)
            pdk_id = batch["pdk_id"].to(device)
            
            preds = model(features, pdk_meta, pdk_id)
            probs = preds["probs"].cpu().numpy()
            
            all_preds.append(probs)
            all_targets.append(batch["target_regions"].numpy())
            
    preds_flat = np.concatenate(all_preds).flatten()
    targets_flat = np.concatenate(all_targets).flatten()
    
    if len(np.unique(targets_flat)) > 1:
        return average_precision_score(targets_flat, preds_flat)
    return 0.0

def set_parameter_requires_grad(model, strategy):
    if strategy == "Full":
        for param in model.parameters(): 
            param.requires_grad = True
    elif strategy == "Frozen":
        unfreeze_keys = ['attn', 'attention', 'film', 'head', 'classifier', 'fc']
        for name, param in model.named_parameters():
            param.requires_grad = any(key in name.lower() for key in unfreeze_keys)

def main():
    parser = argparse.ArgumentParser(description="Few-shot transfer learning training for LiteDRC")
    parser.add_argument("--target-pdk", type=str, default="sky130", help="Target PDK for transfer learning")
    parser.add_argument("--data-root", type=str, default="data/processed", help="Path to processed dataset root")
    parser.add_argument("--out-csv", type=str, default="few_shot_results.csv", help="Output CSV path for results")
    parser.add_argument("--max-steps", type=int, default=200, help="Max weight update steps")
    parser.add_argument("--eval-every", type=int, default=50, help="Evaluation frequency in steps")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    train_dir = os.path.join(args.data_root, "train")
    test_dir = os.path.join(args.data_root, "val")
    stats_path = os.path.join(train_dir, "global_zscore_stats.json")
    if not os.path.exists(stats_path):
        stats_path = "global_zscore_stats.json" if os.path.exists("global_zscore_stats.json") else stats_path
        
    train_ds = EDAChipDataset(
        data_dir=train_dir if os.path.exists(train_dir) else args.data_root,
        stats_path=stats_path,
        patch_size=config.get("data", {}).get("patch_size", 64),
        samples_per_file=config.get("data", {}).get("samples_per_file", 5)
    )
    test_ds = EDAChipDataset(
        data_dir=test_dir if os.path.exists(test_dir) else args.data_root,
        stats_path=stats_path,
        patch_size=config.get("data", {}).get("patch_size", 64),
        mode='test'
    )
    
    base_train_indices = [i for i in range(len(train_ds)) if args.target_pdk in str(train_ds.files[i // 5])]
    test_indices = [i for i in range(len(test_ds)) if args.target_pdk in str(test_ds.files[test_ds.pdk_grid[i][0]])]
    val_loader = DataLoader(Subset(test_ds, test_indices), batch_size=512, shuffle=False, num_workers=2)
    
    fractions = [0.01, 0.05, 0.10, 0.25, 1.0]
    seeds = [42, 43, 44]
    batch_size = config.get("training", {}).get("batch_size", 128)
    
    weights_map = {
        "Baseline": "ckpt/zero_shot/gf180_to_sky130_edl_none.pth",
        "FiLM": "ckpt/zero_shot/gf180_to_sky130_edl_film.pth",
        "CrossAttention": "ckpt/zero_shot/gf180_to_sky130_edl_cross_attn.pth"
    }
    
    results_data = []
    
    for seed in seeds:
        print(f"\n🌱 SEED = {seed}")
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        full_indices = base_train_indices.copy()
        random.shuffle(full_indices)
        
        for model_name, weight_path in weights_map.items():
            print(f"  🔥 Architecture: {model_name}")
            for frac in fractions:
                subset_size = max(1, int(len(full_indices) * frac))
                current_indices = full_indices[:subset_size]
                loader = DataLoader(Subset(train_ds, current_indices), batch_size=batch_size, shuffle=True, drop_last=False)
                
                for strategy in ["Frozen", "Full"]:
                    if model_name == "CrossAttention":
                        model = EvidentialLightweightRegionNetAttn(in_channels=3, num_classes=3, use_film=True).to(device)
                    elif model_name == "FiLM":
                        model = EvidentialLightweightRegionNet(in_channels=3, num_classes=3, use_film=True).to(device)
                    else:
                        model = EvidentialLightweightRegionNet(in_channels=3, num_classes=3, use_film=False).to(device)
                        
                    if os.path.exists(weight_path):
                        model.load_state_dict(torch.load(weight_path, map_location=device))
                    else:
                        print(f"    [Warning] Checkpoint {weight_path} not found. Initializing randomly.")
                        
                    set_parameter_requires_grad(model, strategy)
                    lr = 2e-4 if strategy == "Frozen" else 1e-5
                    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-2)
                    criterion = BetaEDLLoss(annealing_steps=100)
                    
                    best_pr_auc = 0.0
                    step = 0
                    train_iter = iter(loader)
                    model.train()
                    
                    while step < args.max_steps:
                        try:
                            batch = next(train_iter)
                        except StopIteration:
                            train_iter = iter(loader)
                            batch = next(train_iter)
                            
                        features = batch["features"].to(device)
                        pdk_meta = batch["pdk_meta"].to(device)
                        pdk_id = batch["pdk_id"].to(device)
                        targets = batch["target_regions"].to(device)
                        
                        optimizer.zero_grad()
                        preds = model(features, pdk_meta, pdk_id)
                        loss = criterion(preds["alpha"], preds["beta"], targets, current_step=step)
                        loss.backward()
                        
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        step += 1
                        
                        if step % args.eval_every == 0 or step == args.max_steps:
                            current_pr = get_pr_auc(model, val_loader, device)
                            if current_pr > best_pr_auc:
                                best_pr_auc = current_pr
                            model.train()
                            
                    print(f"    [{frac*100}% | {strategy}] Best PR-AUC: {best_pr_auc:.4f}")
                    results_data.append({
                        "Seed": seed, "Model": model_name, "Data_Fraction": frac,
                        "Strategy": strategy, "PR_AUC": best_pr_auc
                    })
                    
    df = pd.DataFrame(results_data)
    df.to_csv(args.out_csv, index=False)
    print(f"\n✅ Few-shot training complete. Results saved to {args.out_csv}")

if __name__ == "__main__":
    main()
