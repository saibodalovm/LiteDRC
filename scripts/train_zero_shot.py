#!/usr/bin/env python3
import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.metrics import average_precision_score

from litedrc.data import EDAChipDataset
from litedrc.models import (
    EvidentialLightweightRegionNet,
    EvidentialLightweightRegionNetAttn
)
from litedrc.losses import BetaEDLLoss
from litedrc.metrics import compute_region_metrics

def load_config(config_path="configs/default.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

def train_epoch(model, dataloader, optimizer, criterion, device, current_epoch):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        features = batch["features"].to(device)
        pdk_meta = batch["pdk_meta"].to(device)
        pdk_id = batch["pdk_id"].to(device)
        targets = batch["target_regions"].to(device)
        
        optimizer.zero_grad()
        preds = model(features, pdk_meta, pdk_id)
        alpha = preds["alpha"]
        beta = preds["beta"]
        
        loss = criterion(alpha, beta, targets, current_epoch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device, max_epoch=200):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            pdk_meta = batch["pdk_meta"].to(device)
            pdk_id = batch["pdk_id"].to(device)
            targets = batch["target_regions"].to(device)
            
            preds = model(features, pdk_meta, pdk_id)
            alpha = preds["alpha"]
            beta = preds["beta"]
            probs = preds["probs"]
            
            loss = criterion(alpha, beta, targets, max_epoch)
            running_loss += loss.item()
            
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    
    pr_auc = average_precision_score(all_targets, all_preds) if len(np.unique(all_targets)) > 1 else 0.0
    return running_loss / len(dataloader), pr_auc

def main():
    parser = argparse.ArgumentParser(description="Zero-shot training for LiteDRC")
    parser.add_argument("--train-pdk", type=str, default="sky130", help="Training PDK")
    parser.add_argument("--test-pdk", type=str, default="gf180", help="Testing PDK")
    parser.add_argument("--cond", type=str, choices=["none", "film", "cross_attn"], default="cross_attn", help="Conditioning module")
    parser.add_argument("--loss", type=str, choices=["edl", "focal"], default="edl", help="Loss function")
    parser.add_argument("--use-distances", action="store_true", help="Use power grid distance channels")
    parser.add_argument("--data-root", type=str, default="data/processed", help="Path to processed dataset root")
    parser.add_argument("--out-dir", type=str, default="ckpt/train_zero_shot", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    epochs = args.epochs if args.epochs is not None else config.get("training", {}).get("epochs", 200)
    batch_size = args.batch_size if args.batch_size is not None else config.get("training", {}).get("batch_size", 512)
    lr = args.lr if args.lr is not None else config.get("training", {}).get("lr", 0.0002)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")
    stats_path = os.path.join(train_dir, "global_zscore_stats.json")
    
    if not os.path.exists(stats_path):
        # Fallback or check if data_root is provided differently
        stats_path = "global_zscore_stats.json" if os.path.exists("global_zscore_stats.json") else stats_path
    
    train_dataset = EDAChipDataset(
        data_dir=train_dir if os.path.exists(train_dir) else args.data_root,
        stats_path=stats_path,
        patch_size=config.get("data", {}).get("patch_size", 64),
        samples_per_file=config.get("data", {}).get("samples_per_file", 5),
        use_distances=args.use_distances
    )
    test_dataset = EDAChipDataset(
        data_dir=val_dir if os.path.exists(val_dir) else args.data_root,
        stats_path=stats_path,
        patch_size=config.get("data", {}).get("patch_size", 64),
        samples_per_file=2,
        mode='test',
        use_distances=args.use_distances
    )
    
    train_indices = [i for i in range(len(train_dataset)) if args.train_pdk in str(train_dataset.files[i // 5])]
    val_source_indices = [i for i in range(len(test_dataset)) if args.train_pdk in str(test_dataset.files[test_dataset.pdk_grid[i][0]])]
    val_target_indices = [i for i in range(len(test_dataset)) if args.test_pdk in str(test_dataset.files[test_dataset.pdk_grid[i][0]])]
    
    print(f"Train PDK: {args.train_pdk} ({len(train_indices)} patches)")
    print(f"Test PDK (Zero-shot): {args.test_pdk} ({len(val_target_indices)} patches)")
    
    train_subset = Subset(train_dataset, train_indices)
    val_source_subset = Subset(test_dataset, val_source_indices)
    val_target_subset = Subset(test_dataset, val_target_indices)
    
    all_weights = train_dataset.sample_weights(power=0.2, floor=1.0)
    filtered_weights = [all_weights[i] for i in train_indices] if all_weights is not None else None
    
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=filtered_weights, num_samples=len(filtered_weights), replacement=True
    ) if filtered_weights is not None else None
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    val_source_loader = DataLoader(val_source_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    val_target_loader = DataLoader(val_target_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    model_cfg = config.get("model", {})
    in_channels = model_cfg.get("in_channels", 3) + (2 if args.use_distances else 0)
    num_classes = model_cfg.get("num_classes", 3)
    
    if args.cond == "none":
        model = EvidentialLightweightRegionNet(in_channels=in_channels, num_classes=num_classes, use_film=False).to(device)
    elif args.cond == "film":
        model = EvidentialLightweightRegionNet(in_channels=in_channels, num_classes=num_classes, use_film=True).to(device)
    else:
        model = EvidentialLightweightRegionNetAttn(in_channels=in_channels, num_classes=num_classes, use_film=True).to(device)
        
    criterion = BetaEDLLoss(annealing_epochs=config.get("training", {}).get("annealing_epochs", 50))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.get("training", {}).get("weight_decay", 1e-2))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, f"best_model_zeroshot_train_{args.train_pdk}_test_{args.test_pdk}_{args.cond}.pth")
    
    best_pr_auc = 0.0
    for epoch in range(epochs):
        t_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        _, v_pr_source = validate(model, val_source_loader, criterion, device, max_epoch=epochs)
        _, v_pr_target = validate(model, val_target_loader, criterion, device, max_epoch=epochs)
        scheduler.step()
        
        print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {t_loss:.4f} | Source PR-AUC: {v_pr_source:.4f} | Target PR-AUC: {v_pr_target:.4f}")
        
        if v_pr_source > best_pr_auc:
            best_pr_auc = v_pr_source
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best model to {save_path}")

if __name__ == "__main__":
    main()
