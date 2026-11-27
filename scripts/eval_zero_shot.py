import argparse
import csv
import os
import random
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

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def load_config(config_path="configs/default.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def compute_max_f1(targets, preds):
    if len(np.unique(targets)) <= 1:
        return 0.0
    prec, rec, _ = precision_recall_curve(targets, preds)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) != 0)
    return float(np.max(f1))


def infer_train_pdk(weights_path):
    name = os.path.basename(weights_path).lower()
    if "train_sky130" in name:
        return "sky130"
    if "train_gf180" in name:
        return "gf180"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Evaluate zero-shot / in-domain performance for LiteDRC")
    parser.add_argument("--train-pdk", type=str, default=None, help="Source PDK used for training (auto-detected from weights if omitted)")
    parser.add_argument("--test-pdk", type=str, default="sky130", help="PDK to evaluate on")
    parser.add_argument("--weights", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--cond", type=str, choices=["none", "concat", "film", "cross_attn"], default="cross_attn", help="Conditioning type")
    parser.add_argument("--data-root", type=str, default="data/processed", help="Path to processed dataset root")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--out-csv", type=str, default="results/zero_shot_metrics.csv", help="Path to output CSV file")

    args = parser.parse_args()
    config = load_config(args.config)
    train_pdk = args.train_pdk if args.train_pdk else infer_train_pdk(args.weights)

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
        mode="test"
    )

    test_indices = [i for i in range(len(dataset)) if args.test_pdk in str(dataset.files[dataset.pdk_grid[i][0]])]
    subset = Subset(dataset, test_indices)
    dataloader = DataLoader(subset, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    if args.cond in ("none", "concat", "film"):
        model = EvidentialLightweightRegionNet(
            in_channels=3,
            num_classes=3,
            use_film=(args.cond == "film"),
            cond_type=args.cond
        ).to(device)
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

    # Global (micro-averaged across all classes & spatial windows)
    preds_flat = preds_arr.flatten()
    targets_flat = targets_arr.flatten()
    pr_auc = average_precision_score(targets_flat, preds_flat) if len(np.unique(targets_flat)) > 1 else 0.0
    roc_auc = roc_auc_score(targets_flat, preds_flat) if len(np.unique(targets_flat)) > 1 else 0.0
    global_f1 = compute_max_f1(targets_flat, preds_flat)

    # Binary hotspot detection (Any DRC violation vs. clean)
    bin_preds = np.max(preds_arr, axis=1).flatten()
    bin_targets = np.max(targets_arr, axis=1).flatten()
    bin_prauc = average_precision_score(bin_targets, bin_preds) if len(np.unique(bin_targets)) > 1 else 0.0
    bin_f1 = compute_max_f1(bin_targets, bin_preds)

    print(f"\n--- Evaluation Results: {train_pdk.upper()} -> {args.test_pdk.upper()} [{args.cond}] ---")
    print(f"Binary (Any DRC) PR-AUC: {bin_prauc:.4f} | F1: {bin_f1 * 100:.2f}%")
    print(f"Global PR-AUC: {pr_auc:.4f} | Global F1: {global_f1 * 100:.2f}%")
    print(f"Global ROC-AUC: {roc_auc:.4f}")

    class_metrics = {}
    for c, c_name in enumerate(["Short", "Spacing", "Others"]):
        t_c = targets_arr[:, c, :, :].flatten()
        p_c = preds_arr[:, c, :, :].flatten()
        c_prauc = average_precision_score(t_c, p_c) if len(np.unique(t_c)) > 1 else 0.0
        c_f1 = compute_max_f1(t_c, p_c)
        class_metrics[c_name] = (c_prauc, c_f1)
        print(f"  Class {c_name} PR-AUC: {c_prauc:.4f} | F1: {c_f1 * 100:.2f}%")

    # Save to CSV
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    regime = "in_domain" if train_pdk == args.test_pdk else "zero_shot"
    header = [
        "train_pdk", "test_pdk", "regime", "cond",
        "global_pr_auc", "global_f1_pct",
        "bin_pr_auc", "bin_f1_pct", "global_roc_auc",
        "c0_short_pr_auc", "c0_short_f1_pct",
        "c1_spacing_pr_auc", "c1_spacing_f1_pct",
        "c2_others_pr_auc", "c2_others_f1_pct",
        "weights"
    ]
    row = [
        train_pdk, args.test_pdk, regime, args.cond,
        round(pr_auc, 4), round(global_f1 * 100, 2),
        round(bin_prauc, 4), round(bin_f1 * 100, 2), round(roc_auc, 4),
        round(class_metrics["Short"][0], 4), round(class_metrics["Short"][1] * 100, 2),
        round(class_metrics["Spacing"][0], 4), round(class_metrics["Spacing"][1] * 100, 2),
        round(class_metrics["Others"][0], 4), round(class_metrics["Others"][1] * 100, 2),
        args.weights
    ]

    write_header = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)
    print(f"Saved metrics to {args.out_csv}")


if __name__ == "__main__":
    main()