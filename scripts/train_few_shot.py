#!/usr/bin/env python3
import argparse
import os
import random
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import average_precision_score, precision_recall_curve

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


def compute_max_f1(targets, preds):
    if len(np.unique(targets)) <= 1:
        return 0.0
    prec, rec, _ = precision_recall_curve(targets, preds)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) != 0)
    return float(np.max(f1))


def evaluate_metrics(model, dataloader, device):
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

    preds_arr = np.concatenate(all_preds)
    targets_arr = np.concatenate(all_targets)

    # Micro-averaged multi-label metrics
    preds_flat = preds_arr.flatten()
    targets_flat = targets_arr.flatten()
    pr_auc = average_precision_score(targets_flat, preds_flat) if len(np.unique(targets_flat)) > 1 else 0.0
    f1_pct = compute_max_f1(targets_flat, preds_flat) * 100.0

    # Binary hotspot metrics (Any DRC)
    bin_preds = np.max(preds_arr, axis=1).flatten()
    bin_targets = np.max(targets_arr, axis=1).flatten()
    bin_pr_auc = average_precision_score(bin_targets, bin_preds) if len(np.unique(bin_targets)) > 1 else 0.0
    bin_f1_pct = compute_max_f1(bin_targets, bin_preds) * 100.0

    return {
        "PR_AUC": float(pr_auc),
        "F1_pct": float(f1_pct),
        "Binary_PR_AUC": float(bin_pr_auc),
        "Binary_F1_pct": float(bin_f1_pct),
    }


def set_parameter_requires_grad(model, strategy):
    if strategy == "Full":
        for param in model.parameters():
            param.requires_grad = True
    elif strategy == "Frozen":
        unfreeze_keys = ["attn", "attention", "film", "concat", "pdk_embed", "head", "classifier", "fc"]
        for name, param in model.named_parameters():
            param.requires_grad = any(key in name.lower() for key in unfreeze_keys)


def resolve_checkpoint(ckpt_dir, source_pdk, target_pdk, cond_key):
    candidates = [
        os.path.join(ckpt_dir, f"best_model_zeroshot_train_{source_pdk}_test_{target_pdk}_{cond_key}.pth"),
        os.path.join(ckpt_dir, f"{source_pdk}_to_{target_pdk}_edl_{cond_key}.pth"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Checkpoint for {source_pdk} -> {target_pdk} [{cond_key}] not found in {ckpt_dir}. Checked: {candidates}"
    )


def build_model(model_name, device):
    if model_name == "CrossAttention":
        return EvidentialLightweightRegionNetAttn(in_channels=3, num_classes=3, use_film=True).to(device)
    elif model_name == "FiLM":
        return EvidentialLightweightRegionNet(in_channels=3, num_classes=3, use_film=True, cond_type="film").to(device)
    elif model_name == "Concat":
        return EvidentialLightweightRegionNet(in_channels=3, num_classes=3, use_film=False, cond_type="concat").to(device)
    elif model_name == "Baseline":
        return EvidentialLightweightRegionNet(in_channels=3, num_classes=3, use_film=False, cond_type="none").to(device)
    raise ValueError(f"Unknown model name: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Few-shot transfer learning training for LiteDRC")
    parser.add_argument("--source-pdk", type=str, default=None, help="Source PDK (inferred automatically if omitted)")
    parser.add_argument("--target-pdk", type=str, default="sky130", help="Target PDK for transfer learning")
    parser.add_argument("--ckpt-dir", type=str, default="ckpt/train_zero_shot", help="Directory with zero-shot checkpoints")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Baseline", "Concat", "FiLM", "CrossAttention"],
        choices=["Baseline", "Concat", "FiLM", "CrossAttention"],
        help="Architectures to evaluate"
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.05],
        help="Target data fractions (e.g. 0.05 for Table 5 or 0.01 0.05 0.10 0.25 1.0 for full curves)"
    )
    parser.add_argument("--data-root", type=str, default="data/processed", help="Path to processed dataset root")
    parser.add_argument("--out-csv", type=str, default="results/few_shot_results.csv", help="Output CSV path for results")
    parser.add_argument("--max-steps", type=int, default=200, help="Max weight update steps")
    parser.add_argument("--eval-every", type=int, default=50, help="Evaluation frequency in steps")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")

    args = parser.parse_args()
    config = load_config(args.config)

    source_pdk = args.source_pdk
    if source_pdk is None:
        source_pdk = "gf180" if args.target_pdk == "sky130" else "sky130"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Transfer: {source_pdk.upper()} -> {args.target_pdk.upper()}")

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
        mode="test"
    )

    base_train_indices = [i for i in range(len(train_ds)) if args.target_pdk in str(train_ds.files[i // 5])]
    test_indices = [i for i in range(len(test_ds)) if args.target_pdk in str(test_ds.files[test_ds.pdk_grid[i][0]])]
    val_loader = DataLoader(Subset(test_ds, test_indices), batch_size=512, shuffle=False, num_workers=2)

    cond_key_map = {
        "Baseline": "none",
        "Concat": "concat",
        "FiLM": "film",
        "CrossAttention": "cross_attn",
    }
    weights_map = {
        m: resolve_checkpoint(args.ckpt_dir, source_pdk, args.target_pdk, cond_key_map[m])
        for m in args.models
    }

    seeds = [42, 43, 44]
    batch_size = config.get("training", {}).get("batch_size", 128)
    results_data = []

    for seed in seeds:
        print(f"\nSEED = {seed}")
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        full_indices = base_train_indices.copy()
        random.shuffle(full_indices)

        for model_name, weight_path in weights_map.items():
            print(f"  Architecture: {model_name} ({weight_path})")
            for frac in args.fractions:
                subset_size = max(1, int(len(full_indices) * frac))
                current_indices = full_indices[:subset_size]
                loader = DataLoader(Subset(train_ds, current_indices), batch_size=batch_size, shuffle=True, drop_last=False)

                for strategy in ["Frozen", "Full"]:
                    model = build_model(model_name, device)
                    model.load_state_dict(torch.load(weight_path, map_location=device))

                    set_parameter_requires_grad(model, strategy)
                    lr = 2e-4 if strategy == "Frozen" else 1e-5
                    optimizer = torch.optim.AdamW(
                        [p for p in model.parameters() if p.requires_grad],
                        lr=lr,
                        weight_decay=1e-2
                    )
                    criterion = BetaEDLLoss(annealing_epochs=100)

                    best_metrics = {
                        "PR_AUC": -1.0,
                        "F1_pct": 0.0,
                        "Binary_PR_AUC": 0.0,
                        "Binary_F1_pct": 0.0,
                    }
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
                        loss = criterion(preds["alpha"], preds["beta"], targets, step)
                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        step += 1

                        if step % args.eval_every == 0 or step == args.max_steps:
                            metrics = evaluate_metrics(model, val_loader, device)
                            if metrics["PR_AUC"] > best_metrics["PR_AUC"]:
                                best_metrics = metrics
                            model.train()

                    print(
                        f"    [{frac*100:.0f}% | {strategy}] "
                        f"PR-AUC: {best_metrics['PR_AUC']:.4f} | F1: {best_metrics['F1_pct']:.2f}% | "
                        f"Bin PR-AUC: {best_metrics['Binary_PR_AUC']:.4f} | Bin F1: {best_metrics['Binary_F1_pct']:.2f}%"
                    )
                    results_data.append({
                        "Seed": seed,
                        "Source_PDK": source_pdk,
                        "Target_PDK": args.target_pdk,
                        "Model": model_name,
                        "Data_Fraction": frac,
                        "Strategy": strategy,
                        "PR_AUC": round(best_metrics["PR_AUC"], 4),
                        "F1_pct": round(best_metrics["F1_pct"], 2),
                        "Binary_PR_AUC": round(best_metrics["Binary_PR_AUC"], 4),
                        "Binary_F1_pct": round(best_metrics["Binary_F1_pct"], 2),
                    })

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(results_data)
    df.to_csv(args.out_csv, index=False)

    summary = (
        df.groupby(["Source_PDK", "Target_PDK", "Model", "Data_Fraction", "Strategy"], as_index=False)
        .agg(
            PR_AUC_mean=("PR_AUC", "mean"),
            PR_AUC_std=("PR_AUC", "std"),
            F1_pct_mean=("F1_pct", "mean"),
            F1_pct_std=("F1_pct", "std"),
            Binary_PR_AUC_mean=("Binary_PR_AUC", "mean"),
            Binary_PR_AUC_std=("Binary_PR_AUC", "std"),
            Binary_F1_pct_mean=("Binary_F1_pct", "mean"),
            Binary_F1_pct_std=("Binary_F1_pct", "std"),
        )
        .round(4)
    )
    summary_path = args.out_csv.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\nFew-shot training complete.\n  Raw per-seed CSV: {args.out_csv}\n  Summary CSV:      {summary_path}")


if __name__ == "__main__":
    main()