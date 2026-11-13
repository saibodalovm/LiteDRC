#!/usr/bin/env python3
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1 import make_axes_locatable

from litedrc.models.EncoderCrossAttDerichle import EvidentialLightweightRegionNet as EvidentialLightweightRegionNetAttn 
from litedrc.data.patch_dataset import EDAChipDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize full chip predictions and uncertainty.")
    parser.add_argument("--pdk", type=str, default="sky130", help="Target PDK name.")
    parser.add_argument("--file-idx", type=int, help="Index of the file to visualize. If not provided, lists all files.")
    parser.add_argument("--test-dir", type=str, required=True, help="Directory containing test .npz files.")
    parser.add_argument("--stats-path", type=str, required=True, help="Path to global_zscore_stats.json.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--mode", type=str, choices=["agg", "per_class"], default="agg", help="Visualization mode: 'agg' (aggregated) or 'per_class'.")
    parser.add_argument("--save-dir", type=str, default="visualizations", help="Directory to save plots.")
    parser.add_argument("--crop-x", type=int, default=2, help="Crop pixels from the left side (default 2 patches = 16 pixels).")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading dataset...")
    test_ds = EDAChipDataset(data_dir=args.test_dir, stats_path=args.stats_path, patch_size=64, mode='test')
    
    if args.file_idx is None:
        print("\n📋 LIST OF AVAILABLE RUNS:")
        for idx, filepath in enumerate(test_ds.files):
            if args.pdk in str(filepath):
                rel_path = os.path.join(os.path.basename(os.path.dirname(filepath)), os.path.basename(filepath))
                print(f"  [{idx}] {rel_path}")
        print("\n👉 Choose an index and run again with --file-idx IDX")
        return

    if args.file_idx < 0 or args.file_idx >= len(test_ds.files):
        print(f"Error: Invalid index {args.file_idx}!")
        return
        
    target_filepath = test_ds.files[args.file_idx]
    design_name = f"{os.path.basename(os.path.dirname(target_filepath))}_{os.path.basename(target_filepath).split('.')[0]}"
    
    file_indices = [i for i in range(len(test_ds)) if test_ds.pdk_grid[i][0] == args.file_idx]
    
    if not file_indices:
        print("No patches found for the selected file!")
        return

    max_row = max(test_ds.pdk_grid[i][1] for i in file_indices)
    max_col = max(test_ds.pdk_grid[i][2] for i in file_indices)
    
    full_h = (max_row + 1) * 8
    full_w = (max_col + 1) * 8
    
    print(f"\n🎯 Selected run: {design_name}")
    print(f"Macro grid: {max_row+1} x {max_col+1} patches")
    print(f"Full map resolution: {full_h} x {full_w} pixels")

    if args.mode == "agg":
        full_gt = np.zeros((full_h, full_w))
        full_pred = np.zeros((full_h, full_w))
        full_unc = np.zeros((full_h, full_w))
    else:
        full_gt = np.zeros((3, full_h, full_w))
        full_pred = np.zeros((3, full_h, full_w))
        full_unc = np.zeros((3, full_h, full_w))

    print("Initializing model...")
    model = EvidentialLightweightRegionNetAttn(in_channels=3, num_classes=3, use_film=True).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    print("Inference and map assembly...")
    with torch.no_grad():
        for idx in file_indices:
            _, row, col = test_ds.pdk_grid[idx]
            
            sample = test_ds[idx]
            features = sample["features"].unsqueeze(0).to(device)
            pdk_meta = sample["pdk_meta"].unsqueeze(0).to(device)
            pdk_id = sample["pdk_id"].unsqueeze(0).to(device)
            targets = sample["target_regions"].numpy() 
            
            preds = model(features, pdk_meta, pdk_id)
            probs = preds["probs"].cpu().numpy()[0]          
            uncertainty = preds["uncertainty"].cpu().numpy()[0] 
            
            if args.mode == "agg":
                # Aggregate across classes (max)
                full_gt[row*8 : (row+1)*8, col*8 : (col+1)*8] = np.max(targets, axis=0)
                full_pred[row*8 : (row+1)*8, col*8 : (col+1)*8] = np.max(probs, axis=0)
                full_unc[row*8 : (row+1)*8, col*8 : (col+1)*8] = np.max(uncertainty, axis=0)
            else:
                full_gt[:, row*8 : (row+1)*8, col*8 : (col+1)*8] = targets
                full_pred[:, row*8 : (row+1)*8, col*8 : (col+1)*8] = probs
                full_unc[:, row*8 : (row+1)*8, col*8 : (col+1)*8] = uncertainty

    # Cropping
    crop_pixels = args.crop_x * 8
    if args.mode == "agg":
        full_gt = full_gt[:, crop_pixels:]
        full_pred = full_pred[:, crop_pixels:]
        full_unc = full_unc[:, crop_pixels:]
    else:
        full_gt = full_gt[:, :, crop_pixels:]
        full_pred = full_pred[:, :, crop_pixels:]
        full_unc = full_unc[:, :, crop_pixels:]

    if args.mode == "agg":
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), dpi=300)
        title_fontsize = 9.5
        cbar_tick_fontsize = 8

        # Ground Truth
        axes[0].imshow(full_gt, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Ground Truth (All DRCs)", fontsize=title_fontsize, pad=6)
        axes[0].axis("off")
        div0 = make_axes_locatable(axes[0])
        div0.append_axes("right", size="5%", pad=0.06).axis("off")

        # Probability
        im1 = axes[1].imshow(full_pred, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Predicted Probability Map", fontsize=title_fontsize, pad=6)
        axes[1].axis("off")
        div1 = make_axes_locatable(axes[1])
        cax1 = div1.append_axes("right", size="5%", pad=0.06)
        fig.colorbar(im1, cax=cax1).ax.tick_params(labelsize=cbar_tick_fontsize, length=2, pad=2)

        # Uncertainty
        im2 = axes[2].imshow(full_unc, cmap="magma", vmin=0, vmax=1)
        axes[2].set_title("Evidential Uncertainty ($u$)", fontsize=title_fontsize, pad=6)
        axes[2].axis("off")
        div2 = make_axes_locatable(axes[2])
        cax2 = div2.append_axes("right", size="5%", pad=0.06)
        fig.colorbar(im2, cax=cax2).ax.tick_params(labelsize=cbar_tick_fontsize, length=2, pad=2)

        plt.subplots_adjust(left=0.01, right=0.96, top=0.88, bottom=0.05, wspace=0.25)
        save_path = os.path.join(args.save_dir, f"full_chip_{design_name}.pdf")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
        print(f"✅ Aggregated map saved to: {save_path}")

    else:
        class_names = ['Short', 'Metal Spacing', 'Others']
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))
        fig.suptitle(f'Per-Class DRC Hotspots: {design_name}', fontsize=20, y=0.95)

        for c in range(3):
            axes[c, 0].imshow(full_gt[c], cmap='gray', vmin=0, vmax=1)
            axes[c, 0].set_title(f'GT: {class_names[c]}', fontsize=14)
            axes[c, 0].axis('off')

            im_pred = axes[c, 1].imshow(full_pred[c], cmap='jet', vmin=0, vmax=1)
            axes[c, 1].set_title(f'Predicted: {class_names[c]}', fontsize=14)
            axes[c, 1].axis('off')

            im_unc = axes[c, 2].imshow(full_unc[c], cmap='magma', vmin=0, vmax=1)
            axes[c, 2].set_title(f'Uncertainty (u): {class_names[c]}', fontsize=14)
            axes[c, 2].axis('off')

        cbar_ax1 = fig.add_axes([0.92, 0.4, 0.02, 0.2])
        fig.colorbar(im_pred, cax=cbar_ax1, orientation='vertical', label='Probability')
        cbar_ax2 = fig.add_axes([0.92, 0.1, 0.02, 0.2])
        fig.colorbar(im_unc, cax=cbar_ax2, orientation='vertical', label='Uncertainty')

        plt.subplots_adjust(wspace=0.1, hspace=0.2, right=0.9)
        save_path = os.path.join(args.save_dir, f"full_chip_per_class_{design_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Per-class map saved to: {save_path}")

    plt.close()

if __name__ == "__main__":
    main()
