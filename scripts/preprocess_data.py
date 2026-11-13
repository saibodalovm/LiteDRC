#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from litedrc.data.parse_features import preprocess_sample
from litedrc.data.parse_drc import build_class_config, build_drc_targets, parse_def_grid_and_cells, build_cell_density_map

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess EDA dataset into .npz files.")
    parser.add_argument("--targets-root", type=Path, required=True, help="Root directory containing DEF and DRC files.")
    parser.add_argument("--features-root", type=Path, required=True, help="Root directory containing RUDY and Pin Density files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for processed .npz files.")
    parser.add_argument("--val-size", type=float, default=0.15, help="Fraction of designs to use for validation (Leave-Design-Out).")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for train/val split.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Step 1: Scanning dataset directories...")
    runs = []
    
    # Scan targets root for DRC reports
    for drc_path in args.targets_root.glob("**/drc_report.txt"):
        # run_targets_dir = drc_path.parents[2]  # Adjust based on your directory structure
        # In the original script it was: drc_path.parents[2]
        # structure seems to be: PDK/design/run/route/drc_report/drc_report.txt
        run_targets_dir = drc_path.parents[2]
        def_file = run_targets_dir / "route" / "def" / "def.def"
        
        if def_file.exists():
            rel_path = run_targets_dir.relative_to(args.targets_root)
            run_features_dir = args.features_root / rel_path
            
            runs.append({
                "targets_dir": run_targets_dir,
                "features_dir": run_features_dir,
                "rel_path": rel_path
            })
            
    print(f"Found {len(runs)} valid runs.")
    if len(runs) == 0:
        print("[ERROR] No valid folders found! Check --targets-root and --features-root.")
        return

    dataset_records = []
    for idx, run_info in enumerate(runs):
        rel_parts = run_info["rel_path"].parts  # (PDK, design, run)
        pdk_name = rel_parts[0]
        design_name = rel_parts[1]
        run_name = rel_parts[2]
        
        dataset_records.append({
            "sample_id": f"sample_{idx:05d}",
            "targets_dir": str(run_info["targets_dir"]),
            "features_dir": str(run_info["features_dir"]),
            "pdk": pdk_name,
            "design": design_name,
            "run": run_name,
            "prefix": f"{pdk_name}_{design_name}_{run_name}"
        })
        
    df = pd.DataFrame(dataset_records)

    print("Step 2: Splitting dataset (Leave-Design-Out)...")
    unique_designs = df["design"].unique()
    
    train_designs, val_designs = train_test_split(
        unique_designs, 
        test_size=args.val_size, 
        random_state=args.random_state
    )
    
    print(f"Designs in Train: {len(train_designs)}")
    print(f"Designs in Val (unseen): {len(val_designs)}")

    df.loc[df["design"].isin(train_designs), "split"] = "train"
    df.loc[df["design"].isin(val_designs), "split"] = "val"

    for split in ["train", "val"]:
        (args.output_dir / split).mkdir(parents=True, exist_ok=True)

    print("Step 3: Processing features and generating targets...")
    success_count = 0
    class_names, class_to_idx = build_class_config()
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = row["sample_id"]
        split = row["split"]
        prefix = row["prefix"]
        
        targets_path = Path(row["targets_dir"])
        features_path = Path(row["features_dir"])
        
        def_file = targets_path / "route" / "def" / "def.def"
        drc_file = targets_path / "route" / "drc_report" / "drc_report.txt"
        
        try:
            # 1. Load features from features root
            feat_data = preprocess_sample(features_path)
            H, W = feat_data["grid_shape"]
            
            # 2. Parse DEF and generate cell density from targets root
            parsed_def = parse_def_grid_and_cells(def_file)
            cell_density = build_cell_density_map(parsed_def, H, W)
            
            # 3. Build DRC targets
            targets = build_drc_targets(
                def_path=def_file,
                drc_path=drc_file,
                class_to_idx=class_to_idx,
                grid_shape=(H, W)
            )
            
            target_save_path = args.output_dir / split / f"{prefix}.npz"
            
            save_dict = {
                "rudy": feat_data["rudy"],
                "pin_density": feat_data["pin_density"],
                "pin_count": feat_data["pin_count"],
                "cell_area": feat_data["cell_area"],
                "cell_density": cell_density,
                "target_objectness": targets["objectness"],
                "target_class_map": targets["class_map"],
                "target_offset": targets["offset"],
                "target_offset_mask": targets["offset_mask"]
            }
            
            if feat_data.get("placement_density") is not None:
                save_dict["placement_density"] = feat_data["placement_density"]
                
            np.savez_compressed(target_save_path, **save_dict)
            success_count += 1
            
        except Exception as e:
            print(f"\n[ERROR] Failed to process {prefix}: {e}")
            df.loc[df["sample_id"] == sample_id, "split"] = "error"

    df.to_csv(args.output_dir / "dataset_registry.csv", index=False)
    print(f"\nSuccessfully processed: {success_count}/{len(df)} samples.")
    print(f"Registry saved to: {args.output_dir / 'dataset_registry.csv'}")

if __name__ == "__main__":
    main()
