#!/usr/bin/env python3
import json
import argparse
import numpy as np
from pathlib import Path
import re

pattern = re.compile(
    r"^(?P<pdk>[a-zA-Z0-9]+)_(?P<design>.+?)_CLK_(?P<CLK>[\d\.]+)_IO_(?P<IO>[\d\.]+)_"
    r"CU_(?P<CU>[\d\.]+)_AR_(?P<AR>[\d\.]+)_HW_(?P<HW>[\d\.]+)_HS_(?P<HS>[\d\.]+)_"
    r"HP_(?P<HP>[\d\.]+)_VW_(?P<VW>[\d\.]+)_VS_(?P<VS>[\d\.]+)_VP_(?P<VP>[\d\.]+)"
)

def collect_global_zscore_stats(data_dir, output_path=None):
    data_dir = Path(data_dir)
    files = list(data_dir.glob("*.npz"))
    
    # Features to collect (IO is excluded)
    features = ["CLK", "CU", "AR", "HW", "HS", "HP", "VW", "VS", "VP"]
    values = {k: [] for k in features}
    
    print(f"Collecting global Z-score statistics from {len(files)} files in {data_dir}...")
    
    for f in files:
        match = pattern.search(f.name)
        if not match:
            continue
            
        d = match.groupdict()
        
        # 1. Special processing for CLK (asap7 correction and log1p)
        clk_raw = float(d["CLK"])
        if d["pdk"] == "asap7":
            clk_raw /= 1000.0
        values["CLK"].append(np.log1p(clk_raw))
        
        # 2. Collect other parameters
        for k in features[1:]:
            values[k].append(float(d[k]))
            
    if not values["CLK"]:
        print("[ERROR] No valid files found to collect statistics!")
        return

    # Calculate Mean and Std for Z-score
    stats = {"z_score": {}}
    for k in features:
        arr = np.array(values[k])
        std = float(np.std(arr))
        # Protection against zero variance
        if std == 0.0:
            std = 1.0 
            
        stats["z_score"][k] = {
            "mean": float(np.mean(arr)),
            "std": std
        }
    
    if output_path is None:
        output_path = data_dir / "global_zscore_stats.json"
    else:
        output_path = Path(output_path)
        
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"Global Z-score statistics saved to {output_path}!")

def parse_args():
    parser = argparse.ArgumentParser(description="Collect global Z-score statistics from processed .npz files.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing processed .npz files (usually 'train').")
    parser.add_argument("--output", type=Path, help="Output path for the .json file. Defaults to data-dir/global_zscore_stats.json.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    collect_global_zscore_stats(args.data_dir, args.output)
