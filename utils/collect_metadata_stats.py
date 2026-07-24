import json
import numpy as np
from pathlib import Path
import re

pattern = re.compile(
    r"^(?P<pdk>[a-zA-Z0-9]+)_(?P<design>.+?)_CLK_(?P<CLK>[\d\.]+)_IO_(?P<IO>[\d\.]+)_"
    r"CU_(?P<CU>[\d\.]+)_AR_(?P<AR>[\d\.]+)_HW_(?P<HW>[\d\.]+)_HS_(?P<HS>[\d\.]+)_"
    r"HP_(?P<HP>[\d\.]+)_VW_(?P<VW>[\d\.]+)_VS_(?P<VS>[\d\.]+)_VP_(?P<VP>[\d\.]+)"
)

def collect_global_zscore_stats(data_dir):
    data_dir = Path(data_dir)
    files = list(data_dir.glob("*.npz"))
    
    # Ключи, которые мы оставляем (IO выкинут)
    features = ["CLK", "CU", "AR", "HW", "HS", "HP", "VW", "VS", "VP"]
    values = {k: [] for k in features}
    
    print(f"Сбор глобальной Z-score статистики по {len(files)} файлам...")
    
    for f in files:
        match = pattern.search(f.name)
        if not match:
            continue
            
        d = match.groupdict()
        
        # 1. Спец. обработка для CLK (чиним asap7 и берем log1p)
        clk_raw = float(d["CLK"])
        if d["pdk"] == "asap7":
            clk_raw /= 1000.0
        values["CLK"].append(np.log1p(clk_raw))
        
        # 2. Остальные параметры просто собираем
        for k in features[1:]:
            values[k].append(float(d[k]))
            
    # Вычисляем Mean и Std для Z-score
    stats = {"z_score": {}}
    for k in features:
        arr = np.array(values[k])
        std = float(np.std(arr))
        # Защита от нулевой дисперсии (например для HW внутри одного сета, если вдруг)
        if std == 0.0:
            std = 1.0 
            
        stats["z_score"][k] = {
            "mean": float(np.mean(arr)),
            "std": std
        }
        
    with open(data_dir / "global_zscore_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    print("Глобальная Z-score статистика сохранена!")

if __name__ == "__main__":
    collect_global_zscore_stats(r"J:\processed_dataset\train")