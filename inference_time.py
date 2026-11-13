import time
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import copy

# Модель и утилиты парсинга
from models.EncoderCrossAttDerichle import EvidentialLightweightRegionNet as EvidentialLightweightRegionNetAttn 
from utils.parse_features import preprocess_sample
from utils.parse_drc import parse_def_grid_and_cells, build_cell_density_map

# 1. Корневые директории сырых данных
TARGETS_ROOT = Path(r"J:\dataset_Nikita_v1\dataset")                  # Здесь лежат def.def
FEATURES_ROOT = Path(r"J:\dataset_Nikita_v1\dataset_features_prects") # Здесь лежат сырые RUDY и Pin Density

# 2. Инициализация устройства и модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EvidentialLightweightRegionNetAttn(in_channels=3, num_classes=3, use_film=True).to(device) 
model.eval()

# Прогрев GPU
dummy = torch.randn(64, 3, 64, 64, device=device)
with torch.no_grad():
    for _ in range(10):
        _ = model(dummy)

# 3. Список бенчмарков (указываем PDK, имя дизайна и запуск)
benchmarks = [
    {
        "name": "Small (i2c)",
        "pdk": "sky130",
        "design": "i2c_master_top",
        "run": "CLK_21.2_IO_0.00_CU_20_AR_1.0_HW_4_HS_4_HP_32_VW_4_VS_4_VP_32",
        "t_route_sec": 222.0
    },
    {
        "name": "Medium (gng)",
        "pdk": "sky130",
        "design": "gng",
        "run": "CLK_31.2_IO_0.00_CU_30_AR_1.0_HW_4_HS_4_HP_32_VW_4_VS_4_VP_32",
        "t_route_sec": 843.25
    },
    {
        "name": "Large (idft)",
        "pdk": "sky130",
        "design": "idft_top",
        "run": "CLK_41.2_IO_0.00_CU_20_AR_0.5_HW_4_HS_4_HP_64_VW_4_VS_4_VP_64",
        "t_route_sec": 12038.13
    }
]

results = []
patch_size = 64

model_cpu = copy.deepcopy(model).to("cpu")
model_cpu.eval()

results = []
patch_size = 64

for item in benchmarks:
    rel_path = Path(item["pdk"]) / item["design"] / item["run"]
    features_dir = FEATURES_ROOT / rel_path
    def_file = TARGETS_ROOT / rel_path / "route" / "def" / "def.def"
    
    if not def_file.exists():
        print(f"[ERROR] Не найден DEF файл: {def_file}")
        continue
        
    print(f"Обработка дизайна: {item['name']}...")

    # --- ЭТАП 1: Извлечение физических признаков (Feature Extraction) ---
    t0 = time.perf_counter()
    
    feat_data = preprocess_sample(features_dir)
    H, W = feat_data["grid_shape"]
    parsed_def = parse_def_grid_and_cells(def_file)
    cell_density = build_cell_density_map(parsed_def, H, W)
    
    # Сборка 3D массива признаков
    F = np.stack([feat_data["rudy"], feat_data["pin_density"], cell_density], axis=0).astype(np.float32)
    
    t_feat = time.perf_counter() - t0
    
    # --- ЭТАП 2: Нарезка полного чипа на регулярную сетку патчей ---
    t_tile_start = time.perf_counter()
    
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        F = np.pad(F, ((0,0), (0, pad_h), (0, pad_w)), mode='constant')
        
    _, H_pad, W_pad = F.shape
    patches = []
    for y in range(0, H_pad, patch_size):
        for x in range(0, W_pad, patch_size):
            patches.append(F[:, y:y+patch_size, x:x+patch_size])
            
    patches_np = np.stack(patches)
    batch_gpu = torch.from_numpy(patches_np).float().to(device)
    batch_cpu = torch.from_numpy(patches_np).float().to("cpu")
    t_tile = time.perf_counter() - t_tile_start
    
    # --- ЭТАП 3: Инференс нейросети на GPU (батчем) ---
    infer_runs_gpu = []
    with torch.no_grad():
        for _ in range(30):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_inf_start = time.perf_counter()
            
            _ = model(batch_gpu)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_runs_gpu.append(time.perf_counter() - t_inf_start)
            
    t_infer_gpu = np.mean(infer_runs_gpu)
    
    # --- ЭТАП 4: Инференс нейросети на CPU (батчем) ---
    infer_runs_cpu = []
    with torch.no_grad():
        for _ in range(10): # 10 прогонов для CPU достаточно
            t_inf_cpu_start = time.perf_counter()
            _ = model_cpu(batch_cpu)
            infer_runs_cpu.append(time.perf_counter() - t_inf_cpu_start)
            
    t_infer_cpu = np.mean(infer_runs_cpu)
    
    # --- Итоги ---
    t_ml_total_gpu = t_feat + t_tile + t_infer_gpu
    t_ml_total_cpu = t_feat + t_tile + t_infer_cpu
    
    speedup_gpu = item["t_route_sec"] / t_ml_total_gpu if t_ml_total_gpu > 0 else 0
    speedup_cpu = item["t_route_sec"] / t_ml_total_cpu if t_ml_total_cpu > 0 else 0
    
    results.append({
        "Design": item["name"],
        "Grid Size": f"{H}x{W}",
        "Patches": len(patches),
        "Routing+DRC (s)": round(item["t_route_sec"], 2),
        "Feat Extr (s)": round(t_feat + t_tile, 3),
        "GPU Infer (ms)": round(t_infer_gpu * 1000, 2),
        "CPU Infer (ms)": round(t_infer_cpu * 1000, 2),
        "Total ML GPU (s)": round(t_ml_total_gpu, 3),
        "Speedup GPU": f"{round(speedup_gpu, 1)}x",
        "Speedup CPU": f"{round(speedup_cpu, 1)}x"
    })

df_runtime = pd.DataFrame(results)
print("\n" + df_runtime.to_markdown(index=False))