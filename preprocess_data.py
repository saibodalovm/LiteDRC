# preprocess_dataset.py
import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from  utils.parse_features import preprocess_sample
from utils.parse_drc import build_class_config, build_drc_targets, parse_def_grid_and_cells, build_cell_density_map

# ---------------------------------------------------------------------------
# [ИСПРАВЛЕНО] Настройки директорий: теперь корни разделены
# ---------------------------------------------------------------------------
TARGETS_ROOT = Path(r"J:\dataset_Nikita_v1\dataset")                 # Тут ищем DEF и DRC
FEATURES_ROOT = Path(r"J:\dataset_Nikita_v1\dataset_features_prects") # Отсюда берем RUDY и Pin Density
PROCESSED_DIR = Path(r"J:\processed_dataset")                        # Сюда пишем готовые .npz

VAL_SIZE = 0.15  # 15% дизайнов уйдут в валидацию
RANDOM_STATE = 42

class_names, class_to_idx = build_class_config()


def main():
    print("Шаг 1: Сканирование директорий датасета...")
    runs = []
    
    # Сканируем таргетированный корень
    for drc_path in TARGETS_ROOT.glob("**/drc_report.txt"):
        run_targets_dir = drc_path.parents[2]  # Папка запуска в TARGETS_ROOT
        def_file = run_targets_dir / "route" / "def" / "def.def"
        
        if def_file.exists():
            # [ИСПРАВЛЕНО] Вычисляем относительный путь от корня
            rel_path = run_targets_dir.relative_to(TARGETS_ROOT)
            
            # Собираем пути к обоим источникам данных для этого запуска
            run_features_dir = FEATURES_ROOT / rel_path
            
            runs.append({
                "targets_dir": run_targets_dir,
                "features_dir": run_features_dir,
                "rel_path": rel_path
            })
            
    print(f"Найдено подходящих запусков: {len(runs)}")
    if len(runs) == 0:
        print("[ERROR] Не найдено подходящих папок! Проверь пути TARGETS_ROOT и FEATURES_ROOT.")
        return

    # Собираем метаданные для разбиения по дизайну
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

    # ---------------------------------------------------------------------------
    # [ИСПРАВЛЕНО] Leave-Design-Out (Group Split) по колонтитулу "design"
    # ---------------------------------------------------------------------------
    print("Шаг 2: Разбиение датасета (Leave-Design-Out)...")
    unique_designs = df["design"].unique()
    
    # Бьем именно УНИКАЛЬНЫЕ дизайны на train/val группы
    train_designs, val_designs = train_test_split(
        unique_designs, 
        test_size=VAL_SIZE, 
        random_state=RANDOM_STATE
    )
    
    print(f"Дизайнов в Train: {len(train_designs)} ({list(train_designs[:5])}...)" )
    print(f"Дизайнов в Val (невиданных в обучении): {len(val_designs)} ({list(val_designs[:5])}...)" )

    # Присваиваем сплиты строкам на основе принадлежности к дизайну
    df.loc[df["design"].isin(train_designs), "split"] = "train"
    df.loc[df["design"].isin(val_designs), "split"] = "val"

    # Создаем выходные папки
    for split in ["train", "val"]:
        (PROCESSED_DIR / split).mkdir(parents=True, exist_ok=True)

    # Шаг 3: Процессинг данных
    print("Шаг 3: Обработка фичей и генерация таргетов...")
    success_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = row["sample_id"]
        split = row["split"]
        prefix = row["prefix"]
        
        # Раздельные пути для фичей и таргетов
        targets_path = Path(row["targets_dir"])
        features_path = Path(row["features_dir"])
        
        def_file = targets_path / "route" / "def" / "def.def"
        drc_file = targets_path / "route" / "drc_report" / "drc_report.txt"
        
        try:
            # 1. Загружаем фичи из FEATURES_ROOT
            feat_data = preprocess_sample(features_path)
            H, W = feat_data["grid_shape"]
            
            df.loc[df["sample_id"] == sample_id, "H"] = H
            df.loc[df["sample_id"] == sample_id, "W"] = W
            
            # 2. Парсим DEF и генерируем плотность ячеек из TARGETS_ROOT
            parsed_def = parse_def_grid_and_cells(def_file)
            cell_density = build_cell_density_map(parsed_def, H, W)
            
            # 3. Строим DRC таргеты
            targets = build_drc_targets(
                def_path=def_file,
                drc_path=drc_file,
                class_to_idx=class_to_idx,
                grid_shape=(H, W)
            )
            
            # Имя архива
            target_save_path = PROCESSED_DIR / split / f"{prefix}.npz"
            
            # Пакуем всё
            save_dict = {
                "rudy": feat_data["rudy"],
                "pin_density": feat_data["pin_density"],
                "pin_count": feat_data["pin_count"],
                "cell_area": feat_data["cell_area"],
                "cell_density": cell_density,
                "placement_density": feat_data["placement_density"],
                "target_objectness": targets["objectness"],
                "target_class_map": targets["class_map"],
                "target_offset": targets["offset"],
                "target_offset_mask": targets["offset_mask"]
            }
            
            if feat_data["placement_density"] is not None:
                save_dict["placement_density"] = feat_data["placement_density"]
                
            np.savez_compressed(target_save_path, **save_dict)
            success_count += 1
            
        except Exception as e:
            print(f"\n[ERROR] Ошибка при обработке {prefix}: {e}")
            df.loc[df["sample_id"] == sample_id, "split"] = "error"

    # Сохраняем реестр (очень полезно, чтобы сразу видеть, где какой сплит и размеры)
    df.to_csv(PROCESSED_DIR / "dataset_registry.csv", index=False)
    print(f"\nУспешно обработано: {success_count}/{len(df)} сэмплов.")
    print(f"Реестр сохранен в: {PROCESSED_DIR / 'dataset_registry.csv'}")


if __name__ == "__main__":
    main()