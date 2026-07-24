import numpy as np
from pathlib import Path

def validate_dataset_shapes(data_dir, min_size=64):  # Поставил 64, так как наш патч теперь 64x64
    data_dir = Path(data_dir)
    files = list(data_dir.glob("*.npz"))
    
    print(f"Проверяю {len(files)} файлов и собираю глобальную статистику...")
    count_small = 0
    
    # Инициализируем переменные для поиска глобальных максимумов
    max_rudy = 0.0
    max_pin = 0.0
    max_placement = 0.0
    
    for f in files:
        data = np.load(f)
        
        keys_2d = ["rudy", "pin_density", "placement_density", "target_objectness", "target_class_map", "target_offset_mask"]
        base_shape = data["rudy"].shape
        
        # 1. Проверка размерностей 2D слоев
        for k in keys_2d:
            if data[k].shape != base_shape:
                print(f"Ошибка в {f.name}: ключ {k} имеет размер {data[k].shape}, ожидалось {base_shape}")
        
        # 2. Проверка 3D offset
        if data["target_offset"].shape != (2, base_shape[0], base_shape[1]):
            print(f"Ошибка в {f.name}: target_offset имеет размер {data['target_offset'].shape}, ожидалось (2, {base_shape[0]}, {base_shape[1]})")
            
        # 3. Подсчет мелких дизайнов
        if base_shape[0] < min_size or base_shape[1] < min_size:
            count_small += 1
            
        # 4. Фиксация максимумов во всем файле
        max_rudy = max(max_rudy, float(np.max(data["rudy"])))
        max_pin = max(max_pin, float(np.max(data["pin_density"])))
        max_placement = max(max_placement, float(np.max(data["placement_density"])))

    print("\n" + "="*50)
    print("=== ИТОГОВЫЕ РЕЗУЛЬТАТЫ АНАЛИЗА ДАТАСЕТА ===")
    print("="*50)
    print(f"Всего проверено файлов: {len(files)}")
    print(f"Маленьких дизайнов (< {min_size}x{min_size}): {count_small}")
    print("-"*50)
    print(f"Глобальный максимум RUDY:              {max_rudy:.6f}")
    print(f"Глобальный максимум PIN DENSITY:       {max_pin:.6f}")
    print(f"Глобальный максимум PLACEMENT DENSITY: {max_placement:.6f}")
    print("="*50)

# Запуск по твоей тренировочной папке
validate_dataset_shapes(r"J:\processed_dataset\train", min_size=64)
#validate_dataset_shapes(r"J:\processed_dataset\val")