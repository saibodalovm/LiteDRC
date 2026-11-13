import os
import numpy as np
from pathlib import Path
from collections import defaultdict

# Настрой пути к датасету
DATASET_DIR = Path("J:/processed_dataset/train") # Укажи свой корень датасета

def analyze_drc_classes():
    stats = defaultdict(lambda: defaultdict(int))
    
    # Ищем все файлы (поменяй расширение на .npy, если у тебя они)
    files = list(DATASET_DIR.rglob("*.npz"))
    print(f"Найдено файлов: {len(files)}")
    
    for file_path in files:
        # Определяем PDK по пути (например, gf180 или sky130)
        if "gf180" in str(file_path).lower():
            pdk = "GF180"
        elif "sky130" in str(file_path).lower():
            pdk = "SKY130"
        else:
            continue
            
        try:
            data = np.load(file_path)
            # Берем целевую маску (убедись, что ключ правильный, как в __getitem__)
            cls_map = data["target_class_map"] 
            
            # Считаем количество пикселей каждого класса
            unique, counts = np.unique(cls_map, return_counts=True)
            for cls_idx, count in zip(unique, counts):
                stats[pdk][cls_idx] += count
                
        except Exception as e:
            print(f"Ошибка чтения {file_path.name}: {e}")

    # Вывод красивой таблицы
    print("\n" + "="*70)
    print(f"{'PDK':<10} | {'Class 0 (Фон)':<15} | {'Class 1 (Short)':<15} | {'Class 2 (Spacing)':<15} | {'Class 3 (MinArea/Other)':<15}")
    print("="*70)
    
    for pdk in ["GF180", "SKY130"]:
        if pdk not in stats:
            continue
            
        pdk_stats = stats[pdk]
        total_pixels = sum(pdk_stats.values())
        total_defects = total_pixels - pdk_stats.get(0, 0)
        
        c0 = pdk_stats.get(0, 0)
        c1 = pdk_stats.get(1, 0)
        c2 = pdk_stats.get(2, 0)
        c3 = pdk_stats.get(3, 0)
        
        # Проценты от общего числа дефектов (чтобы понять баланс между самими дефектами)
        p1 = (c1 / total_defects * 100) if total_defects > 0 else 0
        p2 = (c2 / total_defects * 100) if total_defects > 0 else 0
        p3 = (c3 / total_defects * 100) if total_defects > 0 else 0
        
        print(f"{pdk:<10} | {c0:<15} | {c1:<6} ({p1:>4.1f}%) | {c2:<6} ({p2:>4.1f}%) | {c3:<6} ({p3:>4.1f}%)")
        print("-" * 70)

if __name__ == "__main__":
    analyze_drc_classes()