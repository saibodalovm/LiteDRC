import re

def extract_power_grids(def_file_path, dbu_per_micron=2000.0):
    in_specialnets = False
    current_net = None
    current_layer = "Unknown"
    
    # Регулярка для извлечения координат (числа или звездочка)
    coord_pattern = re.compile(r'\(\s*(-?\d+|\*)\s+(-?\d+|\*)\s*\)')
    power_shapes = {"VDD": [], "VSS": []}
    
    # Храним последние координаты для распаковки звездочек
    last_x, last_y = 0.0, 0.0
    
    print(f"Открываем {def_file_path}...")
    
    with open(def_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # Ловим границы секции
            if line.startswith("SPECIALNETS"):
                in_specialnets = True
                print("Нашли секцию SPECIALNETS! Парсим...")
                continue
            if in_specialnets and line.startswith("END SPECIALNETS"):
                print("Секция SPECIALNETS закончилась.")
                break
                
            if in_specialnets:
                # Определяем сеть
                if line.startswith("- VDD") or line.startswith("- VSS"):
                    current_net = line.split()[1] # Вытащит "VDD" или "VSS"
                    current_layer = "Unknown"
                    continue
                elif line.startswith("- "):
                    current_net = None
                    continue
                    
                if current_net:
                    tokens = line.split()
                    
                    # 1. Ищем обновление слоя (ключевые слова ROUTED или NEW)
                    for i, tok in enumerate(tokens):
                        if tok in ["ROUTED", "NEW"] and i + 1 < len(tokens):
                            current_layer = tokens[i+1]
                    
                    # 2. Ищем координаты
                    raw_coords = coord_pattern.findall(line)
                    if raw_coords:
                        resolved_coords = []
                        for cx, cy in raw_coords:
                            # Обрабатываем * (звездочку) и переводим в микроны
                            x = last_x if cx == '*' else float(cx) / dbu_per_micron
                            y = last_y if cy == '*' else float(cy) / dbu_per_micron
                            
                            resolved_coords.append((x, y))
                            last_x, last_y = x, y
                        
                        # Добавляем в словарь только если это отрезок (2 и более точек)
                        if len(resolved_coords) >= 2 and current_layer != "Unknown":
                            power_shapes[current_net].append({
                                "layer": current_layer,
                                "coords": resolved_coords
                            })

    print(f"✅ Парсинг завершен! Отрезков VDD: {len(power_shapes['VDD'])}, VSS: {len(power_shapes['VSS'])}")
    return power_shapes

if __name__ == "__main__":
    # Замени на свой путь
    DEF_FILE = r"J:\dataset_Nikita_v1\dataset\gf180\aes_core\CLK_17.5_IO_0.00_CU_20_AR_1.0_HW_4_HS_4_HP_32_VW_4_VS_4_VP_32\prects\def\def.def"
    
    shapes = extract_power_grids(DEF_FILE)
    
    print("\nПример правильных линий VDD (в микронах):")
    for i, shape in enumerate(shapes["VDD"][:5]):
        print(f"Линия {i+1} | Слой: {shape['layer']:<8} | Точки: {shape['coords']}")
    from collections import Counter
    print("Слои VDD:", Counter([s['layer'] for s in shapes["VDD"]]))
    print("Слои VSS:", Counter([s['layer'] for s in shapes["VSS"]]))