import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import re


def get_def_metadata(def_file_path):
    """Динамически достает DBU, размеры сетки и физический шаг GCell из шапки .def"""
    dbu_per_micron = 1000.0  # Дефолт на всякий случай
    grid_shape = [0, 0]      # [Строки(Y), Колонки(X)]
    step_x = 0.0
    
    with open(def_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith("UNITS DISTANCE MICRONS"):
                # Пример: UNITS DISTANCE MICRONS 2000 ;
                dbu_per_micron = float(line.split()[3])
                
            elif line.startswith("GCELLGRID X"):
                # Пример: GCELLGRID X 0 DO 228 STEP 16200 ;
                parts = line.split()
                grid_shape[1] = int(parts[4])  # Ширина
                step_x = float(parts[6])       # Шаг в DBU
                
            elif line.startswith("GCELLGRID Y"):
                parts = line.split()
                grid_shape[0] = int(parts[4])  # Высота
                
            # Оптимизация: если дошли до секции COMPONENTS, значит шапка кончилась
            if line.startswith("COMPONENTS") or line.startswith("SPECIALNETS"):
                break
                
    gcell_size_um = step_x / dbu_per_micron if dbu_per_micron else 0.0
    return dbu_per_micron, tuple(grid_shape), gcell_size_um

# ============================================================================
# 2. ИЗВЛЕЧЕНИЕ КООРДИНАТ ПИТАНИЯ
# ============================================================================
def extract_power_grids(def_file_path, dbu_per_micron):
    in_specialnets = False
    current_net = None
    current_layer = "Unknown"
    
    coord_pattern = re.compile(r'\(\s*(-?\d+|\*)\s+(-?\d+|\*)\s*\)')
    power_shapes = {"VDD": [], "VSS": []}
    last_x, last_y = 0.0, 0.0
    
    with open(def_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            if line.startswith("SPECIALNETS"):
                in_specialnets = True
                continue
            if in_specialnets and line.startswith("END SPECIALNETS"):
                break
                
            if in_specialnets:
                if line.startswith("- VDD") or line.startswith("- VSS"):
                    current_net = line.split()[1]
                    current_layer = "Unknown"
                    continue
                elif line.startswith("- "):
                    current_net = None
                    continue
                    
                if current_net:
                    tokens = line.split()
                    for i, tok in enumerate(tokens):
                        if tok in ["ROUTED", "NEW"] and i + 1 < len(tokens):
                            current_layer = tokens[i+1]
                    
                    raw_coords = coord_pattern.findall(line)
                    if raw_coords:
                        resolved_coords = []
                        for cx, cy in raw_coords:
                            x = last_x if cx == '*' else float(cx) / dbu_per_micron
                            y = last_y if cy == '*' else float(cy) / dbu_per_micron
                            resolved_coords.append((x, y))
                            last_x, last_y = x, y
                        
                        if len(resolved_coords) >= 2 and current_layer != "Unknown":
                            power_shapes[current_net].append({
                                "layer": current_layer,
                                "coords": resolved_coords
                            })
    return power_shapes

# ============================================================================
# 3. РАСТЕРИЗАЦИЯ НА СЕТКУ GCELL
# ============================================================================
def rasterize_power_shapes(power_shapes, grid_shape, gcell_size_um, ignore_layers):
    vdd_mask = np.zeros(grid_shape, dtype=bool)
    gnd_mask = np.zeros(grid_shape, dtype=bool)
    
    def draw_lines(shapes_list, out_mask):
        for shape in shapes_list:
            if shape['layer'] in ignore_layers:
                continue
                
            coords = shape['coords']
            for i in range(len(coords) - 1):
                x1_um, y1_um = coords[i]
                x2_um, y2_um = coords[i+1]
                
                col1, col2 = int(x1_um / gcell_size_um), int(x2_um / gcell_size_um)
                row1, row2 = int(y1_um / gcell_size_um), int(y2_um / gcell_size_um)
                
                min_r, max_r = max(0, min(row1, row2)), min(grid_shape[0]-1, max(row1, row2))
                min_c, max_c = max(0, min(col1, col2)), min(grid_shape[1]-1, max(col1, col2))
                
                out_mask[min_r:max_r+1, min_c:max_c+1] = True

    draw_lines(power_shapes.get("VDD", []), vdd_mask)
    draw_lines(power_shapes.get("VSS", []), gnd_mask)
    return [vdd_mask], [gnd_mask]

# ============================================================================
# 4. ПОСТРОЕНИЕ КАРТ РАССТОЯНИЙ (АЛГОРИТМ КЛОДА)
# ============================================================================
def build_pg_distance_channels(vdd_masks_per_layer, gnd_masks_per_layer, max_dist_gcells=10.0):
    def _union_then_distance(masks_per_layer):
        union_mask = np.zeros_like(masks_per_layer[0], dtype=bool)
        for m in masks_per_layer:
            union_mask |= (m > 0)
            
        if not union_mask.any():
            return np.full(union_mask.shape, max_dist_gcells, dtype=np.float32)
            
        dist = distance_transform_edt(~union_mask)
        dist = np.clip(dist, 0, max_dist_gcells)
        return (dist / max_dist_gcells).astype(np.float32)
        
    return _union_then_distance(vdd_masks_per_layer), _union_then_distance(gnd_masks_per_layer)

# ============================================================================
# 5. ОСНОВНОЙ БЛОК: ТЕСТИРОВАНИЕ И ОТРИСОВКА
# ============================================================================
if __name__ == "__main__":
    # ---> ТУТ МОЖЕШЬ ПОДСТАВИТЬ ЛЮБОЙ ФАЙЛ (GF180 или Sky130) <---
    #DEF_FILE = r"J:\dataset_Nikita_v1\dataset\gf180\aes_core\CLK_17.5_IO_0.00_CU_20_AR_1.0_HW_4_HS_4_HP_32_VW_4_VS_4_VP_32\prects\def\def.def"
    DEF_FILE = r"J:\dataset_Nikita_v1\dataset\sky130\tv80s\CLK_40.5_IO_0.00_CU_20_AR_0.5_HW_4_HS_4_HP_32_VW_4_VS_4_VP_32\prects\def\def.def"
    
    print(f"Открываем файл: {DEF_FILE}")
    
    # 1. Читаем динамические параметры чипа
    dbu, grid_shape, gcell_um = get_def_metadata(DEF_FILE)
    print(f"[*] Метаданные: DBU={dbu}, Сетка={grid_shape}, Шаг={gcell_um:.2f} мкм")
    
    # 2. Вытаскиваем пути (с учетом DBU)
    shapes = extract_power_grids(DEF_FILE, dbu_per_micron=dbu)
    
    # 3. Растеризуем, игнорируя нижние слои для контрастности
    # Поддерживаем и GF180 (Metal1/2) и Sky130 (met1/2, li1)
    ignore_list = ['Metal1', 'Metal2', 'met1', 'met2', 'li1']
    vdd_masks, gnd_masks = rasterize_power_shapes(
        power_shapes=shapes, 
        grid_shape=grid_shape, 
        gcell_size_um=gcell_um, 
        ignore_layers=ignore_list
    )
    
    # 4. Считаем градиенты. max_dist_gcells=2.0 "растянет" контраст 
    # для плотных сеток
    vdd_dist, gnd_dist = build_pg_distance_channels(vdd_masks, gnd_masks, max_dist_gcells=2.0)
    
    print("\nОтрисовываем результаты...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(vdd_dist, cmap='magma', origin='lower')
    axes[0].set_title(f'VDD Macro-Grid (DBU={dbu})', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('X (GCells)')
    axes[0].set_ylabel('Y (GCells)')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='Dist (0.0=VDD, 1.0=Далеко)')
    
    im2 = axes[1].imshow(gnd_dist, cmap='magma', origin='lower')
    axes[1].set_title(f'VSS Macro-Grid (DBU={dbu})', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('X (GCells)')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Dist (0.0=VSS, 1.0=Далеко)')
    
    plt.tight_layout()
    #plt.savefig('power_grid_universal.png', dpi=300, bbox_inches='tight')
    print("Успех! Картинка сохранена как 'power_grid_universal.png'.")
    plt.show()