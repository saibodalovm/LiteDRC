import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# --- 1. ПАРСИНГ DEF (Реальная сетка и ячейки) ---
def parse_def_grid_and_cells(def_path: Path):
    gcellgrid_x = None  # (start, num, step)
    gcellgrid_y = None
    die_area = None
    units = 1000
    cells = []
    in_components = False
    
    with open(def_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith('UNITS DISTANCE MICRONS'):
                units = int(line.split()[-2])
                continue
            if line.startswith('DIEAREA'):
                nums = list(map(int, re.findall(r'-?\d+', line)))
                die_area = (nums[0], nums[1], nums[2], nums[3])
                continue
            if line.startswith('GCELLGRID X'):
                nums = list(map(int, re.findall(r'-?\d+', line)))
                gcellgrid_x = (nums[0], nums[1], nums[2])
                continue
            if line.startswith('GCELLGRID Y'):
                nums = list(map(int, re.findall(r'-?\d+', line)))
                gcellgrid_y = (nums[0], nums[1], nums[2])
                continue
            if line.startswith('COMPONENTS '):
                in_components = True
                continue
            if line.startswith('END COMPONENTS'):
                in_components = False
                continue
            if in_components and line.startswith('-'):
                m = re.search(r'\(\s*(-?\d+)\s+(-?\d+)\s*\)', line)
                if m:
                    cells.append((int(m.group(1)), int(m.group(2))))
                    
    return {
        'units_per_micron': units,
        'die_area': die_area,
        'gcellgrid_x': gcellgrid_x,
        'gcellgrid_y': gcellgrid_y,
        'cells': cells,
    }

def build_gcell_density_map(parsed):
    gx, gy = parsed['gcellgrid_x'], parsed['gcellgrid_y']
    nx, ny = gx[1], gy[1]
    step_x, step_y = gx[2], gy[2]
    origin_x, origin_y = gx[0], gy[0]
    
    density = np.zeros((ny, nx), dtype=np.float32)
    for (x, y) in parsed['cells']:
        col = int((x - origin_x) // step_x)
        row = int((y - origin_y) // step_y)
        if 0 <= row < ny and 0 <= col < nx:
            density[row, col] += 1
            
    return density

# --- 2. ПАРСИНГ DRC И ПРЕОБРАЗОВАНИЕ ---
def parse_drc_report(drc_path: Path):
    violations = []
    current_type = None
    with open(drc_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('violation type:'):
            current_type = line.split('violation type:')[1].strip()
        elif line.startswith('bbox'):
            m = re.search(r'bbox = \(([\d.]+), ([\d.]+)\) - \(([\d.]+), ([\d.]+)\) on Layer (\S+)', line)
            if m:
                x0, y0, x1, y1, layer = m.groups()
                x0, y0, x1, y1 = map(float, (x0, y0, x1, y1))
                violations.append({
                    'type': current_type, 'layer': layer,
                    'cx': (x0 + x1) / 2, 'cy': (y0 + y1) / 2,
                })
        i += 1
    return violations

def micron_to_gcell(cx_um, cy_um, units_per_micron, gx, gy):
    x_defunits = cx_um * units_per_micron
    y_defunits = cy_um * units_per_micron
    col = (x_defunits - gx[0]) / gx[2]
    row = (y_defunits - gy[0]) / gy[2]
    return col, row

def draw_gaussian(heatmap, center, radius=1.0):
    h, w = heatmap.shape
    cx, cy = center
    sigma = max(radius / 3, 0.3)
    y0, y1 = max(0, int(cy - 3*sigma)), min(h, int(cy + 3*sigma) + 1)
    x0, x1 = max(0, int(cx - 3*sigma)), min(w, int(cx + 3*sigma) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    gaussian = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], gaussian)

# --- 3. ГЛАВНАЯ ФУНКЦИЯ ---
def main():
    # Пути (измени, если нужно)
    base_dir = Path(r"J:\dataset_Nikita_v1\dataset\freepdk45\trigonometric\CLK_4.6_IO_0.00_CU_20_AR_1.0_HW_4_HS_4_HP_32_VW_4_VS_4_VP_32")
    def_file = base_dir / "route" / "def" / "def.def"
    drc_file = base_dir / "route" / "drc_report" / "drc_report.txt"

    if not def_file.exists() or not drc_file.exists():
        print("Ошибка путей!")
        return

    print("Парсим DEF...")
    parsed_def = parse_def_grid_and_cells(def_file)
    density = build_gcell_density_map(parsed_def)
    
    gx, gy = parsed_def['gcellgrid_x'], parsed_def['gcellgrid_y']
    nx, ny = gx[1], gy[1]
    units = parsed_def['units_per_micron']

    print("Парсим DRC...")
    violations = parse_drc_report(drc_file)
    print(f"Всего нарушений распарсено в памяти: {len(violations)}")
    
    if len(violations) == 0:
        print("Что-то пошло не так, нарушения не найдены!")
        return

    from collections import Counter
    raw_counts = Counter([v['layer'] for v in violations])
    # ФИКС: Динамически ищем, какие слои реально есть в файле
    all_layers = [v['layer'] for v in violations]
    layer_counts = Counter(all_layers)
    print(f"\nНайдено нарушений по слоям:\n{dict(layer_counts)}")

    # Берем Топ-5 слоев с самым большим числом ошибок для визуализации
    target_layers = [layer for layer, count in layer_counts.most_common(5)]
    print(f"\nВыбраны Топ-5 слоев для графиков: {target_layers}")

    heatmaps = {layer: np.zeros((ny, nx), dtype=np.float32) for layer in target_layers}

    out_of_bounds = 0
    for v in violations:
        if v['layer'] in heatmaps:
            col, row = micron_to_gcell(v['cx'], v['cy'], units, gx, gy)
            if 0 <= col < nx and 0 <= row < ny:
                draw_gaussian(heatmaps[v['layer']], (col, row), radius=1.0)
            else:
                out_of_bounds += 1

    print(f"Точек вне сетки (вылетели за край): {out_of_bounds}")

    print("Строим визуализацию...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()

    axes[0].imshow(density, cmap='gray', origin='lower', aspect='auto')
    axes[0].set_title('Input: Real Cell Density')

    for i, layer in enumerate(target_layers):
        ax = axes[i+1]
        hm = heatmaps[layer]
        
        ax.imshow(density, cmap='gray', origin='lower', aspect='auto', alpha=0.5)
        masked = np.ma.masked_where(hm < 0.05, hm)
        ax.imshow(masked, cmap='hot', origin='lower', aspect='auto', vmin=0, vmax=1)
        
        n_viol = (hm > 0.5).sum()
        raw_total = raw_counts[layer]
        ax.set_title(f'{layer}: {raw_total} raw -> ~{n_viol} spots')

    # Если уникальных слоев меньше 5, скрываем пустые графики
    for j in range(len(target_layers) + 1, 6):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig("drc_perfect_overlay.png", dpi=110)
    print("\nКарта сохранена как 'drc_perfect_overlay.png'!")

if __name__ == "__main__":
    main()