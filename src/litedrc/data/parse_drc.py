"""
Генерация таргетов для CenterNet-style 3-head модели DRC detection:
  1. objectness heatmap (H,W)      -> focal loss
  2. class map (H,W), int          -> softmax/cross-entropy (только в позитивных точках)
  3. offset map (2,H,W) + mask     -> L1 loss (только в позитивных точках)

Важно: список классов ДОЛЖЕН быть посчитан один раз по всему датасету
(build_global_class_list), а не по одному сэмплу — иначе class_id будет
означать разные вещи в разных сэмплах.
"""

import re
import numpy as np
from pathlib import Path
from collections import Counter


# ---------------------------------------------------------------------------
# 1. Парсинг DEF: сетка + (опционально) реальная плотность ячеек
# ---------------------------------------------------------------------------
def parse_def_grid_and_cells(def_path: Path):
    gcellgrid_x = None
    gcellgrid_y = None
    units = 1000
    cells = []
    in_components = False
    pending_inst = False  # на случай, если "- inst CELL" и "+ PLACED (...)" на разных строках

    with open(def_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("UNITS DISTANCE MICRONS"):
                units = int(line.split()[-2])
                continue
            if line.startswith("GCELLGRID X"):
                nums = list(map(int, re.findall(r"-?\d+", line)))
                gcellgrid_x = (nums[0], nums[1], nums[2])
                continue
            if line.startswith("GCELLGRID Y"):
                nums = list(map(int, re.findall(r"-?\d+", line)))
                gcellgrid_y = (nums[0], nums[1], nums[2])
                continue
            if line.startswith("COMPONENTS "):
                in_components = True
                continue
            if line.startswith("END COMPONENTS"):
                in_components = False
                continue
            if in_components and line.startswith("-"):
                m = re.search(r"\(\s*(-?\d+)\s+(-?\d+)\s*\)", line)
                if m:
                    cells.append((int(m.group(1)), int(m.group(2))))
                    pending_inst = False
                else:
                    # координаты могут быть на следующей строке (+ PLACED ( x y ) ...)
                    pending_inst = True
                continue
            if in_components and pending_inst:
                m = re.search(r"\(\s*(-?\d+)\s+(-?\d+)\s*\)", line)
                if m:
                    cells.append((int(m.group(1)), int(m.group(2))))
                pending_inst = False

    if gcellgrid_x is None or gcellgrid_y is None:
        raise ValueError(f"{def_path}: не найден GCELLGRID X/Y")

    return {
        "units_per_micron": units,
        "gcellgrid_x": gcellgrid_x,
        "gcellgrid_y": gcellgrid_y,
        "cells": cells,
    }


def grid_shape_from_def(parsed):
    """ВАЖНО: DO n в GCELLGRID — это число линий сетки, число ячеек = n - 1.
    Здесь этот сдвиг учтён (в отличие от исходного скрипта)."""
    gx, gy = parsed["gcellgrid_x"], parsed["gcellgrid_y"]
    nx = gx[1] - 1
    ny = gy[1] - 1
    return ny, nx  # (H, W)


def build_cell_density_map(parsed, H, W):
    gx, gy = parsed["gcellgrid_x"], parsed["gcellgrid_y"]
    step_x, step_y = gx[2], gy[2]
    origin_x, origin_y = gx[0], gy[0]

    density = np.zeros((H, W), dtype=np.float32)
    for (x, y) in parsed["cells"]:
        col = int((x - origin_x) // step_x)
        row = int((y - origin_y) // step_y)
        if 0 <= row < H and 0 <= col < W:
            density[row, col] += 1
    return density


# ---------------------------------------------------------------------------
# 2. Парсинг DRC-репорта
# ---------------------------------------------------------------------------
def parse_drc_report(drc_path: Path):
    violations = []
    current_type = None
    with open(drc_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("violation type:"):
            current_type = line.split("violation type:")[1].strip()
        elif line.startswith("bbox"):
            m = re.search(
                r"bbox = \(([\d.]+), ([\d.]+)\) - \(([\d.]+), ([\d.]+)\) on Layer (\S+)",
                line,
            )
            if m:
                x0, y0, x1, y1, layer = m.groups()
                x0, y0, x1, y1 = map(float, (x0, y0, x1, y1))
                violations.append(
                    {
                        "type": current_type,
                        "layer": layer,
                        "cx": (x0 + x1) / 2,
                        "cy": (y0 + y1) / 2,
                    }
                )
    return violations


# ---------------------------------------------------------------------------
# 3. Список классов — фиксирован по статистике всего датасета.
#    По вашему распределению: Metal Spacing и Short — доминирующие классы
#    (581528 и 569569 соответственно), всё остальное (Cut Spacing, Lef58EolKeepOut,
#    Cut Short, NS Metal, Recheck, ...) — на 1-2 порядка реже, объединяем в "others".
#    Легко расширить: просто добавьте нужный type в FOREGROUND_CLASSES.
# ---------------------------------------------------------------------------
FOREGROUND_CLASSES = ["Metal Spacing", "Short"]
OTHERS_LABEL = "others"


def build_class_config(foreground_classes=FOREGROUND_CLASSES):
    class_names = list(foreground_classes) + [OTHERS_LABEL]
    class_to_idx = {name: i + 1 for i, name in enumerate(class_names)}  # 0 = background
    return class_names, class_to_idx


def classify(violation_type: str, class_to_idx: dict) -> str:
    return violation_type if violation_type in class_to_idx else OTHERS_LABEL


# ---------------------------------------------------------------------------
# 4. Координаты микроны -> дробные GCell
# ---------------------------------------------------------------------------
def micron_to_gcell(cx_um, cy_um, units_per_micron, gx, gy):
    x_defunits = cx_um * units_per_micron
    y_defunits = cy_um * units_per_micron
    col = (x_defunits - gx[0]) / gx[2]
    row = (y_defunits - gy[0]) / gy[2]
    return col, row


def draw_gaussian_max(heatmap, center, radius=1.0):
    h, w = heatmap.shape
    cx, cy = center
    sigma = max(radius / 3, 0.3)
    y0, y1 = max(0, int(cy - 3 * sigma)), min(h, int(cy + 3 * sigma) + 1)
    x0, x1 = max(0, int(cx - 3 * sigma)), min(w, int(cx + 3 * sigma) + 1)
    if y1 <= y0 or x1 <= x0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], gaussian)


# ---------------------------------------------------------------------------
# 5. Генерация таргетов для одного сэмпла
# ---------------------------------------------------------------------------
def build_drc_targets(
    def_path: Path,
    drc_path: Path,
    class_to_idx: dict,
    radius: float = 3.0,
    grid_shape: tuple | None = None,
):
    """
    grid_shape: (H, W) — если передать сетку, уже построенную из
    rudy_congestion.txt (см. предыдущий скрипт), таргет гарантированно
    совпадёт по форме и выравниванию с входными фичами. Если None —
    сетка берётся из DEF (с учётом off-by-one фикса), но тогда лучше
    сверить H,W с фичами отдельно (assert), т.к. это два независимых парсера.
    """
    parsed = parse_def_grid_and_cells(def_path)
    if grid_shape is None:
        H, W = grid_shape_from_def(parsed)
    else:
        H, W = grid_shape

    gx, gy = parsed["gcellgrid_x"], parsed["gcellgrid_y"]
    units = parsed["units_per_micron"]

    violations = parse_drc_report(drc_path)

    objectness = np.zeros((H, W), dtype=np.float32)
    class_map = np.zeros((H, W), dtype=np.int64)  # 0 = background
    offset_map = np.zeros((2, H, W), dtype=np.float32)
    offset_mask = np.zeros((H, W), dtype=bool)

    n_out_of_bounds = 0
    n_dropped_no_type = 0
    n_class_collisions = 0
    class_hits = Counter()

    # Один и тот же физический дефект нередко репортится DRC-чекером несколько
    # раз подряд с чуть разными bbox (одна и та же пара сетей, почти одинаковые
    # координаты) — если писать class/offset "последней попавшей в ячейку"
    # записью, offset будет зависеть от порядка строк в файле, а не от
    # геометрии. Поэтому сначала группируем все нарушения по дискретной
    # GCell-ячейке, в которую они попадают, и берём среднюю дробную координату
    # внутри группы — так offset стабилен и не зависит от количества дублей.
    cell_groups: dict[tuple[int, int], list] = {}
    for v in violations:
        if v["type"] is None:
            n_dropped_no_type += 1
            continue
        col, row = micron_to_gcell(v["cx"], v["cy"], units, gx, gy)
        if not (0 <= col < W and 0 <= row < H):
            n_out_of_bounds += 1
            continue
        c, r = int(col), int(row)
        cell_groups.setdefault((r, c), []).append((col, row, classify(v["type"], class_to_idx)))

    for (r, c), items in cell_groups.items():
        cols = [it[0] for it in items]
        rows = [it[1] for it in items]
        classes = [it[2] for it in items]
        mean_col = float(np.mean(cols))
        mean_row = float(np.mean(rows))

        cls_counts = Counter(classes)
        if len(cls_counts) > 1:
            n_class_collisions += 1
        # приоритет: если в ячейке смешались классы, берём самый частый,
        # при равенстве — приоритет foreground-классам над "others"
        priority = {name: i for i, name in enumerate(list(class_to_idx.keys()))}
        cls_name = max(cls_counts.items(), key=lambda kv: (kv[1], -priority.get(kv[0], 999)))[0]

        class_map[r, c] = class_to_idx[cls_name]
        offset_map[0, r, c] = mean_col - c
        offset_map[1, r, c] = mean_row - r
        offset_mask[r, c] = True
        class_hits[cls_name] += 1

        draw_gaussian_max(objectness, (mean_col, mean_row), radius=radius)

    if n_out_of_bounds:
        print(f"[WARN] {n_out_of_bounds} нарушений вне сетки (за границей die/GCellGrid).")
    if n_dropped_no_type:
        print(f"[WARN] {n_dropped_no_type} нарушений без 'violation type' — пропущены.")
    if n_class_collisions:
        print(
            f"[WARN] {n_class_collisions} GCell-ячеек получили нарушения разных классов "
            f"одновременно — записан самый частый класс в ячейке."
        )

    return {
        "objectness": objectness,
        "class_map": class_map,
        "offset": offset_map,
        "offset_mask": offset_mask,
        "class_hits": class_hits,
        "shape": (H, W),
        "n_raw_violations": len(violations),
        "n_merged_points": len(cell_groups),
    }


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    base_dir = Path(
        r"J:\dataset_Nikita_v1\dataset\freepdk45\trigonometric\CLK_4.6_IO_0.00_CU_20_AR_1.0_HW_4_HS_4_HP_32_VW_4_VS_4_VP_32"
    )
    def_file = base_dir / "route" / "def" / "def.def"
    drc_file = base_dir / "route" / "drc_report" / "drc_report.txt"

    if not def_file.exists() or not drc_file.exists():
        print("Ошибка путей!")
    else:
        class_names, class_to_idx = build_class_config()
        print(f"Классы (0=background): {class_to_idx}")
        targets = build_drc_targets(def_file, drc_file, class_to_idx)
        print(f"Grid shape: {targets['shape']}")
        print(f"Raw violations: {targets['n_raw_violations']}, merged points: {targets['n_merged_points']}")
        print(f"Class hits: {dict(targets['class_hits'])}")