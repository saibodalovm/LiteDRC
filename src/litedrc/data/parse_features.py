# parse_features.py
import numpy as np
import pandas as pd
from pathlib import Path
import bisect

REQUIRED_RUDY_COLS = {"x0", "y0", "x1", "y1", "value (%)"}
REQUIRED_PIN_COLS = {"x0", "y0", "x1", "y1", "value (pins)"}


def _validate_columns(df: pd.DataFrame, required: set, name: str):
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{name}: отсутствуют ожидаемые колонки {missing}. "
            f"Найдены: {list(df.columns)}"
        )


def preprocess_sample(run_dir: Path, atol: float = 1e-3) -> dict:
    """
    Парсит фичи для одного запуска. 
    Возвращает словарь со всеми массивами и метаданными сетки.
    """
    # Ищем файлы в корне запуска или во вложенных папках
    rudy_path = run_dir / "rudy_congestion.txt"
    pin_path = run_dir / "pin_density.txt"
    
    if not rudy_path.exists():
        rudy_path = run_dir / "route" / "congestion" / "rudy_congestion.txt"
    if not pin_path.exists():
        pin_path = run_dir / "route" / "congestion" / "pin_density.txt"

    if not rudy_path.exists() or not pin_path.exists():
        raise FileNotFoundError(
            f"Не найдены обязательные файлы фичей в {run_dir}. "
            f"RUDY: {rudy_path.exists()}, Pin Density: {pin_path.exists()}"
        )

    rudy_df = pd.read_csv(rudy_path)
    pin_df = pd.read_csv(pin_path)

    _validate_columns(rudy_df, REQUIRED_RUDY_COLS, "rudy_congestion.txt")
    _validate_columns(pin_df, REQUIRED_PIN_COLS, "pin_density.txt")

    # 1. Динамически восстанавливаем сетку GCell по уникальным координатам RUDY
    x_coords = np.array(sorted(set(rudy_df["x0"].round(4)) | set(rudy_df["x1"].round(4))))
    y_coords = np.array(sorted(set(rudy_df["y0"].round(4)) | set(rudy_df["y1"].round(4))))

    W = len(x_coords) - 1
    H = len(y_coords) - 1

    # Площадь каждой target-ячейки для нормализации
    cell_w = np.diff(x_coords)  # (W,)
    cell_h = np.diff(y_coords)  # (H,)
    cell_area = cell_h[:, None] * cell_w[None, :]  # (H, W)

    rudy_grid = np.zeros((H, W), dtype=np.float32)
    x_to_idx = {round(v, 4): i for i, v in enumerate(x_coords)}
    y_to_idx = {round(v, 4): i for i, v in enumerate(y_coords)}

    # 2. RUDY -> напрямую в target-сетку
    c_idx = rudy_df["x0"].round(4).map(x_to_idx)
    r_idx = rudy_df["y0"].round(4).map(y_to_idx)
    if c_idx.isna().any() or r_idx.isna().any():
        raise ValueError("Часть координат rudy_congestion.txt не попала в построенную сетку.")
        
    c_idx = c_idx.to_numpy(dtype=int)
    r_idx = r_idx.to_numpy(dtype=int)
    rudy_grid[r_idx, c_idx] = rudy_df["value (%)"].to_numpy()

    # Sanity check
    n_unique_cells = len(set(zip(r_idx.tolist(), c_idx.tolist())))
    if len(rudy_df) != H * W or n_unique_cells != len(rudy_df):
        print(
            f"[WARN] RUDY: строк={len(rudy_df)}, ожидалось H*W={H * W}, "
            f"уникальных ячеек={n_unique_cells}."
        )

    # 3. Растеризация пинов
    def _rasterize(bin_df: pd.DataFrame, value_col: str, mode: str) -> np.ndarray:
        assert mode in ("sum", "mean")
        bx0s = bin_df["x0"].to_numpy()
        by0s = bin_df["y0"].to_numpy()
        bx1s = bin_df["x1"].to_numpy()
        by1s = bin_df["y1"].to_numpy()
        vals = bin_df[value_col].to_numpy()

        out = np.zeros((H, W), dtype=np.float64)
        weight = np.zeros((H, W), dtype=np.float64) if mode == "mean" else None

        for bx0, by0, bx1, by1, val in zip(bx0s, by0s, bx1s, by1s, vals):
            bin_area = (bx1 - bx0) * (by1 - by0)
            if bin_area <= 0:
                continue

            c_start = max(0, bisect.bisect_right(x_coords, bx0) - 1)
            c_end = min(W, bisect.bisect_left(x_coords, bx1))
            r_start = max(0, bisect.bisect_right(y_coords, by0) - 1)
            r_end = min(H, bisect.bisect_left(y_coords, by1))

            for r in range(r_start, r_end):
                py0, py1 = y_coords[r], y_coords[r + 1]
                overlap_y = max(0.0, min(by1, py1) - max(by0, py0))
                if overlap_y == 0:
                    continue
                for c in range(c_start, c_end):
                    px0, px1 = x_coords[c], x_coords[c + 1]
                    overlap_x = max(0.0, min(bx1, px1) - max(bx0, px0))
                    if overlap_x == 0:
                        continue
                    overlap_area = overlap_x * overlap_y
                    if mode == "sum":
                        out[r, c] += val * (overlap_area / bin_area)
                    else:
                        out[r, c] += val * overlap_area
                        weight[r, c] += overlap_area

        if mode == "mean":
            covered = weight > 0
            out[covered] = out[covered] / weight[covered]

        return out

    pin_count_grid = _rasterize(pin_df, "value (pins)", mode="sum")

    # Sanity check для сохранения массы
    total_in = pin_df["value (pins)"].sum()
    total_out = pin_count_grid.sum()
    if not np.isclose(total_in, total_out, rtol=atol):
        print(
            f"[WARN] Pin density: суммы не совпадают до/после растеризации "
            f"({total_in:.3f} -> {total_out:.3f})."
        )

    pin_density_grid = (pin_count_grid / cell_area).astype(np.float32)

    # 3b. Опциональный placement density
    placement_grid = None
    placement_path = run_dir / "placement_density.txt"
    if not placement_path.exists():
        placement_path = run_dir / "route" / "congestion" / "placement_density.txt"

    if placement_path.exists():
        placement_df = pd.read_csv(placement_path)
        _validate_columns(placement_df, {"x0", "y0", "x1", "y1", "value (%)"}, "placement_density.txt")
        placement_grid = _rasterize(placement_df, "value (%)", mode="mean").astype(np.float32)

    # Возвращаем все собранные массивы и динамический размер сетки H, W
    return {
        "rudy": rudy_grid,
        "pin_density": pin_density_grid,
        "pin_count": pin_count_grid.astype(np.float32),
        "cell_area": cell_area.astype(np.float32),
        "placement_density": placement_grid,
        "grid_shape": (H, W)
    }