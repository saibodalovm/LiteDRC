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
                dbu_per_micron = float(line.split()[3])
                
            elif line.startswith("GCELLGRID X"):
                parts = line.split()
                grid_shape[1] = int(parts[4])  # Ширина
                step_x = float(parts[6])       # Шаг в DBU
                
            elif line.startswith("GCELLGRID Y"):
                parts = line.split()
                grid_shape[0] = int(parts[4])  # Высота
                
            if line.startswith("COMPONENTS") or line.startswith("SPECIALNETS"):
                break
                
    gcell_size_um = step_x / dbu_per_micron if dbu_per_micron else 0.0
    return dbu_per_micron, tuple(grid_shape), gcell_size_um

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
