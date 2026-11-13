import torch
import torch.nn.functional as F_torch  # Добавили для one_hot и max_pool2d
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import random
import json
import re
import os
from utils.build_track_distance_map import (
    get_def_metadata, extract_power_grids, 
    rasterize_power_shapes, build_pg_distance_channels
)

pattern = re.compile(
    r"^(?P<pdk>[a-zA-Z0-9]+)_(?P<design>.+?)_CLK_(?P<CLK>[\d\.]+)_IO_(?P<IO>[\d\.]+)_"
    r"CU_(?P<CU>[\d\.]+)_AR_(?P<AR>[\d\.]+)_HW_(?P<HW>[\d\.]+)_HS_(?P<HS>[\d\.]+)_"
    r"HP_(?P<HP>[\d\.]+)_VW_(?P<VW>[\d\.]+)_VS_(?P<VS>[\d\.]+)_VP_(?P<VP>[\d\.]+)"
)

PDK_TO_IDX = {"asap7": 0, "freepdk45": 1, "gf180": 2, "sky130": 3}

class EDAChipDataset(Dataset):
    def __init__(self, data_dir, stats_path, patch_size=64, window_size=8, num_classes=4, samples_per_file=5, cache_stats=True, mode='train', 
                                                                                    use_distances=False, def_base_dir="J:/dataset_Nikita_v1/dataset"):
        self.files = list(Path(data_dir).glob("*.npz"))
        self.patch_size = patch_size
        self.window_size = window_size  # Размер макро-окна (по умолчанию 8)
        self.num_classes = num_classes  # 4 класса: 0(Фон), 1(Short), 2(Spacing), 3(MinArea)
        self.samples_per_file = samples_per_file
        self.mode = mode
        
        # Загружаем глобальную Z-score статистику
        with open(stats_path, "r") as f:
            self.meta_stats = json.load(f)
            
        # Для трейна считаем веса сэмплинга
        self.file_n_pos = self._load_or_compute_stats(data_dir, cache_stats)
        
        # ЕСЛИ РЕЖИМ ТЕСТА: Строим детерминированную сетку патчей по всем файлам
        if self.mode == 'test':
            self.pdk_grid = []
            for file_idx, f in enumerate(self.files):
                data = np.load(f)
                H, W = data["target_objectness"].shape
                
                n_patches_h = int(np.ceil(H / self.patch_size))
                n_patches_w = int(np.ceil(W / self.patch_size))
                
                for ph in range(n_patches_h):
                    for pw in range(n_patches_w):
                        self.pdk_grid.append((file_idx, ph, pw))

        self.use_distances = use_distances
        
        # Директория для кэшей (создадим автоматически)
        self.cache_dir = os.path.join(data_dir, "..", "distance_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Базовая папка с сырыми данными (откуда брать .def, если кэша нет)
        self.def_base_dir = def_base_dir

    def _get_global_distance_map(self, original_def_path, chip_name):
        """
        Лениво ищет/создает кэш карты дистанций для всего чипа (без кропа).
        """
        cache_path = os.path.join(self.cache_dir, f"{chip_name}_distances.npy")
        
        if not os.path.exists(cache_path):
            print(f"\n[КЭШ ПУСТ] Генерируем Distance Maps для: {chip_name}...")
            
            dbu, grid_shape, gcell_um = get_def_metadata(original_def_path)
            shapes = extract_power_grids(original_def_path, dbu_per_micron=dbu)
            
            ignore_list = ['Metal1', 'Metal2', 'met1', 'met2', 'li1']
            vdd_masks, gnd_masks = rasterize_power_shapes(shapes, grid_shape, gcell_um, ignore_list)
            vdd_dist, gnd_dist = build_pg_distance_channels(vdd_masks, gnd_masks, max_dist_gcells=2.0)
            
            # Собираем в один тензор [2, H, W]
            dist_tensor = np.stack([vdd_dist, gnd_dist], axis=0).astype(np.float32)
            
            # Сохраняем на диск
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, dist_tensor)
            print(f"[УСПЕХ] Кэш сохранен: {cache_path}")

        # Загружаем не весь файл в RAM, а через mmap
        return np.load(cache_path, mmap_mode='r')
    
    def _load_or_compute_stats(self, data_dir, cache_stats):
        cache_path = Path(data_dir) / "_n_pos_cache.json"
        cached = {}
        if cache_stats and cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)

        n_pos_list = []
        updated = False
        for f in self.files:
            key = f.name
            if key in cached:
                n_pos_list.append(cached[key])
                continue
            d = np.load(f)
            # Считаем количество реальных дефектов для весов файлов
            n_pos = int((d["target_offset_mask"] > 0).sum())
            n_pos_list.append(n_pos)
            cached[key] = n_pos
            updated = True

        if cache_stats and updated:
            with open(cache_path, "w") as f:
                json.dump(cached, f)

        return n_pos_list

    def sample_weights(self, power: float = 0.5, floor: float = 1.0):
        if self.mode != 'train':
            return None
        weights = []
        for n_pos in self.file_n_pos:
            w = (n_pos + floor) ** power
            weights.extend([w] * self.samples_per_file)
        return torch.tensor(weights, dtype=torch.float)
    
    def _parse_filename_to_vector_and_pdk(self, filename: str):
        match = pattern.search(filename)
        if not match:
            raise ValueError(f"Не удалось распарсить параметры: {filename}")
    
        d = match.groupdict()
        pdk_name = d["pdk"]
        if pdk_name not in PDK_TO_IDX:
            raise ValueError(f"Неизвестный PDK '{pdk_name}' в файле {filename}")
        pdk_id = PDK_TO_IDX[pdk_name]
    
        clk_raw = float(d["CLK"])
        if pdk_name == "asap7":
            clk_raw /= 1000.0
        clk_val = np.log1p(clk_raw)
    
        vector = []
        mean = self.meta_stats["z_score"]["CLK"]["mean"]
        std = self.meta_stats["z_score"]["CLK"]["std"]
        vector.append((clk_val - mean) / std)
    
        keys = ["CU", "AR", "HW", "HS", "HP", "VW", "VS", "VP"]
        for k in keys:
            val = float(d[k])
            mean = self.meta_stats["z_score"][k]["mean"]
            std = self.meta_stats["z_score"][k]["std"]
            vector.append((val - mean) / std)
    
        return np.array(vector, dtype=np.float32), pdk_id

    def __len__(self):
        if self.mode == 'train':
            return len(self.files) * self.samples_per_file
        else:
            return len(self.pdk_grid)

    def __getitem__(self, idx):
        pad_h, pad_w = 0, 0  

        if self.mode == 'train':
            file_idx = idx // self.samples_per_file
            file_path = self.files[file_idx]
            data = np.load(file_path)
            
            pdk_vector, pdk_id = self._parse_filename_to_vector_and_pdk(file_path.name)
            
            F = np.stack([data["rudy"], data["pin_density"], data["placement_density"]], axis=0).astype(np.float32)
            Obj = data["target_objectness"]
            Cls = data["target_class_map"]
            Mask = data["target_offset_mask"]
            
            H, W = Obj.shape
            
            if H < self.patch_size or W < self.patch_size:
                pad_h = max(0, self.patch_size - H)
                pad_w = max(0, self.patch_size - W)
                F = np.pad(F, ((0,0), (0, pad_h), (0, pad_w)), mode='constant')
                Obj = np.pad(Obj, ((0, pad_h), (0, pad_w)), mode='constant')
                Cls = np.pad(Cls, ((0, pad_h), (0, pad_w)), mode='constant')
                Mask = np.pad(Mask, ((0, pad_h), (0, pad_w)), mode='constant')
                H, W = Obj.shape
                
            pos_indices = np.argwhere(Obj > 0)
            if len(pos_indices) > 0 and random.random() < 0.7:
                y, x = pos_indices[np.random.randint(len(pos_indices))]
            else:
                y, x = np.random.randint(0, H), np.random.randint(0, W)
                
            y_start = max(0, min(y - self.patch_size // 2, H - self.patch_size))
            x_start = max(0, min(x - self.patch_size // 2, W - self.patch_size))
            
        else:
            file_idx, ph, pw = self.pdk_grid[idx]
            file_path = self.files[file_idx]
            data = np.load(file_path)
            
            pdk_vector, pdk_id = self._parse_filename_to_vector_and_pdk(file_path.name)
            
            F = np.stack([data["rudy"], data["pin_density"], data["placement_density"]], axis=0).astype(np.float32)
            Cls = data["target_class_map"]
            Mask = data["target_offset_mask"]
            
            H, W = Cls.shape
            
            pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
            pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
            
            if pad_h > 0 or pad_w > 0:
                F = np.pad(F, ((0,0), (0, pad_h), (0, pad_w)), mode='constant')
                Cls = np.pad(Cls, ((0, pad_h), (0, pad_w)), mode='constant')
                Mask = np.pad(Mask, ((0, pad_h), (0, pad_w)), mode='constant')
            
            y_start = ph * self.patch_size
            x_start = pw * self.patch_size

        F_crop = F[:, y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]
        Cls_crop = Cls[y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]
        Mask_crop = Mask[y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]

        features_tensor = torch.from_numpy(F_crop.astype(np.float32))

        if getattr(self, 'use_distances', False):
            # 1. Парсим имя файла через твой регулярный паттерн
            match = pattern.search(file_path.name)
            if not match:
                raise ValueError(f"Не удалось распарсить имя файла: {file_path.name}")
            
            pdk_name = match.group("pdk")
            design_name = match.group("design")
            chip_name = file_path.name.replace('.npz', '').replace('.npy', '')
            
            # --- 🔥 ИСПРАВЛЕНИЕ ЗДЕСЬ 🔥 ---
            # Откусываем "pdk_design_" от имени файла, чтобы получить точное имя папки
            prefix = f"{pdk_name}_{design_name}_"
            folder_name = chip_name.replace(prefix, "")
            
            # 2. Формируем путь к def.def, используя folder_name
            original_def_path = os.path.join(
                self.def_base_dir, pdk_name, design_name, folder_name, "prects", "def", "def.def"
            )
            # -------------------------------
            
            # 3. Ленивая загрузка (а вот в кэш передаем полное chip_name, чтобы имена не дублировались)
            D_global = self._get_global_distance_map(original_def_path, chip_name)
            
            # 4. Тот же паддинг, что и для F, но с заполнением 1.0
            if pad_h > 0 or pad_w > 0:
                D_global = np.pad(D_global, ((0,0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=1.0) 
            
            # 5. Кроп по тем же координатам
            D_crop = D_global[:, y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]
            
            # 6. Конкатенация!
            dist_tensor = torch.from_numpy(D_crop.astype(np.float32))
            features_tensor = torch.cat([features_tensor, dist_tensor], dim=0)

        cls_tensor = torch.from_numpy(Cls_crop.astype(np.int64))
        mask_tensor = torch.from_numpy(Mask_crop.astype(np.float32))
        
        one_hot = F_torch.one_hot(cls_tensor, num_classes=self.num_classes).permute(2, 0, 1).float()
        
        target_raw = one_hot * mask_tensor

        target_defects = target_raw[1:, :, :]
        
        target_regions = F_torch.max_pool2d(
            target_defects.unsqueeze(0), 
            kernel_size=self.window_size, 
            stride=self.window_size
        ).squeeze(0)

        return {
            "features": features_tensor,
            "pdk_meta": torch.from_numpy(pdk_vector),
            "pdk_id": torch.tensor(pdk_id, dtype=torch.long),
            "target_regions": target_regions
        }