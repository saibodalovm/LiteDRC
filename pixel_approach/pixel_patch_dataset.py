import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import random
import json
import re

pattern = re.compile(
    r"^(?P<pdk>[a-zA-Z0-9]+)_(?P<design>.+?)_CLK_(?P<CLK>[\d\.]+)_IO_(?P<IO>[\d\.]+)_"
    r"CU_(?P<CU>[\d\.]+)_AR_(?P<AR>[\d\.]+)_HW_(?P<HW>[\d\.]+)_HS_(?P<HS>[\d\.]+)_"
    r"HP_(?P<HP>[\d\.]+)_VW_(?P<VW>[\d\.]+)_VS_(?P<VS>[\d\.]+)_VP_(?P<VP>[\d\.]+)"
)

PDK_TO_IDX = {"asap7": 0, "freepdk45": 1, "gf180": 2, "sky130": 3}

class EDAChipDataset(Dataset):
    def __init__(self, data_dir, stats_path, patch_size=64, samples_per_file=5, cache_stats=True, mode='train'):
        self.files = list(Path(data_dir).glob("*.npz"))
        self.patch_size = patch_size
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
                
                # Считаем, сколько патчей нужно, чтобы полностью закрыть чип
                n_patches_h = int(np.ceil(H / self.patch_size))
                n_patches_w = int(np.ceil(W / self.patch_size))
                
                for ph in range(n_patches_h):
                    for pw in range(n_patches_w):
                        # Сохраняем (индекс файла, номер патча по Y, номер патча по X)
                        self.pdk_grid.append((file_idx, ph, pw))

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
            raise ValueError(f"Не удалось распарсить параметры из имени файла: {filename}")
    
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
        if self.mode == 'train':
            # --- ТРЕНИРОВОЧНЫЙ СЛУЧАЙНЫЙ КРОП ---
            file_idx = idx // self.samples_per_file
            file_path = self.files[file_idx]
            data = np.load(file_path)
            
            pdk_vector, pdk_id = self._parse_filename_to_vector_and_pdk(file_path.name)
            
            F = np.stack([data["rudy"], data["pin_density"], data["placement_density"]], axis=0).astype(np.float32)
            Obj = data["target_objectness"]
            Cls = data["target_class_map"]
            Off = data["target_offset"]
            Mask = data["target_offset_mask"]
            
            H, W = Obj.shape
            
            if H < self.patch_size or W < self.patch_size:
                pad_h = max(0, self.patch_size - H)
                pad_w = max(0, self.patch_size - W)
                F = np.pad(F, ((0,0), (0, pad_h), (0, pad_w)), mode='constant')
                Obj = np.pad(Obj, ((0, pad_h), (0, pad_w)), mode='constant')
                Cls = np.pad(Cls, ((0, pad_h), (0, pad_w)), mode='constant')
                Off = np.pad(Off, ((0,0), (0, pad_h), (0, pad_w)), mode='constant')
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
            # --- ДЕТЕРМИНИРОВАННЫЙ ТЕСТОВЫЙ ТАЙЛИНГ ---
            file_idx, ph, pw = self.pdk_grid[idx]
            file_path = self.files[file_idx]
            data = np.load(file_path)
            
            pdk_vector, pdk_id = self._parse_filename_to_vector_and_pdk(file_path.name)
            
            F = np.stack([data["rudy"], data["pin_density"], data["placement_density"]], axis=0).astype(np.float32)
            Obj = data["target_objectness"]
            Cls = data["target_class_map"]
            Off = data["target_offset"]
            Mask = data["target_offset_mask"]
            
            H, W = Obj.shape
            
            # Добиваем весь чип паддингом до кратности patch_size, чтобы не потерять края
            pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
            pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
            
            if pad_h > 0 or pad_w > 0:
                F = np.pad(F, ((0,0), (0, pad_h), (0, pad_w)), mode='constant')
                Obj = np.pad(Obj, ((0, pad_h), (0, pad_w)), mode='constant')
                Cls = np.pad(Cls, ((0, pad_h), (0, pad_w)), mode='constant')
                Off = np.pad(Off, ((0,0), (0, pad_h), (0, pad_w)), mode='constant')
                Mask = np.pad(Mask, ((0, pad_h), (0, pad_w)), mode='constant')
            
            # Строго вырезаем текущий неперекрывающийся патч по сетке
            y_start = ph * self.patch_size
            x_start = pw * self.patch_size

        # Кроп (одинаковый для обоих режимов, но координаты теперь либо рандомные, либо по сетке)
        F = F[:, y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]
        Obj = Obj[y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]
        Cls = Cls[y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]
        Off = Off[:, y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]
        Mask = Mask[y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]

        return {
            "features": torch.from_numpy(F.astype(np.float32)),
            "pdk_meta": torch.from_numpy(pdk_vector),
            "pdk_id": torch.tensor(pdk_id, dtype=torch.long),
            "obj": torch.from_numpy(Obj.astype(np.float32)).unsqueeze(0),
            "cls": torch.from_numpy(Cls.astype(np.int64)),
            "offset": torch.from_numpy(Off.astype(np.float32)),
            "mask": torch.from_numpy(Mask.astype(np.float32)).unsqueeze(0)
        }