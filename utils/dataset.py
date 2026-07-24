import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

class DRCDataset(Dataset):
    def __init__(self, data_dir: Path, overfit_sample_name: str = None):
        self.data_dir = Path(data_dir)
        
        if overfit_sample_name:
            # Насильно заставляем датасет видеть только ОДИН конкретный .npz файл
            target_file = self.data_dir / f"{overfit_sample_name}.npz"
            if not target_file.exists():
                raise FileNotFoundError(f"Сэмпл для оверфита не найден: {target_file}")
            # Дублируем его, например, 100 раз, чтобы эпоха не состояла из 1 шага
            self.filepaths = [target_file] * 128
            print(f"[DEBUG] Режим оверфита! Загружен только сэмпл: {overfit_sample_name} (размер эпохи: {len(self.filepaths)})")
        else:
            self.filepaths = sorted(list(self.data_dir.glob("*.npz")))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        data = np.load(self.filepaths[idx])
        
        # 1. Собираем фичи в один стек каналов (С, H, W)
        # Входные фичи: rudy, pin_density, pin_count, cell_area, cell_density
        # (Все они должны иметь одинаковую размерность H x W)
        features = np.stack([
            data["rudy"],
            data["pin_density"],
            data["placement_density"],   
        ], axis=0).astype(np.float32)
        
        # Если у тебя есть placement_density, можно добавить и его:
        # if "placement_density" in data: ...  


        # 2. Извлекаем таргеты
        target_objectness = data["target_objectness"].astype(np.float32)  # (H, W)
        target_class_map = data["target_class_map"].astype(np.int64)     # (H, W)
        target_offset = data["target_offset"].astype(np.float32)         # (2, H, W)
        target_offset_mask = data["target_offset_mask"].astype(np.float32) # (H, W)
        
        return {
            "features": torch.from_numpy(features),
            "target_objectness": torch.from_numpy(target_objectness).unsqueeze(0), # (1, H, W) для Focal Loss
            "target_class_map": torch.from_numpy(target_class_map),               # (H, W)
            "target_offset": torch.from_numpy(target_offset),                     # (2, H, W)
            "target_offset_mask": torch.from_numpy(target_offset_mask)            # (H, W)
        }
    
if __name__ == "__main__":
    path_to_data = r"J:\processed_dataset\train"
    dataset = DRCDataset(path_to_data)
    print(dataset[0].shape)