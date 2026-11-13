import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

# Подключаем корень проекта
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from models.EncoderCrossAttDerichle import EvidentialLightweightRegionNet as EvidentialLightweightRegionNetAttn 
from data_processing.patch_dataset import EDAChipDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    # === НАСТРОЙКИ ===
    TARGET_PDK = "sky130"
    
    # Укажи точный индекс файла (число), который выдал красивую картинку до этого.
    # Если None, скрипт распечатает все пути с их номерами.
    TARGET_FILE_IDX = 154 
    
    TEST_DIR = r"J:\processed_dataset\val"
    STATS_PATH = r"J:\processed_dataset\train\global_zscore_stats.json"
    WEIGHT_PATH = "ckpt/train_sky_true_DERICHLE/best_model_zeroshot_train_sky130_test_gf180_evidential_attn.pth"
    
    SAVE_DIR = "visualizations"
    os.makedirs(SAVE_DIR, exist_ok=True)
    # =================================

    print("Загрузка датасета...")
    test_ds = EDAChipDataset(data_dir=TEST_DIR, stats_path=STATS_PATH, patch_size=64, mode='test')
    
    if TARGET_FILE_IDX is None:
        print("\n📋 ПРОНУМЕРОВАННЫЙ СПИСОК ВСЕХ ЗАПУСКОВ:")
        for idx, filepath in enumerate(test_ds.files):
            if TARGET_PDK in str(filepath):
                rel_path = os.path.join(os.path.basename(os.path.dirname(filepath)), os.path.basename(filepath))
                print(f"  [{idx}] {rel_path}")
        print("\n👉 Выбери индекс нужного run'а, впиши его в TARGET_FILE_IDX = ... и запусти снова!")
        return

    if TARGET_FILE_IDX < 0 or TARGET_FILE_IDX >= len(test_ds.files):
        print("Ошибка: неверный индекс!")
        return
        
    target_filepath = test_ds.files[TARGET_FILE_IDX]
    design_name = f"{os.path.basename(os.path.dirname(target_filepath))}_{os.path.basename(target_filepath).split('.')[0]}"
    
    file_indices = [i for i in range(len(test_ds)) if test_ds.pdk_grid[i][0] == TARGET_FILE_IDX]
    
    if not file_indices:
        print("Нет патчей для выбранного файла!")
        return

    max_row = max(test_ds.pdk_grid[i][1] for i in file_indices)
    max_col = max(test_ds.pdk_grid[i][2] for i in file_indices)
    
    full_h = (max_row + 1) * 8
    full_w = (max_col + 1) * 8
    
    print(f"\n🎯 Выбран конкретный запуск: {design_name}")
    print(f"Разрешение итоговой карты: {full_h} x {full_w} пикселей")

    full_gt = np.zeros((3, full_h, full_w))
    full_pred = np.zeros((3, full_h, full_w))
    full_unc = np.zeros((3, full_h, full_w))

    print("Инициализация модели CrossAttention...")
    model = EvidentialLightweightRegionNetAttn(in_channels=3, num_classes=3, use_film=True).to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    model.eval()

    print("Инференс и склейка карты...")
    with torch.no_grad():
        for idx in file_indices:
            _, row, col = test_ds.pdk_grid[idx]
            
            sample = test_ds[idx]
            features = sample["features"].unsqueeze(0).to(device)
            pdk_meta = sample["pdk_meta"].unsqueeze(0).to(device)
            pdk_id = sample["pdk_id"].unsqueeze(0).to(device)
            targets = sample["target_regions"].numpy() 
            
            preds = model(features, pdk_meta, pdk_id)
            probs = preds["probs"].cpu().numpy()[0]          
            uncertainty = preds["uncertainty"].cpu().numpy()[0] 
            
            full_gt[:, row*8 : (row+1)*8, col*8 : (col+1)*8] = targets
            full_pred[:, row*8 : (row+1)*8, col*8 : (col+1)*8] = probs
            full_unc[:, row*8 : (row+1)*8, col*8 : (col+1)*8] = uncertainty

    # Отрезаем ту самую левую полоску граничных ячеек (16 пикселей)
    crop_x = 2
    full_gt = full_gt[:, :, crop_x:]
    full_pred = full_pred[:, :, crop_x:]
    full_unc = full_unc[:, :, crop_x:]

    # === ОТРИСОВКА И СОХРАНЕНИЕ 3x3 ===
    class_names = ['Short', 'Metal Spacing', 'Others']
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))
    fig.suptitle(f'Per-Class DRC Hotspots: {design_name}', fontsize=20, y=0.95)

    for c in range(3):
        # 1. Ground Truth (без интерполяции)
        im_gt = axes[c, 0].imshow(full_gt[c], cmap='gray', vmin=0, vmax=1)
        axes[c, 0].set_title(f'GT: {class_names[c]}', fontsize=14)
        axes[c, 0].axis('off')

        # 2. Predicted (без интерполяции)
        im_pred = axes[c, 1].imshow(full_pred[c], cmap='jet', vmin=0, vmax=1)
        axes[c, 1].set_title(f'Predicted: {class_names[c]}', fontsize=14)
        axes[c, 1].axis('off')

        # 3. Uncertainty (без интерполяции)
        im_unc = axes[c, 2].imshow(full_unc[c], cmap='magma', vmin=0, vmax=1)
        axes[c, 2].set_title(f'Uncertainty (u): {class_names[c]}', fontsize=14)
        axes[c, 2].axis('off')

    # Шкалы
    cbar_ax1 = fig.add_axes([0.92, 0.4, 0.02, 0.2])
    fig.colorbar(im_pred, cax=cbar_ax1, orientation='vertical', label='Probability')
    
    cbar_ax2 = fig.add_axes([0.92, 0.1, 0.02, 0.2])
    fig.colorbar(im_unc, cax=cbar_ax2, orientation='vertical', label='Uncertainty')

    plt.subplots_adjust(wspace=0.1, hspace=0.2, right=0.9)
    save_path = os.path.join(SAVE_DIR, f"full_chip_per_class_{design_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Готово! Карта по классам сохранена в папку: {save_path}")

if __name__ == "__main__":
    main()