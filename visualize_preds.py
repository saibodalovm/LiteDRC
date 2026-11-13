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
from models.EncoderDerichle import EvidentialLightweightRegionNet
from data_processing.patch_dataset import EDAChipDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    # === НАСТРОЙКИ (Проверь пути!) ===
    TARGET_PDK = "gf180"
    
    # Укажи точный индекс файла (число), когда найдешь нужный.
    # Если None, скрипт распечатает все пути с их номерами.
    TARGET_FILE_IDX = 18
    
    TEST_DIR = r"J:\processed_dataset\val"
    STATS_PATH = r"J:\processed_dataset\train\global_zscore_stats.json"
    #WEIGHT_PATH = "ckpt/train_sky_true_DERICHLE/best_model_zeroshot_train_sky130_test_gf180_evidential_attn.pth"
    WEIGHT_PATH = "ckpt/train_gf_true_DERICHLE/best_model_zeroshot_train_gf180_test_sky130_evidential_attn.pth"
    # =================================

    print("Загрузка датасета...")
    test_ds = EDAChipDataset(data_dir=TEST_DIR, stats_path=STATS_PATH, patch_size=64, mode='test')
    
    # 1. Поиск или выбор конкретного файла
    if TARGET_FILE_IDX is None:
        print("\n📋 ПРОНУМЕРОВАННЫЙ СПИСОК ВСЕХ ЗАПУСКОВ:")
        for idx, filepath in enumerate(test_ds.files):
            if TARGET_PDK in str(filepath):
                # Печатаем родительскую папку + имя файла (чтобы отличать разные параметры OpenROAD)
                rel_path = os.path.join(os.path.basename(os.path.dirname(filepath)), os.path.basename(filepath))
                print(f"  [{idx}] {rel_path}")
        
        print("\n👉 Выбери индекс нужного run'а, впиши его в TARGET_FILE_IDX = ... и запусти снова!")
        return

    # Если индекс задан:
    if TARGET_FILE_IDX < 0 or TARGET_FILE_IDX >= len(test_ds.files):
        print("Ошибка: неверный индекс!")
        return
        
    target_filepath = test_ds.files[TARGET_FILE_IDX]
    # Формируем имя для графика (папка + файл), чтобы было понятно на картинке
    design_name = f"{os.path.basename(os.path.dirname(target_filepath))}_{os.path.basename(target_filepath).split('.')[0]}"
    
    # 2. Собираем индексы всех патчей для этого конкретного файла
    file_indices = [i for i in range(len(test_ds)) if test_ds.pdk_grid[i][0] == TARGET_FILE_IDX]
    
    if not file_indices:
        print("Нет патчей для выбранного файла!")
        return

    # 3. Вычисляем физические размеры макроса в патчах
    max_row = max(test_ds.pdk_grid[i][1] for i in file_indices)
    max_col = max(test_ds.pdk_grid[i][2] for i in file_indices)
    
    # Итоговое разрешение (каждый патч дает окно 8x8)
    full_h = (max_row + 1) * 8
    full_w = (max_col + 1) * 8
    
    print(f"\n🎯 Выбран конкретный запуск: {design_name}")
    print(f"Сетка макроса: {max_row+1} x {max_col+1} патчей")
    print(f"Разрешение итоговой карты: {full_h} x {full_w} пикселей")

    # Инициализация пустых холстов для всего чипа
    full_gt = np.zeros((full_h, full_w))
    full_pred = np.zeros((full_h, full_w))
    full_unc = np.zeros((full_h, full_w))

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
            
            # БЕРЕМ МАКСИМУМ ПО КЛАССАМ (Агрегируем Short, Spacing и Others в одну маску)
            patch_gt_max = np.max(targets, axis=0)
            patch_pred_max = np.max(probs, axis=0)
            patch_unc_max = np.max(uncertainty, axis=0)
            
            # Вклеиваем 8x8 в нужное место
            full_gt[row*8 : (row+1)*8, col*8 : (col+1)*8] = patch_gt_max
            full_pred[row*8 : (row+1)*8, col*8 : (col+1)*8] = patch_pred_max
            full_unc[row*8 : (row+1)*8, col*8 : (col+1)*8] = patch_unc_max

    # === ОТРИСОВКА И СОХРАНЕНИЕ ===
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt

    # === ОБРЕЗКА ===
    crop_x = 2
    full_gt = full_gt[:, crop_x:]
    full_pred = full_pred[:, crop_x:]
    full_unc = full_unc[:, crop_x:]

    # Физический размер под ширину \textwidth в elsarticle (~7.2 дюйма)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), dpi=300)

    title_fontsize = 9.5
    cbar_tick_fontsize = 8

    # 1. Ground Truth
    im0 = axes[0].imshow(full_gt, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Ground Truth (All DRCs)", fontsize=title_fontsize, pad=6)
    axes[0].axis("off")

    # Добавляем невидимый разделитель для axes[0],
    # чтобы его ширина была 1 в 1 как у axes[1] и axes[2] с колорбарами
    div0 = make_axes_locatable(axes[0])
    cax0 = div0.append_axes("right", size="5%", pad=0.06)
    cax0.axis("off")

    # 2. Predicted Probability
    im1 = axes[1].imshow(full_pred, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Predicted Probability Map", fontsize=title_fontsize, pad=6)
    axes[1].axis("off")

    div1 = make_axes_locatable(axes[1])
    cax1 = div1.append_axes("right", size="5%", pad=0.06)
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.ax.tick_params(labelsize=cbar_tick_fontsize, length=2, pad=2)

    # 3. Uncertainty
    im2 = axes[2].imshow(full_unc, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title(
        "Evidential Uncertainty ($u$)", fontsize=title_fontsize, pad=6
    )
    axes[2].axis("off")

    div2 = make_axes_locatable(axes[2])
    cax2 = div2.append_axes("right", size="5%", pad=0.06)
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.ax.tick_params(labelsize=cbar_tick_fontsize, length=2, pad=2)

    plt.subplots_adjust(left=0.01, right=0.96, top=0.88, bottom=0.05, wspace=0.25)

    save_path = f"full_chip_{design_name}.pdf"  # PDF предпочтительнее для статей
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"✅ Готово! Макро-карта сохранена как: {save_path}")

if __name__ == "__main__":
    main()