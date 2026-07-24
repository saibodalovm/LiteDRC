import matplotlib.pyplot as plt
import numpy as np

rudy_grid = np.load(r'J:\dataset_Nikita_v1\dataset_features_prects\asap7\wbqspiflash\CLK_800_IO_0.00_CU_20_AR_0.5_HW_4.5_HS_4.5_HP_32_VW_4.5_VS_4.5_VP_32\features\rudy_congestion.npy')
pin_density_grid = np.load(r'J:\dataset_Nikita_v1\dataset_features_prects\asap7\wbqspiflash\CLK_800_IO_0.00_CU_20_AR_0.5_HW_4.5_HS_4.5_HP_32_VW_4.5_VS_4.5_VP_32\features\pin_density.npy')
placement_density_grid = np.load(r'J:\dataset_Nikita_v1\dataset_features_prects\asap7\wbqspiflash\CLK_800_IO_0.00_CU_20_AR_0.5_HW_4.5_HS_4.5_HP_32_VW_4.5_VS_4.5_VP_32\features\placement_density.npy')
# Визуализация фичей одного семпла
fig, axes = plt.subplots(1, 3, figsize=(12, 7))

im0 = axes[0].imshow(rudy_grid, origin='lower', cmap='hot')
axes[0].set_title("RUDY Congestion Map (91x47)")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(pin_density_grid, origin='lower', cmap='viridis')
axes[1].set_title("Rasterized Pin Density Map (91x47)")
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(placement_density_grid, origin='lower', cmap='viridis')
axes[2].set_title("Rasterized Placement Density Map (91x47)")
fig.colorbar(im1, ax=axes[2])

plt.tight_layout()
plt.show()