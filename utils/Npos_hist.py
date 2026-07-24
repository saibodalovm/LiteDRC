import numpy as np
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=Path)
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("*.npz"))
    n_pos_list = []
    for f in files:
        d = np.load(f)
        n_pos = int((d["target_offset_mask"] > 0).sum())
        n_pos_list.append(n_pos)

    n_pos_arr = np.array(n_pos_list)
    print(f"Всего файлов: {len(files)}")
    print(f"Файлов с n_pos == 0: {(n_pos_arr == 0).sum()} ({100*(n_pos_arr==0).mean():.1f}%)")
    print(f"Файлов с n_pos <= 2: {(n_pos_arr <= 2).sum()} ({100*(n_pos_arr<=2).mean():.1f}%)")
    print(f"Суммарно n_pos по всему train: {n_pos_arr.sum()}")

    # На сколько процентов позитивного сигнала приходится топ-10% "богатых" файлов
    sorted_desc = np.sort(n_pos_arr)[::-1]
    top10pct = max(1, len(sorted_desc) // 10)
    share_top10 = sorted_desc[:top10pct].sum() / (sorted_desc.sum() + 1e-9)
    print(f"Доля позитивного сигнала в топ-10% файлов: {100*share_top10:.1f}%")

    print("\nПерцентили n_pos:", {p: int(np.percentile(n_pos_arr, p)) for p in [50, 75, 90, 95, 99]})


if __name__ == "__main__":
    main()