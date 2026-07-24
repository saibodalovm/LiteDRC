import argparse
from collections import defaultdict, Counter
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from layer_utils import normalize_layer_name, report_unmapped_layers


def parse_drc_file(file_path: Path) -> list[tuple[str, str, str]]:
    """
    Читает файл отчета DRC и вытаскивает тройки
    (violation_type, raw_layer, normalized_layer) для каждого нарушения.
    raw_layer сохраняем отдельно, чтобы потом можно было глазами проверить
    качество нормализации через report_unmapped_layers().
    """
    violations = []
    type_pattern = re.compile(r"violation type:\s*([\w\s]+)")
    layer_pattern = re.compile(r"on Layer\s+(\S+)")

    current_type = None
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_type = type_pattern.search(line)
            if m_type:
                current_type = m_type.group(1).strip()
                continue
            m_layer = layer_pattern.search(line)
            if m_layer and current_type is not None:
                raw_layer = m_layer.group(1)
                violations.append((current_type, raw_layer, normalize_layer_name(raw_layer)))

    return violations


def main():
    parser = argparse.ArgumentParser(description="Сбор статистики по DRC нарушениям во всем датасете.")
    parser.add_argument("--dataset_path", required=True, type=Path, help="Путь к корню датасета Никиты.")
    args = parser.parse_args()

    drc_files = sorted(args.dataset_path.rglob("drc_report.txt"))
    print(f"Найдено файлов отчетов DRC: {len(drc_files)}")

    global_stats = defaultdict(int)          # type -> count
    layer_stats = defaultdict(int)           # normalized layer -> count
    raw_layer_stats = Counter()              # raw layer string -> count (для проверки нормализации)
    type_layer_stats = defaultdict(int)      # (type, normalized layer) -> count
    design_to_violations = {}

    for drc_file in drc_files:
        sample_id = drc_file.relative_to(args.dataset_path).parent
        violations = parse_drc_file(drc_file)

        for v_type, raw_layer, norm_layer in violations:
            global_stats[v_type] += 1
            layer_stats[norm_layer] += 1
            raw_layer_stats[raw_layer] += 1
            type_layer_stats[(v_type, norm_layer)] += 1

        design_to_violations[str(sample_id)] = len(violations)

    if not global_stats:
        print("Нарушений не найдено. Проверь имя файлов drc_report.txt в rglob().")
        return

    # Обязательно проверить глазами перед тем, как доверять layer_stats/heatmap:
    # разные PDK могут использовать конвенции, которые regex не покрывает.
    report_unmapped_layers(raw_layer_stats)

    # --- Распределение по типам (как было) ---
    df_type = pd.DataFrame(list(global_stats.items()), columns=["Violation Type", "Count"])
    df_type = df_type.sort_values(by="Count", ascending=False).reset_index(drop=True)

    print("\n=== РАСПРЕДЕЛЕНИЕ КЛАССОВ НАРУШЕНИЙ ===")
    print(df_type.to_string())

    # --- Распределение по слоям металлизации ---
    df_layer = pd.DataFrame(list(layer_stats.items()), columns=["Layer", "Count"])
    df_layer = df_layer.sort_values(by="Count", ascending=False).reset_index(drop=True)

    print("\n=== РАСПРЕДЕЛЕНИЕ ПО СЛОЯМ МЕТАЛЛИЗАЦИИ ===")
    print(df_layer.to_string())

    total_drc = sum(global_stats.values())
    print(f"\nВсего DRC нарушений в датасете: {total_drc}")
    print(f"Среднее количество DRC на один чип: {total_drc / len(drc_files):.2f}")

    # NB: это СЫРЫЕ счётчики строк из репорта. Один физический дефект часто
    # репортится DRC-чекером несколько раз с чуть разными bbox (мы уже
    # находили это на примере drc_report.txt — 40 строк сворачивались в 18
    # физических точек). Для распределения по классам/слоям это, как правило,
    # не критично (пропорции сохраняются), но абсолютные числа завышены —
    # не стоит использовать total_drc напрямую как "число дефектов на чип".

    # --- Гистограмма по типам (log scale), как было ---
    plt.figure(figsize=(10, 6))
    plt.bar(df_type["Violation Type"], df_type["Count"], color="royalblue", edgecolor="black")
    plt.yscale("log")
    plt.title("Распределение DRC нарушений во всем датасете (Log Scale)")
    plt.xlabel("Тип нарушения")
    plt.ylabel("Количество (Log count)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("drc_distribution.png")
    print("\nГрафик распределения по типам сохранен в: drc_distribution_.png")

    # --- Гистограмма по слоям (log scale) ---
    plt.figure(figsize=(10, 6))
    plt.bar(df_layer["Layer"], df_layer["Count"], color="seagreen", edgecolor="black")
    plt.yscale("log")
    plt.title("Распределение DRC нарушений по слоям металлизации (Log Scale)")
    plt.xlabel("Слой")
    plt.ylabel("Количество (Log count)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("drc_layer_distribution.png")
    print("График распределения по слоям сохранен в: drc_layer_distribution.png")

    # --- Heatmap type x layer — основная "картинка для статьи" ---
    top_types = df_type["Violation Type"].tolist()   # все типы (обычно их немного, 10-15)
    top_layers = df_layer["Layer"].tolist()          # все слои (обычно тоже немного)

    matrix = np.zeros((len(top_types), len(top_layers)))
    for i, t in enumerate(top_types):
        for j, l in enumerate(top_layers):
            matrix[i, j] = type_layer_stats.get((t, l), 0)

    # log1p, чтобы редкие ячейки тоже были видны на фоне доминирующих классов
    log_matrix = np.log1p(matrix)

    fig, ax = plt.subplots(figsize=(max(6, len(top_layers) * 0.9), max(4, len(top_types) * 0.5)))
    im = ax.imshow(log_matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(top_layers)))
    ax.set_xticklabels(top_layers, rotation=45, ha="right")
    ax.set_yticks(range(len(top_types)))
    ax.set_yticklabels(top_types)
    ax.set_title("DRC нарушения: тип × слой (log1p(count))")

    for i in range(len(top_types)):
        for j in range(len(top_layers)):
            count = int(matrix[i, j])
            if count > 0:
                ax.text(j, i, str(count), ha="center", va="center",
                        color="white" if log_matrix[i, j] > log_matrix.max() * 0.5 else "black",
                        fontsize=7)

    fig.colorbar(im, ax=ax, label="log1p(count)")
    plt.tight_layout()
    plt.savefig("drc_type_by_layer_heatmap.png", dpi=150)
    print("Heatmap тип×слой сохранен в: drc_type_by_layer_heatmap.png")


if __name__ == "__main__":
    main()