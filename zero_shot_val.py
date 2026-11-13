import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve
from torch.utils.data import DataLoader, Subset
import sys
import os
import matplotlib.pyplot as plt

# Получаем абсолютный путь к папке на уровень выше (корень проекта)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Добавляем его в пути Питона
sys.path.append(PROJECT_ROOT)

# Импортируем архитектуры
from models.EncoderDerichle import EvidentialLightweightRegionNet
from models.EncoderCrossAttDerichle import EvidentialLightweightRegionNet as EvidentialLightweightRegionNetAttn 
from data_processing.patch_dataset import EDAChipDataset
import random


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_rejection_curve_data(preds_flat, targets_flat, uncert_flat, best_threshold):
    """
    Только считает точки Precision для разных Rejection Rates (от 0 до 30%).
    Ничего не рисует.
    """
    preds_binary = preds_flat > best_threshold
    positives_mask = preds_binary == 1
    
    if positives_mask.sum() == 0:
        return None, None
        
    positives_uncert = uncert_flat[positives_mask]
    positives_targets = targets_flat[positives_mask]
    
    rejection_rates = [0, 5, 10, 15, 20, 25, 30]
    precisions = []
    
    for rate in rejection_rates:
        if rate == 0:
            prec = positives_targets.mean()
        else:
            uncert_threshold = np.percentile(positives_uncert, 100 - rate)
            confident_mask = positives_uncert <= uncert_threshold
            confident_targets = positives_targets[confident_mask]
            
            prec = confident_targets.mean() if len(confident_targets) > 0 else 0.0
            
        precisions.append(prec * 100)
        
    return rejection_rates, precisions

def plot_combined_rejection_curves(curves_dict, rejection_rates, save_dir="plots"):
    """
    Рисует единый красивый график для всех переданных моделей.
    curves_dict: словарь вида {'No_FiLM': [56.2, 57.8, ...], 'CrossAttention': [...]}
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    # Набор стилей для разных линий (стандартные цвета IEEE)
    styles = [
        {'color': '#1f77b4', 'marker': 'o'}, # Синий
        {'color': '#ff7f0e', 'marker': 'o'}, # Оранжевый (квадраты)
        {'color': '#2ca02c', 'marker': 'o'}  # Зеленый (треугольники)
    ]
    
    for i, (model_name, precisions) in enumerate(curves_dict.items()):
        style = styles[i % len(styles)]
        plt.plot(rejection_rates, precisions, marker=style['marker'], 
                 linewidth=2.5, markersize=8, color=style['color'], label=model_name)
                 
    # Оформление графика
    #plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Rejection Rate (%)', fontsize=12)
    plt.ylabel('Precision (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rejection_rates)
    
    # Добавляем легенду в правый нижний угол (обычно там больше всего свободного места)
    plt.legend(fontsize=11, loc='lower right')
    
    # Убираем лишние рамки
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    filename = "combined_curve_skyderichlet_class1_2.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[УСПЕХ] Объединенный график сохранен: {save_path}")


def plot_rejection_curve(preds_flat, targets_flat, uncert_flat, best_threshold, model_name, save_dir="plots"):
    """
    Строит график Precision-Rejection Curve для оценки качества фильтрации по неопределенности.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Берем только позитивные срабатывания (где модель говорит "ТУТ ДЕФЕКТ")
    preds_binary = preds_flat > best_threshold
    positives_mask = preds_binary == 1
    
    if positives_mask.sum() == 0:
        print(f"[{model_name}] Нет позитивных предсказаний для графика.")
        return
        
    positives_uncert = uncert_flat[positives_mask]
    positives_targets = targets_flat[positives_mask]
    
    # Шаги отбрасывания: от 0% до 30% с шагом 5%
    rejection_rates = [0, 5, 10, 15, 20, 25, 30]
    precisions = []
    
    for rate in rejection_rates:
        if rate == 0:
            # Базовый Precision без фильтрации
            prec = positives_targets.mean()
        else:
            # Находим порог неопределенности для текущего процента отторжения
            uncert_threshold = np.percentile(positives_uncert, 100 - rate)
            confident_mask = positives_uncert <= uncert_threshold
            confident_targets = positives_targets[confident_mask]
            
            prec = confident_targets.mean() if len(confident_targets) > 0 else 0.0
            
        precisions.append(prec * 100) # Переводим в проценты для красоты
        
    # --- Рисуем график ---
    plt.figure(figsize=(8, 6))
    plt.plot(rejection_rates, precisions, marker='o', linewidth=2, markersize=8, color='#1f77b4')
    
    # Оформление по стандартам статей (IEEE style)
    plt.title(f'Uncertainty-Aware Rejection Curve ({model_name})', fontsize=14, pad=15)
    plt.xlabel('Rejection Rate (%)', fontsize=12)
    plt.ylabel('Confident Precision (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rejection_rates)
    
    # Добавляем аннотации значений над точками
    for i, txt in enumerate(precisions):
        plt.annotate(f"{txt:.1f}", (rejection_rates[i], precisions[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
                     
    # Убираем лишние рамки
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    save_path = os.path.join(save_dir, f"rejection_curve_{model_name}_gf_sky.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [График сохранен] {save_path}")

def evaluate_model(model, dataloader, device, model_name='default'):
    model.eval()
    all_preds = []
    all_targets = []
    all_uncertainties = [] # Добавили сбор неопределенности
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            pdk_meta = batch["pdk_meta"].to(device)
            pdk_id = batch["pdk_id"].to(device)
            targets = batch["target_regions"].to(device)
            
            # --- НОВЫЙ FORWARD ---
            # Модель теперь возвращает словарь с готовыми probs и uncertainty
            preds = model(features, pdk_meta, pdk_id)
            
            # Сигмоида больше не нужна, забираем готовые вероятности
            preds_probs = preds["probs"].cpu().numpy()
            uncertainty = preds["uncertainty"].cpu().numpy() 
            
            all_preds.append(preds_probs)
            all_targets.append(targets.cpu().numpy())
            all_uncertainties.append(uncertainty)
            
    # Объединяем батчи (B, 3, 8, 8)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_uncertainties = np.concatenate(all_uncertainties)

    binary_preds = np.max(all_preds, axis=1).flatten()
    binary_targets = np.max(all_targets, axis=1).flatten()
    
    if len(np.unique(binary_targets)) > 1:
        binary_pr_auc = average_precision_score(binary_targets, binary_preds)
        
        # Ищем лучший порог для бинарного F1
        b_precisions, b_recalls, b_thresholds = precision_recall_curve(binary_targets, binary_preds)
        b_f1_scores = np.divide(2 * b_precisions * b_recalls, b_precisions + b_recalls, 
                              out=np.zeros_like(b_precisions), where=(b_precisions + b_recalls) != 0)
        best_b_idx = np.argmax(b_f1_scores)
        binary_f1 = b_f1_scores[best_b_idx]
    else:
        binary_pr_auc, binary_f1 = 0.0, 0.0
        
    print(f"\n[{model_name}] --- Бинарная детекция хотспотов (Any DRC) ---")
    print(f"  Binary PR-AUC: {binary_pr_auc:.4f}")
    print(f"  Binary F1 (Max): {binary_f1*100:.2f}%")
    
    # --- ГЛОБАЛЬНЫЕ МЕТРИКИ ---
    preds_flat = all_preds.flatten()
    targets_flat = all_targets.flatten()
    uncert_flat = all_uncertainties.flatten()
    
    if len(np.unique(targets_flat)) > 1:
        roc_auc = roc_auc_score(targets_flat, preds_flat)
        global_pr_auc = average_precision_score(targets_flat, preds_flat)
        
        # --- ДИНАМИЧЕСКИЙ ПОИСК ЛУЧШЕГО F1 ---
        precisions, recalls, thresholds = precision_recall_curve(targets_flat, preds_flat)
        f1_scores = np.divide(2 * precisions * recalls, precisions + recalls, 
                              out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
        best_idx = np.argmax(f1_scores)
        global_f1 = f1_scores[best_idx] 
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    else:
        roc_auc, global_pr_auc, global_f1, best_threshold = 0.0, 0.0, 0.0, 0.5
        
    # --- ДИАГНОСТИКА УВЕРЕННОСТИ (EVIDENTIAL) ---
    # Смотрим на разницу неопределенности между правильными и ошибочными предсказаниями
    preds_binary = preds_flat > best_threshold
    
    tp_mask = (preds_binary == 1) & (targets_flat == 1) # True Positives
    fp_mask = (preds_binary == 1) & (targets_flat == 0) # False Positives
    fn_mask = (preds_binary == 0) & (targets_flat == 1) # False Negatives
    
    mean_uncert_tp = uncert_flat[tp_mask].mean() if tp_mask.sum() > 0 else 0.0
    mean_uncert_fp = uncert_flat[fp_mask].mean() if fp_mask.sum() > 0 else 0.0
    mean_uncert_fn = uncert_flat[fn_mask].mean() if fn_mask.sum() > 0 else 0.0
    
    print(f"\n[Дебаг Evidential Uncertainty - {model.__class__.__name__}]")
    print(f"  Средняя неопределенность (TP - Угадали дефект):  {mean_uncert_tp:.5f}")
    print(f"  Средняя неопределенность (FP - Ложная тревога):  {mean_uncert_fp:.5f}")
    print(f"  Средняя неопределенность (FN - Пропустили):      {mean_uncert_fn:.5f}")
        
    # --- ПОКЛАССОВЫЕ МЕТРИКИ PR-AUC ---
    # Индекс 0: Short, Индекс 1: Metal Spacing, Индекс 2: Others
    class_pr_aucs = []
    for c in range(3):
        t_c = all_targets[:, c, :, :].flatten()
        p_c = all_preds[:, c, :, :].flatten()
        
        if len(np.unique(t_c)) > 1:
            class_pr_aucs.append(average_precision_score(t_c, p_c))
        else:
            class_pr_aucs.append(0.0)

    preds_binary = preds_flat > best_threshold
    positives_mask = preds_binary == 1
    
    if positives_mask.sum() > 0:
        # Берем неопределенность только для предсказанных дефектов
        positives_uncert = uncert_flat[positives_mask]
        positives_targets = targets_flat[positives_mask]
        
        # Находим порог (отбрасываем 15% самых неуверенных СРЕДИ ПРЕДСКАЗАННЫХ)
        rejection_rate = 0.15
        uncert_threshold = np.percentile(positives_uncert, 100 * (1 - rejection_rate))
        
        # Оставляем только уверенные предсказания
        confident_positives_mask = positives_uncert <= uncert_threshold
        confident_targets = positives_targets[confident_positives_mask]
        
        # Считаем Precision (долю реальных дефектов среди всех срабатываний)
        # До фильтрации:
        base_precision = positives_targets.mean() 
        # После фильтрации:
        confident_precision = confident_targets.mean() if len(confident_targets) > 0 else 0.0
        
        print(f"  [Evidential] Base Precision (Все срабатывания): {base_precision*100:.1f}%")
        print(f"  [Evidential] Confident Precision (Без {rejection_rate*100}% сомнительных): {confident_precision*100:.1f}%")
    else:
        print("  [Evidential] Нет позитивных предсказаний для оценки.")
     
    eval_threshold = 0.5

    # 2. Берем только 2 главных физических класса (0: Short, 1: Spacing), убирая 3-й канал (Others)
    preds_main = all_preds[:, :2, :, :].flatten()
    targets_main = all_targets[:, :2, :, :].flatten()
    uncert_main = all_uncertainties[:, :2, :, :].flatten()

    # Считаем Rejection Curve по очищенным данным
    rates, curve_precisions = get_rejection_curve_data(
        preds_main, targets_main, uncert_main, eval_threshold
    )
            
    # Возвращаем: ROC-AUC, Global PR-AUC, Global F1, Spacing PR, Short PR, Others PR
    return roc_auc, global_pr_auc, global_f1, class_pr_aucs[1], class_pr_aucs[0], class_pr_aucs[2], rates, curve_precisions, binary_f1, binary_pr_auc    


def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}\n")
    
    TEST_DIR = r"J:\processed_dataset\val"
    TEST_PDK = "sky130" 
    PATCH_SIZE = 64
    BATCH_SIZE = 512
    
    '''WEIGHTS = {
        "Baseline": "ckpt/train_gf_true_DERICHLE/best_model_zeroshot_train_gf180_test_sky130_evidential.pth",
        "FiLM": "ckpt/train_gf_true_DERICHLE/best_model_zeroshot_train_gf180_test_sky130_evidential_FiLM.pth",
        "CrossAttention": "ckpt/train_gf_true_DERICHLE/best_model_zeroshot_train_gf180_test_sky130_evidential_attn.pth"
    }'''

    WEIGHTS = {
            "Baseline": "ckpt/train_sky_true_DERICHLE/best_model_zeroshot_train_sky130_test_gf180_evidential.pth",
            "FiLM": "ckpt/train_sky_true_DERICHLE/best_model_zeroshot_train_sky130_test_gf180_evidential_FiLM.pth",
            "CrossAttention": "ckpt/train_sky_true_DERICHLE/best_model_zeroshot_train_sky130_test_gf180_evidential_attn.pth"
        }

    '''WEIGHTS = {
        "Few-Shot (1% Data)": "ckpt/traingf180_DERICHLE_finetuned_sky130/best_fft_gf180_frac_0.01.pth",
        "Few-Shot (5% Data)": "ckpt/traingf180_DERICHLE_finetuned_sky130/best_fft_gf180_frac_0.05.pth"
    }'''
    
    print("Загрузка и фильтрация датасета...")
    dataset = EDAChipDataset(data_dir=TEST_DIR, stats_path="J:/processed_dataset/train/global_zscore_stats.json", patch_size=PATCH_SIZE, samples_per_file=2, mode='test')
    
    # Оставляем только файлы целевого завода
    test_indices = [i for i in range(len(dataset)) if TEST_PDK in str(dataset.files[dataset.pdk_grid[i][0]])]
    test_subset = Subset(dataset, test_indices)
    
    dataloader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Тестовых патчей {TEST_PDK.upper()}: {len(test_indices)}\n")
    
    results = []
    
    # !!! НОВОЕ !!! 
    # Создаем словарь для хранения кривых всех моделей и переменную для x-оси
    curves_data = {}
    saved_rates = None
    
    for model_name, weight_path in WEIGHTS.items():
        print(f"Оценка архитектуры: {model_name}...")
        
        # Инициализация нужной архитектуры
        if model_name == "CrossAttention":
            model = EvidentialLightweightRegionNetAttn(in_channels=3, num_classes=3, use_film=True).to(device)
        elif model_name == "FiLM":
            model = EvidentialLightweightRegionNet(in_channels=3, num_classes=3, use_film=True).to(device)
        else: # No_FiLM
            model = EvidentialLightweightRegionNet(in_channels=3, num_classes=3, use_film=False).to(device)
            
        # Загружаем веса
        try:
            model.load_state_dict(torch.load(weight_path, map_location=device))
        except FileNotFoundError:
            print(f"  [Ошибка] Файл {weight_path} не найден. Пропуск.")
            continue
            
        '''for model_name, weight_path in WEIGHTS.items():
        print(f"Оценка архитектуры: {model_name}...")
        
        # Инициализируем CrossAttention для обеих моделей
        model = EvidentialLightweightRegionNet(in_channels=3, num_classes=3, use_film=False).to(device)
            
        # Загружаем веса
        try:
            model.load_state_dict(torch.load(weight_path, map_location=device))
        except FileNotFoundError:
            print(f"  [Ошибка] Файл {weight_path} не найден. Пропуск.")
            continue
        '''
        
           
        # !!! НОВОЕ !!! 
        # Распаковываем 8 значений вместо 6, забирая данные для графика
        roc_auc, pr_auc, f1, m_spac, short_pr, others_pr, rates, curve_precisions, bin_f1, bin_pr_auc = evaluate_model(model, dataloader, device, model_name=model_name)
        
        # !!! НОВОЕ !!!
        # Сохраняем кривую в словарь, если предсказания были
        if curve_precisions is not None:
            curves_data[model_name] = curve_precisions
            saved_rates = rates # rates (0, 5, 10...) одинаковые для всех, сохраняем один раз
            
        results.append((model_name, TEST_PDK, roc_auc, pr_auc, f1, m_spac, short_pr, others_pr))
        
    # --- ВЫВОД КРАСИВОЙ ТАБЛИЦЫ ---
    print("\n" + "="*102 + " СВОДНАЯ ТАБЛИЦА МЕТРИК ZERO-SHOT " + "="*102)
    print(f"{'Model':>16} {'PDK / Scope':>13} {'ROC-AUC':>10} {'Global PR-AUC':>15} {'Global F1':>11} {'Metal Spacing PR-AUC':>22} {'Short PR-AUC':>14} {'Others PR-AUC':>15}")
    for r in results:
        print(f"{r[0]:>16} {r[1]:>13} {r[2]:10.4f} {r[3]:15.4f} {r[4]*100:10.2f}% {r[5]:22.4f} {r[6]:14.4f} {r[7]:15.4f}")
    print("="*238 + "\n")

    # !!! НОВОЕ !!!
    # В самом конце, когда все модели протестированы, рисуем один общий график
    if curves_data and saved_rates is not None:
        #plot_title = f"In-Domain Rejection Curves ({TEST_PDK} to {TEST_PDK})"
        plot_combined_rejection_curves(curves_dict=curves_data, rejection_rates=saved_rates)

if __name__ == "__main__":
    main()