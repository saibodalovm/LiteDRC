"""
F1 с distance-based matching (Hungarian) для CenterNet-style DRC detection.
 
Заменяет compute_metrics() из train.py: вместо dilation по пикселям и
NMS-равенства, здесь предсказанные и истинные центры извлекаются как точки
(с суб-пиксельной коррекцией через offset head/target), и сопоставляются
оптимальным (Hungarian) алгоритмом по реальному евклидову расстоянию в
GCell-ах, с порогом. Это устраняет два источника шума из старой метрики:
  - "блинчатые" (не острые) предсказания больше не дают ложных множественных
    пиков на плато (см. compute_metrics comment #3 в прошлом обсуждении);
  - жадное/пиксельное сопоставление больше не выбирает пары произвольно,
    если рядом несколько кандидатов — Hungarian даёт глобально оптимальное
    паросочетание.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
 
 
def extract_peaks(prob_map: torch.Tensor, offset_map: torch.Tensor = None,
                   threshold: float = 0.1, nms_kernel: int = 3):
    """
    prob_map: (H, W) — вероятности после sigmoid, ОДИН сэмпл (не батч).
    offset_map: (2, H, W) или None — если передан, координаты пиков
    корректируются суб-пиксельно (offset[0]=dx, offset[1]=dy).
    Возвращает: points (N,2) в формате (row, col) float, scores (N,)
    """
    prob = prob_map.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    pad = nms_kernel // 2
    hmax = F.max_pool2d(prob, kernel_size=nms_kernel, stride=1, padding=pad)
    keep = (prob == hmax) & (prob > threshold)
    keep = keep.squeeze(0).squeeze(0)  # (H,W)
 
    rows, cols = torch.where(keep)
    scores = prob_map[rows, cols]
 
    rows_f = rows.float()
    cols_f = cols.float()
    if offset_map is not None:
        dx = offset_map[0, rows, cols]
        dy = offset_map[1, rows, cols]
        cols_f = cols_f + dx
        rows_f = rows_f + dy
 
    points = torch.stack([rows_f, cols_f], dim=1).cpu().numpy()
    return points, scores.cpu().numpy()
 
 
def extract_true_points(mask: torch.Tensor, offset_map: torch.Tensor, class_map: torch.Tensor):
    """
    mask: (H, W) bool/float — target_offset_mask (дискретные позитивные ячейки)
    offset_map: (2, H, W) — target_offset
    class_map: (H, W) — target_class_map
    Возвращает: points (N,2) в (row, col), classes (N,)
    """
    rows, cols = torch.where(mask > 0)
    dx = offset_map[0, rows, cols]
    dy = offset_map[1, rows, cols]
    rows_f = rows.float() + dy
    cols_f = cols.float() + dx
    points = torch.stack([rows_f, cols_f], dim=1).cpu().numpy()
    classes = class_map[rows, cols].cpu().numpy()
    return points, classes
 
 
def match_points(pred_points: np.ndarray, true_points: np.ndarray, dist_threshold: float):
    """
    Оптимальное (Hungarian) сопоставление по евклидову расстоянию с порогом.
    Возвращает: list[(pred_idx, true_idx)] — только валидные (dist <= threshold) пары.
    """
    if len(pred_points) == 0 or len(true_points) == 0:
        return []
 
    cost = cdist(pred_points, true_points)  # (n_pred, n_true), в единицах GCell
    # Запрещаем пары дальше порога, подставляя большую константу вместо inf
    # (linear_sum_assignment не любит inf при дисбалансе строк/столбцов)
    big = dist_threshold * 1000.0 + 1.0
    cost_capped = np.where(cost > dist_threshold, big, cost)
 
    row_ind, col_ind = linear_sum_assignment(cost_capped)
    matches = [(r, c) for r, c in zip(row_ind, col_ind) if cost[r, c] <= dist_threshold]
    return matches
 
 
def evaluate_sample(pred_obj_prob, pred_offset, pred_cls_logits,
                     true_mask, true_offset, true_cls,
                     obj_threshold=0.1, dist_threshold_gcells=2.0, nms_kernel=3):
    """
    Метрики для ОДНОГО сэмпла (без батч-размерности).
    pred_obj_prob: (H,W) уже после sigmoid
    pred_offset:   (2,H,W)
    pred_cls_logits: (num_classes,H,W)
    true_mask: (H,W), true_offset: (2,H,W), true_cls: (H,W)
    """
    pred_points, pred_scores = extract_peaks(
        pred_obj_prob, pred_offset, threshold=obj_threshold, nms_kernel=nms_kernel
    )
    true_points, true_classes = extract_true_points(true_mask, true_offset, true_cls)
 
    matches = match_points(pred_points, true_points, dist_threshold_gcells)
 
    TP = len(matches)
    FP = len(pred_points) - TP
    FN = len(true_points) - TP
 
    cls_correct = 0
    if TP > 0:
        pred_cls_map = pred_cls_logits.argmax(dim=0)  # (H,W)
        for p_idx, t_idx in matches:
            r, c = pred_points[p_idx]
            r, c = int(round(r)), int(round(c))
            r = min(max(r, 0), pred_cls_map.shape[0] - 1)
            c = min(max(c, 0), pred_cls_map.shape[1] - 1)
            pred_class = pred_cls_map[r, c].item()
            if pred_class == true_classes[t_idx]:
                cls_correct += 1
 
    return TP, FP, FN, cls_correct, len(true_points), len(pred_points)
 
 
def evaluate_batch(preds: dict, targets: dict, obj_threshold=0.1, dist_threshold_gcells=2.0, nms_kernel=3):
    """
    preds: {'obj': (B,1,H,W) logits, 'cls': (B,C,H,W) logits, 'offset': (B,2,H,W)}
    targets: {'mask': (B,1,H,W), 'offset': (B,2,H,W), 'cls': (B,H,W)}
    Возвращает суммарные TP, FP, FN, cls_correct, n_true, n_pred по батчу.
    """
    B = preds['obj'].shape[0]
    pred_obj_prob = torch.sigmoid(preds['obj']).squeeze(1)  # (B,H,W)
 
    totals = dict(TP=0, FP=0, FN=0, cls_correct=0, n_true=0, n_pred=0)
    for b in range(B):
        TP, FP, FN, cls_correct, n_true, n_pred = evaluate_sample(
            pred_obj_prob[b], preds['offset'][b], preds['cls'][b],
            targets['mask'][b, 0], targets['offset'][b], targets['cls'][b],
            obj_threshold=obj_threshold,
            dist_threshold_gcells=dist_threshold_gcells,
            nms_kernel=nms_kernel,
        )
        totals['TP'] += TP
        totals['FP'] += FP
        totals['FN'] += FN
        totals['cls_correct'] += cls_correct
        totals['n_true'] += n_true
        totals['n_pred'] += n_pred
 
    precision = totals['TP'] / (totals['TP'] + totals['FP'] + 1e-6)
    recall = totals['TP'] / (totals['TP'] + totals['FN'] + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    cls_acc = totals['cls_correct'] / (totals['TP'] + 1e-6) if totals['TP'] > 0 else 0.0
 
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cls_acc_on_tp": cls_acc,
        **totals,
    }
 
 
if __name__ == "__main__":
    # Синтетическая проверка на маленьком примере, чтобы убедиться, что
    # matching действительно ведёт себя разумно (без реального train-датасета).
    H, W = 32, 32
    true_mask = torch.zeros(H, W)
    true_offset = torch.zeros(2, H, W)
    true_cls = torch.zeros(H, W, dtype=torch.long)
 
    # два истинных центра
    true_mask[10, 10] = 1; true_offset[:, 10, 10] = torch.tensor([0.3, -0.2]); true_cls[10, 10] = 1
    true_mask[20, 5] = 1; true_offset[:, 20, 5] = torch.tensor([-0.1, 0.4]); true_cls[20, 5] = 2
 
    pred_obj_prob = torch.zeros(H, W)
    pred_offset = torch.zeros(2, H, W)
    pred_cls_logits = torch.zeros(3, H, W)
 
    # одно предсказание рядом с первым истинным центром (должно матчиться)
    pred_obj_prob[10, 11] = 0.9
    pred_cls_logits[1, 10, 11] = 5.0  # argmax -> класс 1, верно
 
    # одно предсказание далеко ото всех (должно стать FP)
    pred_obj_prob[30, 30] = 0.8
    pred_cls_logits[2, 30, 30] = 5.0
 
    res = evaluate_sample(pred_obj_prob, pred_offset, pred_cls_logits,
                           true_mask, true_offset, true_cls,
                           obj_threshold=0.1, dist_threshold_gcells=2.0)
    print("TP, FP, FN, cls_correct, n_true, n_pred =", res)
    # Ожидаем: TP=1 (совпадение рядом с (10,10)), FP=1 (точка (30,30)), FN=1 (пропущен (20,5))