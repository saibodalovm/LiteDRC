import numpy as np
import torch

def compute_region_metrics(preds_probs, targets, threshold=0.5):
    preds_bin = (preds_probs > threshold).float()
    targets_bin = targets.float()
    
    pred_any = preds_bin.any(dim=1)
    target_any = targets_bin.any(dim=1)
    
    tp = (pred_any & target_any).sum().item()
    fp = (pred_any & ~target_any).sum().item()
    fn = (~pred_any & target_any).sum().item()
    
    cls_tp = (preds_bin * targets_bin).sum(dim=(0, 2, 3)).cpu().numpy()
    cls_fp = (preds_bin * (1 - targets_bin)).sum(dim=(0, 2, 3)).cpu().numpy()
    cls_fn = ((1 - preds_bin) * targets_bin).sum(dim=(0, 2, 3)).cpu().numpy()
    
    return tp, fp, fn, cls_tp, cls_fp, cls_fn
