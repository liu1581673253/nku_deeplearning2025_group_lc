import torch
import numpy as np

def compute_max_f(preds: torch.Tensor, masks: torch.Tensor, thresholds=255):
    """
    preds, masks: [N,1,H,W], float tensor [0,1]
    计算最大F值(MaxF)
    """
    preds_np = preds.cpu().numpy()
    masks_np = masks.cpu().numpy()

    max_f = 0
    for t in range(thresholds+1):
        thresh = t / thresholds
        bin_preds = (preds_np >= thresh).astype(np.uint8)
        tp = np.sum(bin_preds * masks_np)
        fp = np.sum(bin_preds * (1 - masks_np))
        fn = np.sum((1 - bin_preds) * masks_np)

        if tp + fp == 0 or tp + fn == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if prec + rec == 0:
            f = 0
        else:
            f = 2 * prec * rec / (prec + rec)
        if f > max_f:
            max_f = f
    return max_f

def compute_mae(preds: torch.Tensor, masks: torch.Tensor):
    """
    计算平均绝对误差 MAE
    """
    mae = torch.abs(preds - masks).mean().item()
    return mae

def compute_mse(preds: torch.Tensor, masks: torch.Tensor):
    """
    计算均方误差 MSE
    """
    mse = torch.mean((preds - masks) ** 2).item()
    return mse
