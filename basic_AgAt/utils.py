import torch
import numpy as np
def iou_loss(preds, targets, smooth=1e-6):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return 1 - (intersection + smooth) / (union + smooth)

def compute_max_f(preds: torch.Tensor, masks: torch.Tensor, thresholds=255):
    preds_np = preds.cpu().numpy()
    masks_np = masks.cpu().numpy()

    max_f = 0
    for t in range(thresholds + 1):
        thresh = t / thresholds
        bin_preds = (preds_np >= thresh).astype(np.uint8)
        tp = np.sum(bin_preds * masks_np)
        fp = np.sum(bin_preds * (1 - masks_np))
        fn = np.sum((1 - bin_preds) * masks_np)

        if tp + fp == 0 or tp + fn == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        max_f = max(max_f, f)
    return max_f

def compute_mae(preds: torch.Tensor, masks: torch.Tensor):
    return torch.abs(preds - masks).mean().item()

def compute_mse(preds: torch.Tensor, masks: torch.Tensor):
    return torch.mean((preds - masks) ** 2).item()
