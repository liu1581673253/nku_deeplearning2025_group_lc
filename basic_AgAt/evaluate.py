import os
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score

def load_image_as_gray(path):
    return np.array(Image.open(path).convert('L'))

def threshold_predictions(pred_gray, thresholds):
    return [(pred_gray >= t).astype(np.uint8) for t in thresholds]

def calculate_metrics(pred_dir, gt_dir):
    max_f_scores = []
    mae_scores = []
    thresholds = np.arange(1, 256, 10)

    pred_files = os.listdir(pred_dir)
    gt_files = os.listdir(gt_dir)

    for file_name in pred_files:
        if file_name in gt_files:
            pred_gray = load_image_as_gray(os.path.join(pred_dir, file_name))
            gt_mask = (load_image_as_gray(os.path.join(gt_dir, file_name)) > 128).astype(np.uint8)

            f_scores = [
                f1_score(gt_mask.flatten(), bin_mask.flatten(), zero_division=1)
                for bin_mask in threshold_predictions(pred_gray, thresholds)
            ]
            max_f = max(f_scores)
            max_f_scores.append(max_f)

            mae = np.mean(np.abs(pred_gray / 255.0 - gt_mask))
            mae_scores.append(mae)

    avg_max_f = np.mean(max_f_scores)
    avg_mae = np.mean(mae_scores)

    return avg_max_f, avg_mae

pred_dir = 'predictions'
gt_dir = 'dataset/masks'

avg_max_f, avg_mae = calculate_metrics(pred_dir, gt_dir)
print(f"Average maxF: {avg_max_f:.4f}")
print(f"Average MAE: {avg_mae:.4f}")
