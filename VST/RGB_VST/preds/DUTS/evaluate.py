import os
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error

# 定义读取图像并转换为二值mask
def load_mask(mask_path):
    # 读取mask图像并转换为灰度图
    img = Image.open(mask_path).convert('L')
    img = np.array(img)
    # 将图像转换为0和1的二值图像，假设显著性区域为白色(255)，背景为黑色(0)
    return (img > 128).astype(np.uint8)

# 计算F-measure和MAE
def calculate_metrics(pred_dir, gt_dir):
    f1_scores = []
    mae_scores = []

    # 获取预测和真实mask文件名列表
    pred_files = os.listdir(pred_dir)
    gt_files = os.listdir(gt_dir)

    for file_name in pred_files:
        if file_name in gt_files:
            # 加载预测mask和真实mask
            pred_mask = load_mask(os.path.join(pred_dir, file_name))
            gt_mask = load_mask(os.path.join(gt_dir, file_name))

            # 计算F1分数
            f1 = f1_score(gt_mask.flatten(), pred_mask.flatten())
            f1_scores.append(f1)

            # 计算MAE
            mae = mean_absolute_error(gt_mask.flatten(), pred_mask.flatten())
            mae_scores.append(mae)

    avg_f1 = np.mean(f1_scores)
    avg_mae = np.mean(mae_scores)

    return avg_f1, avg_mae

# 设置预测mask和真实mask的文件夹路径
pred_dir = 'RGB_VST'
gt_dir = 'DUTS-TE-Mask'

# 计算并输出F-measure和MAE
avg_f1, avg_mae = calculate_metrics(pred_dir, gt_dir)
print(f"Average F-measure: {avg_f1:.4f}")
print(f"Average MAE: {avg_mae:.4f}")