# %%
import numpy as np
import scipy.misc
from PIL import Image
import scipy.io
import os
import cv2
import time
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Make sure that caffe is on the python path:
caffe_root = '../../'
import sys

sys.path.insert(0, caffe_root + 'python')
import caffe

EPSILON = 1e-8

# 初始化指标存储
mae_values = []
f_measure_values = []
image_names = []

data_root = '/tmp/DSS/dataset/images/'
ground_truth_root = '/tmp/DSS/dataset/images/'  # 假设真实标签在此目录
with open('../../data/msra_b/test.lst') as f:
    test_lst = [x.strip() for x in f.readlines()]

# GPU设置
caffe.set_mode_gpu()
caffe.set_device(0)
caffe.SGDSolver.display = 0

# 加载网络
net = caffe.Net('deploy.prototxt', 'ras_iter_7500.caffemodel', caffe.TEST)

# 创建保存目录
save_root = '../../data/result/'
os.makedirs(save_root, exist_ok=True)

start_time = time.time()
processed_count = 0


def calculate_mae(pred, gt):
    """计算平均绝对误差"""
    return np.mean(np.abs(pred - gt))


def calculate_fmeasure(pred, gt, threshold=0.5):
    """计算F-measure"""
    pred_binary = (pred > threshold).astype(np.uint8)
    gt_binary = (gt > 0.5).astype(np.uint8)  # 假设ground truth是二值图

    precision = precision_score(gt_binary.flatten(), pred_binary.flatten())
    recall = recall_score(gt_binary.flatten(), pred_binary.flatten())
    f_measure = (1.3 * precision * recall) / (0.3 * precision + recall + EPSILON)

    return f_measure


for idx, img_path in enumerate(test_lst):
    try:
        # 完整的输入图片路径
        full_img_path = os.path.join(data_root, img_path)
        # 假设ground truth文件名与输入图片相同，但可能扩展名不同
        gt_path = os.path.join(ground_truth_root, os.path.splitext(img_path)[0] + '.png')

        # 加载图片和ground truth
        img = Image.open(full_img_path)
        img = np.array(img, dtype=np.uint8)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            print(f"Warning: Ground truth not found for {img_path}")
            continue

        # 预处理
        im = np.array(img, dtype=np.float32)
        im = im[:, :, ::-1]  # BGR转换
        im -= np.array((104.00698793, 116.66876762, 122.67891434))
        im = im.transpose((2, 0, 1))

        # 网络输入
        net.blobs['data'].reshape(1, *im.shape)
        net.blobs['data'].data[...] = im

        # 前向传播
        net.forward()
        res = net.blobs['sigmoid-score1'].data[0][0, :, :]

        # 归一化
        res = (res - np.min(res) + EPSILON) / (np.max(res) - np.min(res) + EPSILON)
        res = 255 * res

        # 保存结果
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(save_root, f"{base_name}.png")
        cv2.imwrite(output_path, res)

        # 计算指标
        gt_normalized = gt.astype(np.float32) / 255.0
        res_normalized = res.astype(np.float32) / 255.0

        # 调整大小匹配（如果预测和GT尺寸不一致）
        if res_normalized.shape != gt_normalized.shape:
            res_normalized = cv2.resize(res_normalized, (gt_normalized.shape[1], gt_normalized.shape[0]))

        mae = calculate_mae(res_normalized, gt_normalized)
        f_measure = calculate_fmeasure(res_normalized, gt_normalized)

        mae_values.append(mae)
        f_measure_values.append(f_measure)
        image_names.append(base_name)

        processed_count += 1

        # 打印进度
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(test_lst)} images")
            print(f"Current MAE: {mae:.4f}, F-measure: {f_measure:.4f}")

    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        continue

# 计算总体指标
avg_mae = np.mean(mae_values)
avg_f_measure = np.mean(f_measure_values)

# 输出总体结果
diff_time = time.time() - start_time
print(f'\nDetection took {diff_time:.3f}s for {processed_count} images')
print(f'Average time per image: {diff_time / max(1, processed_count):.3f}s')
print(f'Average MAE: {avg_mae:.4f}')
print(f'Average F-measure: {avg_f_measure:.4f}')

# 绘制指标走势图
plt.figure(figsize=(12, 6))

# MAE走势
plt.subplot(1, 2, 1)
plt.plot(range(len(mae_values)), mae_values, 'b-', label='MAE')
plt.axhline(y=avg_mae, color='r', linestyle='--', label=f'Avg MAE: {avg_mae:.4f}')
plt.xlabel('Image Index')
plt.ylabel('MAE Value')
plt.title('MAE Trend')
plt.legend()

# F-measure走势
plt.subplot(1, 2, 2)
plt.plot(range(len(f_measure_values)), f_measure_values, 'g-', label='F-measure')
plt.axhline(y=avg_f_measure, color='r', linestyle='--', label=f'Avg F-measure: {avg_f_measure:.4f}')
plt.xlabel('Image Index')
plt.ylabel('F-measure Value')
plt.title('F-measure Trend')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_root, 'performance_trend.png'))
plt.show()

# 保存详细指标结果
with open(os.path.join(save_root, 'metrics_results.txt'), 'w') as f:
    f.write('Image Name,MAE,F-measure\n')
    for name, mae, fm in zip(image_names, mae_values, f_measure_values):
        f.write(f'{name},{mae:.4f},{fm:.4f}\n')
    f.write(f'\nAverage,{avg_mae:.4f},{avg_f_measure:.4f}\n')

print("Metrics saved to performance_trend.png and metrics_results.txt")
# %%