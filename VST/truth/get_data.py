import os
import random
import shutil

# 设置路径
truth_dir = "../truth"  # 替换为实际的truth文件夹路径
output_dir = "DUTS"  # 替换为你要输出的文件夹路径

# 创建输出文件夹结构
train_dir = os.path.join(output_dir, "DUTS-TR")
test_dir = os.path.join(output_dir, "DUTS-TE")

# 训练集和测试集文件夹
os.makedirs(os.path.join(train_dir, "DUTS-TR-Image"), exist_ok=True)
os.makedirs(os.path.join(train_dir, "DUTS-TR-Mask"), exist_ok=True)
os.makedirs(os.path.join(train_dir, "DUTS-TR-Contour"), exist_ok=True)

os.makedirs(os.path.join(test_dir, "DUTS-TE-Image"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "DUTS-TE-Mask"), exist_ok=True)

# 获取文件名列表
contour_files = sorted(os.listdir(os.path.join(truth_dir, "Contour")))
mask_files = sorted(os.listdir(os.path.join(truth_dir, "ground_truth_mask")))
image_files = sorted(os.listdir(os.path.join(truth_dir, "images")))

# 确保文件列表长度一致
assert len(contour_files) == len(mask_files) == len(image_files), "文件夹中文件数不一致"

# 创建数据列表（图像，掩码，边缘图）
data = list(zip(image_files, mask_files, contour_files))

# 打乱数据
random.shuffle(data)
print(len(data))
# 分割数据，700个训练集，300个测试集
train_data = data[:700]
test_data = data[700:]

# 复制文件到对应目录
def copy_files(data, src_dir, dst_dir, image_folder, mask_folder, contour_folder=None):
    for image, mask, contour in data:
        # 训练集图像
        shutil.copy(os.path.join(src_dir, "images", image), os.path.join(dst_dir, image_folder, image))
        # 训练集掩码
        shutil.copy(os.path.join(src_dir, "ground_truth_mask", mask), os.path.join(dst_dir, mask_folder, mask))
        # 训练集边缘图（仅对训练集需要）
        if contour_folder:
            shutil.copy(os.path.join(src_dir, "Contour", contour), os.path.join(dst_dir, contour_folder, contour))

# 复制训练集文件
copy_files(train_data, truth_dir, train_dir, "DUTS-TR-Image", "DUTS-TR-Mask", "DUTS-TR-Contour")

# 复制测试集文件（不需要边缘图）
copy_files(test_data, truth_dir, test_dir, "DUTS-TE-Image", "DUTS-TE-Mask", contour_folder=None)

print("数据分割完成！")