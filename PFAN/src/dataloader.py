from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys
import cv2
import numpy as np
import glob
import os

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def pad_resize_image(inp_img, out_img=None, target_size=None):
    """
    Function to pad and resize images to a given size.
    out_img is None only during inference. During training and testing
    out_img is NOT None.
    :param inp_img: A H x W x C input image.
    :param out_img: A H x W input image of mask.
    :param target_size: The size of the final images.
    :return: Re-sized inp_img and out_img
    """
    h, w, c = inp_img.shape
    size = max(h, w)

    padding_h = (size - h) // 2
    padding_w = (size - w) // 2

    if out_img is None:
        # For inference
        temp_x = cv2.copyMakeBorder(inp_img, top=padding_h, bottom=padding_h, left=padding_w, right=padding_w,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if target_size is not None:
            temp_x = cv2.resize(temp_x, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return temp_x
    else:
        # For training and testing
        temp_x = cv2.copyMakeBorder(inp_img, top=padding_h, bottom=padding_h, left=padding_w, right=padding_w,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        temp_y = cv2.copyMakeBorder(out_img, top=padding_h, bottom=padding_h, left=padding_w, right=padding_w,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # print(inp_img.shape, temp_x.shape, out_img.shape, temp_y.shape)

        if target_size is not None:
            temp_x = cv2.resize(temp_x, (target_size, target_size), interpolation=cv2.INTER_AREA)
            temp_y = cv2.resize(temp_y, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return temp_x, temp_y


def random_crop_flip(inp_img, out_img):
    """
    Function to randomly crop and flip images.
    :param inp_img: A H x W x C input image.
    :param out_img: A H x W input image.
    :return: The randomly cropped and flipped image.
    """
    h, w = out_img.shape

    rand_h = np.random.randint(h/8)
    rand_w = np.random.randint(w/8)
    offset_h = 0 if rand_h == 0 else np.random.randint(rand_h)
    offset_w = 0 if rand_w == 0 else np.random.randint(rand_w)
    p0, p1, p2, p3 = offset_h, h+offset_h-rand_h, offset_w, w+offset_w-rand_w

    rand_flip = np.random.randint(10)
    if rand_flip >= 5:
        inp_img = inp_img[::, ::-1, ::]
        out_img = out_img[::, ::-1]

    return inp_img[p0:p1, p2:p3], out_img[p0:p1, p2:p3]


def random_rotate(inp_img, out_img, max_angle=25):
    """
    Function to randomly rotate images within +max_angle to -max_angle degrees.
    This algorithm does NOT crops the edges upon rotation.
    :param inp_img: A H x W x C input image.
    :param out_img: A H x W input image.
    :param max_angle: Maximum angle an image can be rotated in either direction.
    :return: The randomly rotated image.
    """
    angle = np.random.randint(-max_angle, max_angle)
    h, w = out_img.shape
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Compute new dimensions of the image and adjust the rotation matrix
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(inp_img, M, (new_w, new_h)), cv2.warpAffine(out_img, M, (new_w, new_h))


def random_rotate_lossy(inp_img, out_img, max_angle=25):
    """
    Function to randomly rotate images within +max_angle to -max_angle degrees.
    This algorithm crops the edges upon rotation.
    :param inp_img: A H x W x C input image.
    :param out_img: A H x W input image.
    :param max_angle: Maximum angle an image can be rotated in either direction.
    :return: The randomly rotated image.
    """
    angle = np.random.randint(-max_angle, max_angle)
    h, w = out_img.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(inp_img, M, (w, h)), cv2.warpAffine(out_img, M, (w, h))


def random_brightness(inp_img):
    """
    Function to randomly perturb the brightness of the input images.
    :param inp_img: A H x W x C input image.
    :return: The image with randomly perturbed brightness.
    """
    contrast = np.random.rand(1) + 0.5
    light = np.random.randint(-20, 20)
    inp_img = contrast * inp_img + light

    return np.clip(inp_img, 0, 255)


class SODLoader(Dataset):
    """
    适配CUHK数据集的DataLoader
    目录结构：
    data/
    └── CUHK_Saliency/
        ├── train/
        │   ├── images/  # 训练图像
        │   └── masks/   # 对应掩码
        └── test/
            ├── images/  # 测试图像
            └── masks/   # 对应掩码
    """
    def __init__(self, mode='train', augment_data=False, target_size=256):
        # 修改路径配置
        base_path = "./data/CUHK_Saliency"
        if mode == 'train':
            self.inp_path = os.path.join(base_path, "train/images")
            self.out_path = os.path.join(base_path, "train/masks")
        elif mode == 'test':
            self.inp_path = os.path.join(base_path, "test/images")
            self.out_path = os.path.join(base_path, "test/masks")
        else:
            raise ValueError("mode必须为 'train' 或 'test'")

        # 验证路径有效性
        if not os.path.exists(self.inp_path):
            raise FileNotFoundError(f"图像路径不存在: {self.inp_path}")
        if not os.path.exists(self.out_path):
            raise FileNotFoundError(f"掩码路径不存在: {self.out_path}")

        # 初始化参数（保持原有）
        self.augment_data = augment_data
        self.target_size = target_size
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # 获取排序后的文件列表（确保图像-掩码对应）
        self.inp_files = sorted(glob.glob(os.path.join(self.inp_path, "*.jpg")))
        self.out_files = sorted(glob.glob(os.path.join(self.out_path, "*.png")))

        # 验证数据一致性
        self._validate_pairing()

    def _validate_pairing(self):
        """验证图像与掩码文件名严格对应"""
        for img_path, mask_path in zip(self.inp_files, self.out_files):
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            mask_id = os.path.splitext(os.path.basename(mask_path))[0]
            if img_id != mask_id:
                raise ValueError(
                    f"文件名不匹配: 图像 {img_id} vs 掩码 {mask_id}"
                )

    def __getitem__(self, idx):
        # 读取图像
        inp_img = cv2.imread(self.inp_files[idx])
        inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
        inp_img = inp_img.astype('float32')

        # 读取掩码
        mask_img = cv2.imread(self.out_files[idx], 0)
        if mask_img is None:
            raise IOError(f"无法读取掩码文件: {self.out_files[idx]}")
        mask_img = mask_img.astype('float32')
        mask_img /= np.max(mask_img)  # 归一化到[0,1]

        # 数据增强
        if self.augment_data:
            inp_img, mask_img = random_crop_flip(inp_img, mask_img)
            inp_img, mask_img = random_rotate(inp_img, mask_img)
            inp_img = random_brightness(inp_img)

        # 尺寸调整
        inp_img, mask_img = pad_resize_image(
            inp_img, mask_img, self.target_size
        )
        
        # 数据标准化
        inp_img = np.clip(inp_img / 255.0, 0, 1)  # 显式归一化
        inp_img = np.transpose(inp_img, (2, 0, 1))
        inp_img = torch.as_tensor(inp_img, dtype=torch.float32)
        inp_img = self.normalize(inp_img)

        mask_img = np.expand_dims(mask_img, axis=0)  # 增加通道维度
        return inp_img, torch.from_numpy(mask_img).float()

    def __len__(self):
        return len(self.inp_files)


class InfDataloader(Dataset):
    """
    Dataloader for Inference.
    """
    def __init__(self, img_folder, target_size=256):
        # 添加路径验证
        if not os.path.exists(img_folder):
            raise FileNotFoundError(f"推理图像目录不存在: {img_folder}")

        self.img_paths = sorted([
            p for p in glob.glob(os.path.join(img_folder, "*"))
            if p.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        self.target_size = target_size
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __getitem__(self, idx):
        """
        __getitem__ for inference
        :param idx: Index of the image
        :return: img_np is a numpy RGB-image of shape H x W x C with pixel values in range 0-255.
        And img_tor is a torch tensor, RGB, C x H x W in shape and normalized.
        """
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Pad images to target size
        img_np = pad_resize_image(img, None, self.target_size)
        img_tor = img_np.astype(np.float32)
        img_tor = img_tor / 255.0
        img_tor = np.transpose(img_tor, axes=(2, 0, 1))
        img_tor = torch.from_numpy(img_tor).float()
        img_tor = self.normalize(img_tor)

        return img_np, img_tor

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    # ====== 极简版数据加载测试 ======
    img_size = 256
    bs = 8
    
    # 快速验证数据加载
    print("[快速测试] 创建数据集实例...")
    train_data = SODLoader(mode='train', augment_data=True, target_size=img_size)
    test_data = SODLoader(mode='test', augment_data=False, target_size=img_size)
    print(f"✅ 训练集样本数: {len(train_data)}, 测试集样本数: {len(test_data)}")

    # 极简数据加载
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=4)

    # 快速检查3个批次
    print("\n[训练数据检查] 前3个批次形状:")
    for i, (inputs, masks) in enumerate(train_loader):
        print(f"批次 {i}: inputs {inputs.shape}, masks {masks.shape}")
        if i == 2:
            break

    # 快速测试数据增强
    print("\n[数据增强测试] 随机采样检查:")
    sample_img = cv2.imread(os.path.join(train_data.inp_path, "001.jpg"))[..., ::-1]  # BGR转RGB
    sample_mask = cv2.imread(os.path.join(train_data.out_path, "001.png"), 0)
    
    # 执行增强流程
    x, y = random_crop_flip(sample_img, sample_mask)
    x, y = random_rotate(x, y)
    x = random_brightness(x)
    x_pad, y_pad = pad_resize_image(x, y, img_size)
    
    # 输出关键信息
    print(f"原始尺寸: {sample_img.shape} → 处理后: {x_pad.shape}")
    print("⚠️ 注意：在无GUI环境中自动跳过图像显示")