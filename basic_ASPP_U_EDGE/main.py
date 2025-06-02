# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import numpy as np
from dataset import SaliencyDataset
from model import SaliencyModel
from utils import compute_max_f, compute_mae


# Dice Loss
def dice_loss(preds, targets, smooth=1e-6):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return 1 - (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


# BCE + Dice 混合损失
def hybrid_loss(preds, targets, bce_fn):
    bce = bce_fn(preds, targets)
    dice = dice_loss(preds, targets)
    return bce + dice


# 简易边缘标签生成
def get_edge_label(mask):
    # 水平/垂直方向梯度
    kernel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], device=mask.device).float()
    kernel_y = kernel_x.transpose(2, 3)
    
    grad_x = F.conv2d(mask, kernel_x, padding=1)
    grad_y = F.conv2d(mask, kernel_y, padding=1)
    edge = torch.sqrt(grad_x**2 + grad_y**2)
    return (edge > 0.1).float()

def save_predictions(model, dataloader, device, output_dir, original_image_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            preds, _ = model(images)
            preds_np = preds.squeeze(1).cpu().numpy()

            for i in range(preds_np.shape[0]):
                dataset_idx = dataloader.dataset.indices[batch_idx * dataloader.batch_size + i]
                base_name = os.path.splitext(dataloader.dataset.dataset.image_filenames[dataset_idx])[0]
                orig_img_path = os.path.join(original_image_dir, base_name + ".jpg")
                orig_img = Image.open(orig_img_path)
                orig_size = orig_img.size

                pred_img = preds_np[i]
                pred_img_pil = Image.fromarray((pred_img * 255).astype('uint8'))
                pred_img_pil = pred_img_pil.resize(orig_size, Image.BILINEAR)

                save_path = os.path.join(output_dir, base_name + ".png")
                pred_img_pil.save(save_path)

    print(f"预测结果已保存到 {output_dir}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    image_dir = "./dataset/images"
    mask_dir = "./dataset/masks"
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    full_dataset = SaliencyDataset(image_dir, mask_dir, transform)
    train_size = 700
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = SaliencyModel(pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    bce_fn = nn.BCELoss()

    epochs = 50
    edge_loss_weight = 0.6  # λ 边缘损失权重

    train_losses = []
    test_mae_list = []
    test_maxf_list = []

    best_maxf = -1.0
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            preds, edge_preds = model(images)
            edge_labels = get_edge_label(masks)

            loss_saliency = hybrid_loss(preds, masks, bce_fn)
            loss_edge = hybrid_loss(edge_preds, edge_labels, bce_fn)
            loss = loss_saliency + edge_loss_weight * loss_edge

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)

        # 验证
        model.eval()
        mae_sum = 0
        maxf_sum = 0
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                masks = masks.to(device)
                preds, _ = model(images)

                mae_sum += compute_mae(preds, masks) * images.size(0)
                maxf_sum += compute_max_f(preds, masks) * images.size(0)

        avg_mae = mae_sum / len(test_loader.dataset)
        avg_maxf = maxf_sum / len(test_loader.dataset)
        test_mae_list.append(avg_mae)
        test_maxf_list.append(avg_maxf)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - MAE: {avg_mae:.4f} - MaxF: {avg_maxf:.4f}")

        # 保存最优模型
        if avg_maxf > best_maxf:
            best_maxf = avg_maxf
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            print(f">>> 新最佳模型已保存（MaxF: {best_maxf:.4f} @ Epoch {best_epoch})")

    print(f"训练结束，最佳 MaxF: {best_maxf:.4f} 出现在第 {best_epoch} 轮")

    # 使用最佳模型做预测
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_model.pth")))
    save_predictions(model, test_loader, device, "./outputs/pred", image_dir)

    # 绘图
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), test_mae_list, label="Test MAE")
    plt.plot(range(1, epochs + 1), test_maxf_list, label="Test MaxF")
    plt.xlabel("Epoch")
    plt.legend()
    plt.title("Training Curve")
    plt.show()


if __name__ == "__main__":
    main()
