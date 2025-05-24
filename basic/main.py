import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from PIL import Image

from dataset import SaliencyDataset
from model import SaliencyModel
from utils import compute_max_f, compute_mae, compute_mse

def save_predictions(model, dataloader, device, output_dir, original_image_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            preds = model(images)  # [B,1,H,W]

            preds_np = preds.squeeze(1).cpu().numpy()  # [B,H,W]

            for i in range(preds_np.shape[0]):
                # Subset索引映射到原始dataset索引
                dataset_idx = dataloader.dataset.indices[batch_idx * dataloader.batch_size + i]
                base_name = os.path.splitext(dataloader.dataset.dataset.image_filenames[dataset_idx])[0]

                # 读取原始图尺寸
                orig_img_path = os.path.join(original_image_dir, base_name + ".jpg")
                orig_img = Image.open(orig_img_path)
                orig_size = orig_img.size  # (width, height)

                # 预测resize到原始尺寸
                pred_img = preds_np[i]
                pred_img_pil = Image.fromarray((pred_img * 255).astype('uint8'))
                pred_img_pil = pred_img_pil.resize(orig_size, Image.BILINEAR)

                # 保存结果
                save_path = os.path.join(output_dir, base_name + ".png")
                pred_img_pil.save(save_path)

    print(f"预测结果已保存到 {output_dir}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_dir = "./dataset/images"
    mask_dir = "./dataset/masks"

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
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_losses, test_losses = [], []
    maxf_scores, mae_scores = [], []

    best_maxf = 0.0
    for epoch in range(1, 5):
        model.train()
        epoch_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        test_loss = 0.0
        all_preds, all_masks = [], []
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                loss = criterion(preds, masks)
                test_loss += loss.item()

                all_preds.append(preds.cpu())
                all_masks.append(masks.cpu())

        test_losses.append(test_loss / len(test_loader))

        preds_all = torch.cat(all_preds, dim=0)
        masks_all = torch.cat(all_masks, dim=0)

        maxf = compute_max_f(preds_all, masks_all)
        mae = compute_mae(preds_all, masks_all)
        maxf_scores.append(maxf)
        mae_scores.append(mae)

        print(f"Epoch {epoch}: Train Loss={train_losses[-1]:.4f}, Test Loss={test_losses[-1]:.4f}, MaxF={maxf:.4f}, MAE={mae:.4f}")

        if maxf > best_maxf:
            best_maxf = maxf
            torch.save(model.state_dict(), "best_saliency_model.pth")
            print(f"保存最佳模型，Epoch {epoch}, MaxF={maxf:.4f}")

    # 绘制Loss, MaxF, MAE曲线
    plt.figure(figsize=(10,6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.plot(maxf_scores, label='MaxF')
    plt.plot(mae_scores, label='MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Score / Loss')
    plt.title('训练过程指标变化')
    plt.legend()
    plt.grid()
    plt.savefig("training_curve.png")
    plt.show()

    # 加载最佳模型，保存测试集预测结果
    model.load_state_dict(torch.load("best_saliency_model.pth", map_location=device))
    save_predictions(model, test_loader, device, output_dir="./predictions", original_image_dir=image_dir)

if __name__ == "__main__":
    main()
