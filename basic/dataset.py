import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SaliencyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.jpeg')])
        self.mask_filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

        assert len(self.image_filenames) == len(self.mask_filenames), \
            "图像和掩码数量不匹配"

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 单通道灰度

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
