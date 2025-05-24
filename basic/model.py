import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SaliencyModel(nn.Module):
    def __init__(self, pretrained=True):
        super(SaliencyModel, self).__init__()
        # 载入ResNet18主干网络
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        # 去掉最后的fc层和avgpool
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # 输出 [B,512,H/32,W/32]

        # 解码器：先上采样到原图大小224x224（默认输入），再conv
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 7x7->14x14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 14x14->28x28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 28x28->56x56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 56x56->112x112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # 112x112->224x224
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),  # 输出单通道
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.decoder(x)
        return x
