import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# 显著性模型主结构（已移除 ASPP，直接连接编码器输出）
class SaliencyModel(nn.Module):
    def __init__(self, pretrained=True):
        super(SaliencyModel, self).__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        self.conv1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # 编码器输出通道为 512
        self.up4 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.conv_up4 = nn.Conv2d(256 + 256, 256, 3, 1, 1)

        self.up3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv_up3 = nn.Conv2d(128 + 128, 128, 3, 1, 1)

        self.up2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.conv_up2 = nn.Conv2d(64 + 64, 64, 3, 1, 1)

        self.up1 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.conv_up1 = nn.Conv2d(64 + 64, 64, 3, 1, 1)

        self.up0 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.conv_up0 = nn.Conv2d(32, 16, 3, 1, 1)

        self.final_conv = nn.Conv2d(16, 1, 1)  # 显著图输出
        self.edge_conv = nn.Conv2d(16, 1, 1)   # 边缘图输出

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)                    # [B, 64, H/2, W/2]
        x2 = self.layer1(self.maxpool(x1))    # [B, 64, H/4, W/4]
        x3 = self.layer2(x2)                  # [B, 128, H/8, W/8]
        x4 = self.layer3(x3)                  # [B, 256, H/16, W/16]
        x5 = self.layer4(x4)                  # [B, 512, H/32, W/32]

        d4 = self.relu(self.up4(x5))          # [B, 256, H/16, W/16]
        d4 = torch.cat([d4, x4], dim=1)       # [B, 512, H/16, W/16]
        d4 = self.relu(self.conv_up4(d4))     # [B, 256, H/16, W/16]

        d3 = self.relu(self.up3(d4))          # [B, 128, H/8, W/8]
        d3 = torch.cat([d3, x3], dim=1)       # [B, 256, H/8, W/8]
        d3 = self.relu(self.conv_up3(d3))     # [B, 128, H/8, W/8]

        d2 = self.relu(self.up2(d3))          # [B, 64, H/4, W/4]
        d2 = torch.cat([d2, x2], dim=1)       # [B, 128, H/4, W/4]
        d2 = self.relu(self.conv_up2(d2))     # [B, 64, H/4, W/4]

        d1 = self.relu(self.up1(d2))          # [B, 64, H/2, W/2]
        d1 = torch.cat([d1, x1], dim=1)       # [B, 128, H/2, W/2]
        d1 = self.relu(self.conv_up1(d1))     # [B, 64, H/2, W/2]

        d0 = self.relu(self.up0(d1))          # [B, 32, H, W]
        d0 = self.relu(self.conv_up0(d0))     # [B, 16, H, W]

        saliency_out = self.final_conv(d0)    # [B, 1, H, W]
        saliency_out = self.sigmoid(saliency_out)

        edge_out = self.edge_conv(d0)         # [B, 1, H, W]
        edge_out = self.sigmoid(edge_out)

        return saliency_out, edge_out
