import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# 定义 ASPP 模块
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        # 1x1卷积分支
        self.atrous_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

        # 空洞率6的3x3卷积分支
        self.atrous_block6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

        # 空洞率12的3x3卷积分支
        self.atrous_block12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

        # 空洞率18的3x3卷积分支
        self.atrous_block18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

        # 全局平均池化分支
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

        # 特征融合层
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[2:]

        # 各分支并行处理
        out1 = self.atrous_block1(x)
        out2 = self.atrous_block6(x)
        out3 = self.atrous_block12(x)
        out4 = self.atrous_block18(x)

        # 全局池化分支（需要上采样到原始尺寸）
        out5 = self.global_avg_pool(x)
        out5 = nn.functional.interpolate(out5, size=size,
                                         mode='bilinear',
                                         align_corners=False)

        # 特征拼接与融合
        out = torch.cat([out1, out2, out3, out4, out5], dim=1)
        out = self.conv1(out)

        return out

# 显著性模型主结构
class SaliencyModel(nn.Module):
    def __init__(self, pretrained=True):
        super(SaliencyModel, self).__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # 编码器各层
        self.conv1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # [B,64,H/2,W/2]
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # [B,64,H/4,W/4]
        self.layer2 = backbone.layer2  # [B,128,H/8,W/8]
        self.layer3 = backbone.layer3  # [B,256,H/16,W/16]
        self.layer4 = backbone.layer4  # [B,512,H/32,W/32]

        # ASPP模块（替换原始的x5直连解码器）
        self.aspp = ASPP(in_channels=512, out_channels=256)

        # 解码器部分
        self.up4 = nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.conv_up4 = nn.Conv2d(256 + 256, 256, 3, 1, 1)

        self.up3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv_up3 = nn.Conv2d(128 + 128, 128, 3, 1, 1)

        self.up2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.conv_up2 = nn.Conv2d(64 + 64, 64, 3, 1, 1)

        self.up1 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.conv_up1 = nn.Conv2d(64 + 64, 64, 3, 1, 1)

        self.up0 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.conv_up0 = nn.Conv2d(32, 16, 3, 1, 1)

        self.final_conv = nn.Conv2d(16, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 编码器
        x1 = self.conv1(x)                        # [B,64,H/2,W/2]
        x2 = self.layer1(self.maxpool(x1))        # [B,64,H/4,W/4]
        x3 = self.layer2(x2)                      # [B,128,H/8,W/8]
        x4 = self.layer3(x3)                      # [B,256,H/16,W/16]
        x5 = self.layer4(x4)                      # [B,512,H/32,W/32]

        # ASPP模块提取多尺度特征
        x5_aspp = self.aspp(x5)                   # [B,256,H/32,W/32]

        # 解码器
        d4 = self.relu(self.up4(x5_aspp))         # [B,256,H/16,W/16]
        d4 = torch.cat([d4, x4], dim=1)           # [B,512,H/16,W/16]
        d4 = self.relu(self.conv_up4(d4))

        d3 = self.relu(self.up3(d4))              # [B,128,H/8,W/8]
        d3 = torch.cat([d3, x3], dim=1)           # [B,256,H/8,W/8]
        d3 = self.relu(self.conv_up3(d3))

        d2 = self.relu(self.up2(d3))              # [B,64,H/4,W/4]
        d2 = torch.cat([d2, x2], dim=1)           # [B,128,H/4,W/4]
        d2 = self.relu(self.conv_up2(d2))

        d1 = self.relu(self.up1(d2))              # [B,64,H/2,W/2]
        d1 = torch.cat([d1, x1], dim=1)           # [B,128,H/2,W/2]
        d1 = self.relu(self.conv_up1(d1))

        d0 = self.relu(self.up0(d1))              # [B,32,H,W]
        d0 = self.relu(self.conv_up0(d0))         # [B,16,H,W]

        out = self.final_conv(d0)                 # [B,1,H,W]
        out = self.sigmoid(out)
        return out
