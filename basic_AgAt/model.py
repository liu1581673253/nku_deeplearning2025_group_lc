import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from agent_attention import AgentAttention

# 轻量级SE模块
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 显著性模型主结构
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

        # 在ResNet的每个残差块后添加SE模块
        self.se1 = SEBlock(64)  # 对应conv1输出
        self.se2 = SEBlock(64)  # 对应layer1输出
        self.se3 = SEBlock(128)  # 对应layer2输出
        self.se4 = SEBlock(256)  # 对应layer3输出
        self.se5 = SEBlock(512)  # 对应layer4输出

        # 添加AgentAttention模块
        self.agent_attn = AgentAttention(dim=512, window_size=[7, 7], num_heads=8, agent_num=49)

        self.up4 = nn.ConvTranspose2d(512, 256, 4, 2, 1)  # 修改输入通道为512
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
        self.edge_conv = nn.Conv2d(16, 1, 1)  # 边缘图输出

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 编码器部分
        x1 = self.conv1(x)
        x1 = self.se1(x1)  # 添加SE

        x2 = self.layer1(self.maxpool(x1))
        x2 = self.se2(x2)  # 添加SE

        x3 = self.layer2(x2)
        x3 = self.se3(x3)  # 添加SE

        x4 = self.layer3(x3)
        x4 = self.se4(x4)  # 添加SE

        x5 = self.layer4(x4)
        x5 = self.se5(x5)  # 添加SE

        # 使用AgentAttention替代ASPP
        x5_attn = self.agent_attn(x5)

        # 解码器部分
        d4 = self.relu(self.up4(x5_attn))  # 使用注意力后的特征
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.relu(self.conv_up4(d4))

        d3 = self.relu(self.up3(d4))
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.relu(self.conv_up3(d3))

        d2 = self.relu(self.up2(d3))
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.relu(self.conv_up2(d2))

        d1 = self.relu(self.up1(d2))
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.relu(self.conv_up1(d1))

        d0 = self.relu(self.up0(d1))
        d0 = self.relu(self.conv_up0(d0))

        # 输出头
        saliency_out = self.final_conv(d0)
        saliency_out = self.sigmoid(saliency_out)

        edge_out = self.edge_conv(d0)
        edge_out = self.sigmoid(edge_out)

        return saliency_out, edge_out