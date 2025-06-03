import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# 边缘引导模块
class EdgeGuidedModule(nn.Module):
    def __init__(self, in_channels):
        super(EdgeGuidedModule, self).__init__()
        
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 3, padding=1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//4, 1, 1)
        )
        
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        edge_map = self.edge_conv(x)
        edge_weight = self.sigmoid(edge_map)
        enhanced_features = self.feature_enhance(x)
        output = x + edge_weight * enhanced_features
        return output, edge_map

# 多尺度上下文聚合模块
class ContextAggregationModule(nn.Module):
    def __init__(self, in_channels):
        super(ContextAggregationModule, self).__init__()
        
        # 全局上下文路径
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True)
        )
        
        # 中等上下文路径
        self.mid_pool = nn.AdaptiveAvgPool2d(2)
        self.mid_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True)
        )
        
        # 局部上下文路径
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True)
        )
        
        # 融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels//4 * 3, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch_size, channels, H, W = x.size()
        
        # 全局上下文
        global_feat = self.global_conv(self.global_pool(x))
        global_feat = global_feat.expand(-1, -1, H, W)
        
        # 中等上下文
        mid_feat = self.mid_conv(self.mid_pool(x))
        mid_feat = nn.functional.interpolate(mid_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        # 局部上下文
        local_feat = self.local_conv(x)
        
        # 融合多尺度上下文
        fused = torch.cat([global_feat, mid_feat, local_feat], dim=1)
        return self.fusion_conv(fused) + x  # 残差连接

# 细节增强模块
class DetailEnhancementModule(nn.Module):
    def __init__(self, in_channels):
        super(DetailEnhancementModule, self).__init__()
        
        # 局部特征提取
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 细节增强
        self.detail_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//4, in_channels, 1)
        )
        
        # 残差连接缩放因子
        self.residual_scale = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 提取局部特征
        local_features = self.local_conv(x)
        
        # 细节增强
        detail_features = self.detail_conv(local_features)
        
        # 注意力机制
        attention = self.sigmoid(detail_features)
        
        # 细节增强输出
        output = x + self.residual_scale * (attention * detail_features)
        return output

# 显著性模型主结构（添加细节增强模块）
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

        # 在深层特征后添加上下文聚合
        self.context_agg = ContextAggregationModule(in_channels=512)
        
        # 边缘引导模块
        self.edge_guided = EdgeGuidedModule(in_channels=512)
        
        # 在通道调整前添加细节增强
        self.detail_enhance = DetailEnhancementModule(in_channels=512)

        # 通道调整
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

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

        self.final_conv = nn.Conv2d(16, 1, 1)  # 显著图输出
        self.edge_conv = nn.Conv2d(16, 1, 1)   # 边缘图输出

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 编码器部分
        x1 = self.conv1(x)
        x2 = self.layer1(self.maxpool(x1))
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        
        # 添加多尺度上下文信息
        x5_context = self.context_agg(x5)
        
        # 应用边缘引导模块
        x5_guided, deep_edge = self.edge_guided(x5_context)
        
        # 应用细节增强
        x5_enhanced = self.detail_enhance(x5_guided)
        
        # 通道调整
        x5_reduced = self.channel_reduce(x5_enhanced)

        # 解码器部分
        d4 = self.relu(self.up4(x5_reduced))
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