import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import TemporalSelfAttention, CrossFrameAttention

class InflatedConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (1, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (1, stride, stride)
        if isinstance(padding, int):
            padding = (0, padding, padding)
            
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding)
        
    def forward(self, x):
        return self.conv(x)

class TemporalBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.temporal_attn = TemporalSelfAttention(dim)
        self.norm = nn.GroupNorm(8, dim)
        
    def forward(self, x):
        return self.temporal_attn(self.norm(x))

class InflatedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, time_dim=8, features=[64, 128, 256, 512]):
        super().__init__()
        self.time_dim = time_dim
        
        # 下采样路径
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # 初始卷积
        self.first = InflatedConv3d(in_channels, features[0], kernel_size=3, padding=1)
        
        # 下采样块
        in_features = features[0]
        for feature in features[1:]:
            self.downs.append(
                nn.Sequential(
                    InflatedConv3d(in_features, feature, kernel_size=3, padding=1),
                    nn.GroupNorm(8, feature),
                    nn.ReLU(inplace=True),
                    TemporalBlock(feature),
                    InflatedConv3d(feature, feature, kernel_size=3, padding=1),
                    nn.GroupNorm(8, feature),
                    nn.ReLU(inplace=True),
                )
            )
            in_features = feature
            
        # 上采样路径
        self.ups = nn.ModuleList()
        for feature in reversed(features[:]):
            self.ups.append(
                nn.Sequential(
                    InflatedConv3d(feature * 2, feature, kernel_size=3, padding=1),
                    nn.GroupNorm(8, feature),
                    nn.ReLU(inplace=True),
                    TemporalBlock(feature),
                    InflatedConv3d(feature, feature, kernel_size=3, padding=1),
                    nn.GroupNorm(8, feature),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose3d(feature, feature // 2, 
                                     kernel_size=(1, 2, 2), 
                                     stride=(1, 2, 2))
                )
            )
            
        # 最终卷积
        self.final_conv = nn.Sequential(
            InflatedConv3d(features[0], features[0] // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, features[0] // 2),
            nn.ReLU(inplace=True),
            InflatedConv3d(features[0] // 2, out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        # x shape: (B, C, T, H, W)
        
        # 初始特征
        x = self.first(x)
        
        # 存储skip connections
        skip_connections = []
        
        # 下采样路径
        for down in self.downs:
            skip_connections.append(x)
            x = self.pool(x)
            x = down(x)
            
        # 上采样路径
        skip_connections = skip_connections[::-1]  # 反转列表
        
        for idx, up in enumerate(self.ups):
            skip = skip_connections[idx]
            x = torch.cat((skip, x), dim=1)
            x = up(x)
            
        # 最终卷积
        return self.final_conv(x)

class SMITEUNet(nn.Module):
    def __init__(self, pretrained_model=None, **kwargs):
        super().__init__()
        self.unet = InflatedUNet(**kwargs)
        
        if pretrained_model is not None:
            from ..utils.transfer_weights import transfer_weights
            self.unet = transfer_weights(pretrained_model, self.unet)
            
    def forward(self, x):
        return self.unet(x) 