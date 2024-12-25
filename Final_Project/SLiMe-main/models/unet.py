import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import FullyFrameAttention, CrossFrameAttention

class InflatedConv3d(nn.Module):
    """
    3D 膨胀卷积层，用于处理时序信息
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, 
            out_channels,
            kernel_size=(3, kernel_size, kernel_size),
            stride=(1, stride, stride),
            padding=(1, padding, padding)
        )
    
    def forward(self, x):
        return self.conv(x)

class TemporalBlock(nn.Module):
    """
    时序处理块，包含膨胀卷积和注意力机制
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.conv = InflatedConv3d(dim, dim)
        self.norm1 = nn.GroupNorm(8, dim)
        self.attn = FullyFrameAttention(dim, num_heads=num_heads)
        self.norm2 = nn.GroupNorm(8, dim)
        
    def forward(self, x):
        # x shape: (B, C, T, H, W)
        identity = x
        
        # 卷积分支
        x = self.conv(x)
        x = self.norm1(x)
        
        # 注意力分支
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
        x = self.attn(x)
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        x = self.norm2(x)
        
        return x + identity

class InflatedUNet(nn.Module):
    """
    膨胀 UNet 主体结构
    """
    def __init__(
        self,
        in_channels=4,
        model_channels=320,
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions=(4, 2, 1),
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        num_heads=8,
        use_temporal_attention=True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.use_temporal_attention = use_temporal_attention

        # 输入投影
        self.input_proj = InflatedConv3d(in_channels, model_channels)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        channels = [model_channels]
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    TemporalBlock(
                        channels[-1],
                        num_heads=num_heads if level in attention_resolutions else 0
                    )
                ]
                channels.append(mult * model_channels)
                self.down_blocks.append(nn.ModuleList(layers))
            
            if level != len(channel_mult) - 1:
                self.down_blocks.append(
                    InflatedConv3d(channels[-1], channels[-1], stride=2)
                )
                channels.append(channels[-1])
        
        # 中间块
        self.middle_block = TemporalBlock(channels[-1], num_heads=num_heads)
        
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                layers = [
                    TemporalBlock(
                        channels.pop() + channels[-1],
                        num_heads=num_heads if level in attention_resolutions else 0
                    )
                ]
                channels.append(mult * model_channels)
                self.up_blocks.append(nn.ModuleList(layers))
                
            if level != 0:
                self.up_blocks.append(
                    nn.ConvTranspose3d(
                        channels[-1], channels[-1],
                        kernel_size=4, stride=2, padding=1
                    )
                )
                channels.append(channels[-1])
        
        # 输出投影
        self.out = nn.Sequential(
            nn.GroupNorm(8, channels[-1]),
            nn.SiLU(),
            InflatedConv3d(channels[-1], out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, timesteps=None):
        # x shape: (B, C, T, H, W)
        hs = []
        
        # 输入投影
        h = self.input_proj(x)
        hs.append(h)
        
        # 下采样路径
        for module in self.down_blocks:
            if isinstance(module, InflatedConv3d):
                h = module(h)
            else:
                for layer in module:
                    h = layer(h)
            hs.append(h)
        
        # 中间块
        h = self.middle_block(h)
        
        # 上采样路径
        for module in self.up_blocks:
            if isinstance(module, nn.ConvTranspose3d):
                h = module(h)
            else:
                h = torch.cat([h, hs.pop()], dim=1)
                for layer in module:
                    h = layer(h)
        
        # 输出投影
        return self.out(h) 