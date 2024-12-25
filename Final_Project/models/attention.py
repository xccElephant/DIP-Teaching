import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FullyFrameAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 (B, T, H, W, C)
            B: batch size, T: 时间维度, H: 高度, W: 宽度, C: 通道数
        """
        B, T, H, W, C = x.shape
        N = H * W
        
        # 重塑输入以处理所有帧
        x = x.reshape(B, T * N, C)
        
        # 计算QKV
        qkv = self.qkv(x).reshape(B, T * N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B, T * N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # 恢复原始形状
        x = x.reshape(B, T, H, W, C)
        return x

class TemporalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.temporal_attn = FullyFrameAttention(dim, num_heads)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        
        # 调整维度顺序以适应注意力层
        x = x.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
        
        # 应用时序自注意力
        identity = x
        x = self.norm(x)
        x = self.temporal_attn(x)
        x = x + identity
        
        # 恢复原始维度顺序
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        return x

class CrossFrameAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x, ref_x):
        """
        Args:
            x: 当前帧特征，形状为 (B, T, H, W, C)
            ref_x: 参考帧特征，形状为 (B, H, W, C)
        """
        B, T, H, W, C = x.shape
        
        # 生成Q、K、V
        q = self.to_q(x).reshape(B, T * H * W, self.num_heads, C // self.num_heads).transpose(1, 2)
        kv = self.to_kv(ref_x).reshape(B, H * W, 2, self.num_heads, C // self.num_heads)
        k, v = kv[..., 0, :, :], kv[..., 1, :, :]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B, T, H, W, C)
        x = self.proj(x)
        
        return x 