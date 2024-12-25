import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyFrameAttention(nn.Module):
    """
    实现跨帧的时间注意力机制
    """
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
        # x shape: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        
        # 重塑输入以处理时间维度
        x = x.reshape(B, T * H * W, C)
        
        # 计算QKV
        qkv = self.qkv(x).reshape(B, T * H * W, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B, T * H * W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # 恢复原始形状
        x = x.reshape(B, T, H, W, C)
        
        return x

class CrossFrameAttention(nn.Module):
    """
    实现跨帧的交叉注意力机制，用于处理参考帧和目标帧之间的关系
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, ref):
        # x: target frame (B, H, W, C)
        # ref: reference frame (B, T, H, W, C)
        B, H, W, C = x.shape
        _, T, Hr, Wr, _ = ref.shape

        # 处理目标帧
        q = self.q(x).reshape(B, H * W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # 处理参考帧
        ref = ref.reshape(B, T * Hr * Wr, C)
        kv = self.kv(ref).reshape(B, T * Hr * Wr, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x 