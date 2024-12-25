import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

class FrequencyFilter:
    def __init__(self, cutoff_freq: float = 0.5):
        """
        初始化频率滤波器
        Args:
            cutoff_freq: 截止频率，范围[0,1]
        """
        self.cutoff_freq = cutoff_freq
        
    def create_lowpass_filter(self, size: Tuple[int, ...]) -> torch.Tensor:
        """
        创建低通滤波器
        Args:
            size: 滤波器大小 (T, H, W)
        Returns:
            低通滤波器mask
        """
        T, H, W = size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建频率网格
        freq_t = torch.fft.fftfreq(T, device=device)
        freq_h = torch.fft.fftfreq(H, device=device)
        freq_w = torch.fft.fftfreq(W, device=device)
        
        # 创建3D频率网格
        freq_grid_t, freq_grid_h, freq_grid_w = torch.meshgrid(
            freq_t, freq_h, freq_w, indexing='ij'
        )
        
        # 计算频率距离
        freq_distance = torch.sqrt(
            freq_grid_t**2 + freq_grid_h**2 + freq_grid_w**2
        )
        
        # 创建理想低通滤波器
        return (freq_distance <= self.cutoff_freq).float()
    
    def dct_3d(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算3D DCT变换
        Args:
            x: 输入张量，形状为 (B, C, T, H, W)
        Returns:
            DCT系数
        """
        B, C = x.shape[:2]
        
        # 对每个维度依次进行DCT
        x = torch.transpose(x, 2, 4)  # B, C, W, H, T
        x = torch.cos(
            torch.pi / (2 * x.size(-1)) * 
            torch.arange(x.size(-1), device=x.device)[None, :] * 
            (2 * torch.arange(x.size(-1), device=x.device)[:, None] + 1)
        ) @ x
        
        x = torch.transpose(x, 2, 3)  # B, C, H, W, T
        x = torch.cos(
            torch.pi / (2 * x.size(-2)) * 
            torch.arange(x.size(-2), device=x.device)[None, :] * 
            (2 * torch.arange(x.size(-2), device=x.device)[:, None] + 1)
        ) @ x
        
        x = torch.transpose(x, 3, 4)  # B, C, H, T, W
        x = torch.cos(
            torch.pi / (2 * x.size(-1)) * 
            torch.arange(x.size(-1), device=x.device)[None, :] * 
            (2 * torch.arange(x.size(-1), device=x.device)[:, None] + 1)
        ) @ x
        
        # 恢复原始维度顺序
        x = torch.transpose(x, 2, 4)  # B, C, W, T, H
        x = torch.transpose(x, 3, 4)  # B, C, W, H, T
        x = torch.transpose(x, 2, 4)  # B, C, T, H, W
        
        return x
    
    def idct_3d(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算3D IDCT变换
        Args:
            x: DCT系数，形状为 (B, C, T, H, W)
        Returns:
            重建信号
        """
        B, C = x.shape[:2]
        
        # 对每个维度依次进行IDCT
        x = torch.transpose(x, 2, 4)  # B, C, W, H, T
        x = torch.cos(
            torch.pi / (2 * x.size(-1)) * 
            torch.arange(x.size(-1), device=x.device)[:, None] * 
            (2 * torch.arange(x.size(-1), device=x.device)[None, :] + 1)
        ) @ x
        
        x = torch.transpose(x, 2, 3)  # B, C, H, W, T
        x = torch.cos(
            torch.pi / (2 * x.size(-2)) * 
            torch.arange(x.size(-2), device=x.device)[:, None] * 
            (2 * torch.arange(x.size(-2), device=x.device)[None, :] + 1)
        ) @ x
        
        x = torch.transpose(x, 3, 4)  # B, C, H, T, W
        x = torch.cos(
            torch.pi / (2 * x.size(-1)) * 
            torch.arange(x.size(-1), device=x.device)[:, None] * 
            (2 * torch.arange(x.size(-1), device=x.device)[None, :] + 1)
        ) @ x
        
        # 恢复原始维度顺序
        x = torch.transpose(x, 2, 4)  # B, C, W, T, H
        x = torch.transpose(x, 3, 4)  # B, C, W, H, T
        x = torch.transpose(x, 2, 4)  # B, C, T, H, W
        
        return x
    
    def apply_filter(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用低通滤波
        Args:
            x: 输入张量，形状为 (B, C, T, H, W)
        Returns:
            滤波后的张量
        """
        # 计算DCT
        x_dct = self.dct_3d(x)
        
        # 创建并应用低通滤波器
        filter_mask = self.create_lowpass_filter(x.shape[2:])
        x_dct = x_dct * filter_mask[None, None, ...]
        
        # 计算IDCT
        return self.idct_3d(x_dct)
    
    def compute_frequency_loss(self, pred: torch.Tensor, 
                             target: torch.Tensor) -> torch.Tensor:
        """
        计算频域损失
        Args:
            pred: 预测张量，形状为 (B, C, T, H, W)
            target: 目标张量，形状为 (B, C, T, H, W)
        Returns:
            频域损失值
        """
        # 计算DCT
        pred_dct = self.dct_3d(pred)
        target_dct = self.dct_3d(target)
        
        # 创建低通滤波器
        filter_mask = self.create_lowpass_filter(pred.shape[2:])
        
        # 计算低频成分的L1损失
        loss = torch.abs(
            pred_dct * filter_mask[None, None, ...] - 
            target_dct * filter_mask[None, None, ...]
        ).mean()
        
        return loss 