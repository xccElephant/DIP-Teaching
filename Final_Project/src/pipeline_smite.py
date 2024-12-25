import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from diffusers import StableDiffusionPipeline

from ..models.unet import SMITEUNet
from .tracking import TrackingModule
from .frequency_filter import FrequencyFilter

class SMITEPipeline:
    def __init__(self, 
                 pretrained_model_path: str,
                 device: str = 'cuda',
                 **kwargs):
        """
        初始化SMITE pipeline
        Args:
            pretrained_model_path: 预训练模型路径
            device: 设备
            **kwargs: 其他参数
        """
        self.device = device
        
        # 加载预训练的Stable Diffusion
        self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_path
        ).to(device)
        
        # 初始化SMITE UNet
        self.unet = SMITEUNet(
            pretrained_model=self.sd_pipeline.unet,
            **kwargs
        ).to(device)
        
        # 初始化跟踪模块和频率滤波器
        self.tracker = TrackingModule(**kwargs)
        self.freq_filter = FrequencyFilter(**kwargs)
        
        # 优化器
        self.optimizer = None
        
    def prepare_latents(self, 
                       video: torch.Tensor,
                       reference_images: List[torch.Tensor],
                       reference_masks: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        准备潜在表示
        Args:
            video: 视频张量，形状为 (B, C, T, H, W)
            reference_images: 参考图像列表
            reference_masks: 参考mask列表
        Returns:
            包含潜在表示的字典
        """
        # 编码视频帧
        video_latents = self.sd_pipeline.vae.encode(
            video.reshape(-1, *video.shape[2:])
        ).latent_dist.sample()
        video_latents = video_latents.reshape(
            video.shape[0], -1, *video_latents.shape[1:]
        )
        
        # 编码参考图像
        ref_latents = []
        ref_mask_latents = []
        for img, mask in zip(reference_images, reference_masks):
            # 编码图像
            img_latent = self.sd_pipeline.vae.encode(img).latent_dist.sample()
            ref_latents.append(img_latent)
            
            # 下采样mask以匹配潜在空间大小
            mask_latent = F.interpolate(
                mask.unsqueeze(1),
                size=img_latent.shape[-2:],
                mode='nearest'
            )
            ref_mask_latents.append(mask_latent)
            
        ref_latents = torch.stack(ref_latents)
        ref_mask_latents = torch.stack(ref_mask_latents)
        
        return {
            'video_latents': video_latents,
            'ref_latents': ref_latents,
            'ref_mask_latents': ref_mask_latents
        }
        
    def optimize_embeddings(self, 
                          latents: Dict[str, torch.Tensor],
                          num_steps: int = 100) -> None:
        """
        优化文本嵌入
        Args:
            latents: 潜在表示字典
            num_steps: 优化步数
        """
        # 初始化优化器
        text_embeddings = self.sd_pipeline.text_encoder.get_input_embeddings()
        self.optimizer = torch.optim.Adam([text_embeddings], lr=1e-5)
        
        for step in range(num_steps):
            self.optimizer.zero_grad()
            
            # 前向传播
            pred_masks = self.unet(latents['ref_latents'])
            
            # 计算损失
            loss = F.binary_cross_entropy_with_logits(
                pred_masks, latents['ref_mask_latents']
            )
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
    def optimize_attention(self, 
                         latents: Dict[str, torch.Tensor],
                         num_steps: int = 100) -> None:
        """
        优化交叉注意力层
        Args:
            latents: 潜在表示字典
            num_steps: 优化步数
        """
        # 解冻交叉注意力层
        for name, param in self.unet.named_parameters():
            if 'attn' in name:
                param.requires_grad = True
                
        # 初始化优化器
        self.optimizer = torch.optim.Adam(
            [p for p in self.unet.parameters() if p.requires_grad],
            lr=1e-5
        )
        
        for step in range(num_steps):
            self.optimizer.zero_grad()
            
            # 前向传播
            pred_masks = self.unet(latents['ref_latents'])
            
            # 计算损失
            loss = F.binary_cross_entropy_with_logits(
                pred_masks, latents['ref_mask_latents']
            )
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
    def apply_temporal_guidance(self, 
                              video_latents: torch.Tensor,
                              window_size: int = 5) -> torch.Tensor:
        """
        应用时间引导
        Args:
            video_latents: 视频潜在表示
            window_size: 时间窗口大小
        Returns:
            优化后的视频潜在表示
        """
        B, T = video_latents.shape[:2]
        device = video_latents.device
        
        # 初始化跟踪器
        self.tracker.initialize_tracker(video_latents)
        
        # 获取关键点
        keypoints = self.get_keypoints(video_latents[:, 0])  # 使用第一帧
        
        # 跟踪关键点
        trajectories = self.tracker.track_points(keypoints)
        
        # 生成跟踪mask
        tracking_masks = self.tracker.get_tracking_mask(trajectories)
        
        # 应用频率滤波
        filtered_latents = self.freq_filter.apply_filter(video_latents)
        
        # 计算总损失
        tracking_loss = F.mse_loss(
            video_latents * tracking_masks.unsqueeze(1),
            filtered_latents * tracking_masks.unsqueeze(1)
        )
        
        freq_loss = self.freq_filter.compute_frequency_loss(
            video_latents, filtered_latents
        )
        
        total_loss = tracking_loss + 0.1 * freq_loss
        
        # 优化潜在表示
        video_latents.requires_grad = True
        optimizer = torch.optim.Adam([video_latents], lr=1e-4)
        
        for _ in range(50):  # 优化步数
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
        return video_latents.detach()
    
    def get_keypoints(self, latents: torch.Tensor, 
                     num_points: int = 100) -> torch.Tensor:
        """
        从潜在表示中提取关键点
        Args:
            latents: 潜在表示
            num_points: 关键点数量
        Returns:
            关键点坐标
        """
        B, C, H, W = latents.shape
        device = latents.device
        
        # 计算梯度幅值
        grad_x = F.conv2d(latents, torch.tensor([[[-1, 0, 1]]], 
                         device=device).float().unsqueeze(0))
        grad_y = F.conv2d(latents, torch.tensor([[[-1], [0], [1]]], 
                         device=device).float().unsqueeze(0))
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        
        # 选择梯度最大的点作为关键点
        grad_mag = grad_mag.reshape(B, -1)
        _, indices = torch.topk(grad_mag, num_points, dim=1)
        
        # 转换为坐标
        y = indices // W
        x = indices % W
        
        return torch.stack([x, y], dim=-1).float()
    
    def __call__(self,
                 video: torch.Tensor,
                 reference_images: List[torch.Tensor],
                 reference_masks: List[torch.Tensor]) -> torch.Tensor:
        """
        执行完整的SMITE pipeline
        Args:
            video: 输入视频
            reference_images: 参考图像列表
            reference_masks: 参考mask列表
        Returns:
            视频分割mask
        """
        # 准备潜在表示
        latents = self.prepare_latents(video, reference_images, reference_masks)
        
        # 优化文本嵌入
        self.optimize_embeddings(latents)
        
        # 优化交叉注意力
        self.optimize_attention(latents)
        
        # 应用时间引导
        guided_latents = self.apply_temporal_guidance(latents['video_latents'])
        
        # 生成最终mask
        final_masks = self.unet(guided_latents)
        
        return torch.sigmoid(final_masks) 