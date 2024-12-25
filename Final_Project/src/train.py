import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import logging
from typing import Dict, List, Optional

from .pipeline_smite import SMITEPipeline

class Trainer:
    def __init__(self,
                 pipeline: SMITEPipeline,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 device: str = 'cuda',
                 **kwargs):
        """
        初始化训练器
        Args:
            pipeline: SMITE pipeline
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 设备
            **kwargs: 其他参数
        """
        self.pipeline = pipeline
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 训练配置
        self.num_epochs = kwargs.get('num_epochs', 100)
        self.save_dir = kwargs.get('save_dir', 'checkpoints')
        self.save_freq = kwargs.get('save_freq', 10)
        
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch
        Returns:
            包含损失值的字典
        """
        self.pipeline.unet.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        for batch in tqdm(self.train_loader, desc='Training'):
            # 解包数据
            video = batch['video'].to(self.device)
            ref_images = [img.to(self.device) for img in batch['reference_images']]
            ref_masks = [mask.to(self.device) for mask in batch['reference_masks']]
            target_masks = batch['target_masks'].to(self.device)
            
            # 准备潜在表示
            latents = self.pipeline.prepare_latents(video, ref_images, ref_masks)
            
            # 优化文本嵌入
            self.pipeline.optimize_embeddings(latents)
            
            # 优化交叉注意力
            self.pipeline.optimize_attention(latents)
            
            # 应用时间引导
            guided_latents = self.pipeline.apply_temporal_guidance(
                latents['video_latents']
            )
            
            # 生成预测mask
            pred_masks = self.pipeline.unet(guided_latents)
            
            # 计算损失
            loss = F.binary_cross_entropy_with_logits(pred_masks, target_masks)
            
            # 反向传播
            loss.backward()
            self.pipeline.optimizer.step()
            self.pipeline.optimizer.zero_grad()
            
            total_loss += loss.item()
            
        return {'loss': total_loss / num_batches}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        在验证集上评估模型
        Returns:
            包含评估指标的字典
        """
        if self.val_loader is None:
            return {}
            
        self.pipeline.unet.eval()
        total_iou = 0
        total_f1 = 0
        num_batches = len(self.val_loader)
        
        for batch in tqdm(self.val_loader, desc='Validating'):
            # 解包数据
            video = batch['video'].to(self.device)
            ref_images = [img.to(self.device) for img in batch['reference_images']]
            ref_masks = [mask.to(self.device) for mask in batch['reference_masks']]
            target_masks = batch['target_masks'].to(self.device)
            
            # 生成预测
            pred_masks = self.pipeline(video, ref_images, ref_masks)
            
            # 计算指标
            pred_binary = (pred_masks > 0.5).float()
            intersection = (pred_binary * target_masks).sum((2, 3, 4))
            union = (pred_binary + target_masks).clamp(0, 1).sum((2, 3, 4))
            iou = (intersection + 1e-6) / (union + 1e-6)
            
            # F1 score
            precision = intersection / (pred_binary.sum((2, 3, 4)) + 1e-6)
            recall = intersection / (target_masks.sum((2, 3, 4)) + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            
            total_iou += iou.mean().item()
            total_f1 += f1.mean().item()
            
        return {
            'val_iou': total_iou / num_batches,
            'val_f1': total_f1 / num_batches
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        保存检查点
        Args:
            epoch: 当前epoch
            metrics: 评估指标
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.pipeline.unet.state_dict(),
            'optimizer_state_dict': self.pipeline.optimizer.state_dict(),
            'metrics': metrics
        }
        
        save_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, save_path)
        self.logger.info(f'Saved checkpoint to {save_path}')
        
    def train(self) -> None:
        """
        执行完整的训练流程
        """
        best_val_iou = 0
        
        for epoch in range(self.num_epochs):
            self.logger.info(f'Epoch {epoch+1}/{self.num_epochs}')
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.validate()
            
            # 记录指标
            metrics = {**train_metrics, **val_metrics}
            metric_str = ' | '.join(f'{k}: {v:.4f}' for k, v in metrics.items())
            self.logger.info(f'Metrics: {metric_str}')
            
            # 保存最佳模型
            if val_metrics.get('val_iou', 0) > best_val_iou:
                best_val_iou = val_metrics['val_iou']
                self.save_checkpoint(epoch, metrics)
                
            # 定期保存检查点
            if (epoch + 1) % self.save_freq == 0:
                self.save_checkpoint(epoch, metrics) 