import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        def discriminator_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        # 输入是连接的RGB图像和语义分割图像, 所以通道数是6
        self.model = nn.Sequential(
            *discriminator_block(6, 64, normalize=False),  # (6, 256, 256) -> (64, 128, 128)
            *discriminator_block(64, 128),    # -> (128, 64, 64)
            *discriminator_block(128, 256),   # -> (256, 32, 32)
            *discriminator_block(256, 512),   # -> (512, 16, 16)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),  # -> (1, 15, 15)
            nn.Sigmoid()
        )
        
    def forward(self, img_A, img_B):
        # img_A: 输入图像, img_B: 目标图像/生成图像
        combined_imgs = torch.cat([img_A, img_B], dim=1)
        return self.model(combined_imgs)
