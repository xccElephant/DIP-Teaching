import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        # Encoder (Convolutional Layers)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # (3, 256, 256) -> (64, 128, 128)
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (64, 128, 128) -> (128, 64, 64)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (128, 64, 64) -> (256, 32, 32)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # (256, 32, 32) -> (512, 16, 16)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Residual Blocks
        self.resblocks = nn.Sequential(   # (512, 16, 16) -> (512, 16, 16)
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512)
        )
        
        # Attention Mechanism
        self.attention = SelfAttention(512)  # (512, 16, 16) -> (512, 16, 16)
        
        # Decoder (Deconvolutional Layers)
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # (512, 16, 16) -> (512, 32, 32)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)  # Dropout layer to prevent overfitting
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=4, stride=2, padding=1),  # (768, 32, 32) -> (256, 64, 64)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(384, 128, kernel_size=4, stride=2, padding=1),  # (384, 64, 64) -> (128, 128, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, kernel_size=4, stride=2, padding=1),  # (192, 128, 128) -> (64, 256, 256)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # (64, 256, 256) -> (3, 256, 256)
            nn.Tanh()
        )


    def forward(self, x):
        # Encoder forward pass
        x1 = self.encoder1(x)  # (64, 128, 128)
        x2 = self.encoder2(x1)  # (128, 64, 64)
        x3 = self.encoder3(x2)  # (256, 32, 32)
        x4 = self.encoder4(x3)  # (512, 16, 16)

        # Middle layers
        x5 = self.resblocks(x4)  # (512, 16, 16)
        x5 = self.attention(x5)  # (512, 16, 16)

        # Decoder forward pass
        x = self.decoder1(x5)  # (512, 32, 32)
        x = torch.cat([x, x3], dim=1)  # skip connection, (768, 32, 32)
        x = self.decoder2(x)  # (256, 64, 64)
        x = torch.cat([x, x2], dim=1)  # skip connection, (384, 64, 64)
        x = self.decoder3(x)  # (128, 128, 128)
        x = torch.cat([x, x1], dim=1)  # skip connection, (192, 128, 128)
        x = self.decoder4(x)  # (64, 256, 256)
        x = self.decoder5(x)  # (3, 256, 256)
        
        return x

    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.block(x)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        q = self.query(x).view(batch_size, -1, H*W).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, H*W)
        v = self.value(x).view(batch_size, -1, H*W)
        
        attention = torch.bmm(q, k)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        return self.gamma * out + x