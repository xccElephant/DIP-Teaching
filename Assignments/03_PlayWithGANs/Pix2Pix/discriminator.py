import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        def discriminator_block(in_channels, out_channels, normalize=True, kernel_size=4):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        # input is concatenated RGB image and semantic segmentation image, so the number of channels is 6
        self.model = nn.Sequential(
            *discriminator_block(6, 64, normalize=False),  # (6, 256, 256) -> (64, 128, 128)
            *discriminator_block(64, 128),    # -> (128, 64, 64)
            *discriminator_block(128, 256),   # -> (256, 32, 32)
            *discriminator_block(256, 512),   # -> (512, 16, 16)
            
            # more layers
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # -> (512, 16, 16)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),  # -> (1, 15, 15)
            nn.Sigmoid()
        )
        
        # spectral normalization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.utils.spectral_norm(m)
        
    def forward(self, img_A, img_B):
        # img_A: semantic segmentation image, img_B: target image or generated image
        combined_imgs = torch.cat([img_A, img_B], dim=1)  # (batch_size, 6, 256, 256)
        return self.model(combined_imgs)
