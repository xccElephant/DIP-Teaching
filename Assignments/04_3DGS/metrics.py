import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(img1, img2):
    """
    计算两张图片的 PSNR (Peak Signal-to-Noise Ratio)
    参数:
        img1, img2: numpy array, 范围 [0, 255] 或 [0, 1]
    返回:
        float: PSNR 值
    """
    # 确保图像范围在 [0, 255]
    if img1.max() <= 1.0:
        img1 = img1 * 255.0
    if img2.max() <= 1.0:
        img2 = img2 * 255.0
        
    # 明确指定数据范围为 255
    return peak_signal_noise_ratio(img1, img2, data_range=255)

def calculate_ssim(img1, img2):
    """
    计算两张图片的 SSIM (Structural Similarity Index)
    """
    # 确保图像范围在 [0, 255]
    if img1.max() <= 1.0:
        img1 = img1 * 255.0
    if img2.max() <= 1.0:
        img2 = img2 * 255.0
    
    return ssim(img1, img2, channel_axis=2)  # 对于 RGB 图像，指定 channel_axis

def calculate_lpips(img1, img2, device='cuda'):
    """
    计算两张图片的 LPIPS (Learned Perceptual Image Patch Similarity)
    参数:
        img1, img2: numpy array 或 torch.Tensor, 范围 [0, 1]
        device: 使用的设备 ('cuda' 或 'cpu')
    返回:
        float: LPIPS 值
    """
    # 初始化 LPIPS 模型
    loss_fn = lpips.LPIPS(net='alex', verbose=False).to(device)
    
    # 转换为 torch.Tensor
    if not isinstance(img1, torch.Tensor):
        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()
    
    # 确保通道顺序正确 (H,W,C) -> (C,H,W)
    if img1.shape[-1] == 3:  # 如果通道在最后一维
        img1 = img1.permute(2, 0, 1)
        img2 = img2.permute(2, 0, 1)
    
    # 确保尺寸正确 (N,C,H,W)
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    # 移动到指定设备
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    with torch.no_grad():
        lpips_value = loss_fn(img1, img2).item()
    
    return lpips_value


if __name__ == "__main__":
    # 读取图片
    # img = Image.open('data/chair/checkpoints/debug_images/epoch_0199/r_31.png')
    # img = np.array(img)
    # # 分为左右两半
    # img1 = img[:, :img.shape[1]//2, :]
    # img2 = img[:, img.shape[1]//2:, :]

    img1 = Image.open('assets/gt_0.png')
    img1 = np.array(img1)
    img2 = Image.open('assets/render_0.png')
    img2 = np.array(img2)
    print("psnr:", calculate_psnr(img1, img2))
    print("ssim:", calculate_ssim(img1, img2))
    print("lpips:", calculate_lpips(img1, img2))