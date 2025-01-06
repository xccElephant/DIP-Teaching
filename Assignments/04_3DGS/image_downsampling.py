"""
读入一个文件夹下的所有图片，并进行下采样，设置下采样比例，保存到另一个文件夹
"""

import os
import cv2
import numpy as np


def downsample_images(input_dir, output_dir, downsample_factor):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录下的所有图片
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # 读取图片，保持透明通道
            image = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_UNCHANGED)
            
            # 如果是PNG图片且有透明通道
            if image.shape[-1] == 4:  # 检查是否有alpha通道
                # 分离颜色通道和透明通道
                bgr = image[:, :, :3]
                alpha = image[:, :, 3]
                
                # 对颜色通道进行下采样
                bgr_downsampled = cv2.resize(bgr, (0, 0), 
                                           fx=1/downsample_factor, 
                                           fy=1/downsample_factor)
                
                # 对透明通道进行下采样
                alpha_downsampled = cv2.resize(alpha, (0, 0), 
                                             fx=1/downsample_factor, 
                                             fy=1/downsample_factor)
                
                # 合并通道
                downsampled_image = np.dstack((bgr_downsampled, alpha_downsampled))
            else:
                # 对于没有透明通道的图片，直接下采样
                downsampled_image = cv2.resize(image, (0, 0), 
                                             fx=1/downsample_factor, 
                                             fy=1/downsample_factor)
            
            # 保存下采样后的图片
            cv2.imwrite(os.path.join(output_dir, filename), downsampled_image)

if __name__ == "__main__":
    input_dir = "data/chair/images"
    output_dir = "data/chair_downsampled/images"
    downsample_factor = 8
    downsample_images(input_dir, output_dir, downsample_factor)
