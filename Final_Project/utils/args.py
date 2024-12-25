import argparse

def get_args():
    parser = argparse.ArgumentParser(description='SMITE: Video Segmentation with Diffusion Models')
    
    # 基本参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='设备选择')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True, help='数据集根目录')
    parser.add_argument('--category', type=str, default='all', help='训练类别')
    parser.add_argument('--num_frames', type=int, default=8, help='处理的视频帧数')
    parser.add_argument('--image_size', type=int, default=512, help='图像大小')
    
    # 模型参数
    parser.add_argument('--pretrained_model', type=str, default='stabilityai/stable-diffusion-2-1',
                      help='预训练扩散模型路径')
    parser.add_argument('--unet_additional_kwargs', type=dict, default={},
                      help='UNet额外参数')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--lambda_tracking', type=float, default=1.0, help='跟踪损失权重')
    parser.add_argument('--lambda_reg', type=float, default=0.1, help='正则化损失权重')
    
    # 时间一致性参数
    parser.add_argument('--window_size', type=int, default=5, help='时间窗口大小')
    parser.add_argument('--tracking_threshold', type=float, default=0.7, help='跟踪阈值')
    
    # 保存和加载参数
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='检查点目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    return args 