import torch
import torch.nn as nn
import copy

def inflate_kernel(kernel_2d, time_dim):
    """将2D卷积核膨胀为3D卷积核"""
    if len(kernel_2d.shape) == 4:
        out_c, in_c, h, w = kernel_2d.shape
        kernel_3d = torch.zeros((out_c, in_c, time_dim, h, w))
        # 在时间维度上平均分配权重
        kernel_3d[:, :, time_dim//2, :, :] = kernel_2d
        return kernel_3d
    else:
        return kernel_2d

def transfer_weights(model_2d, model_3d):
    """
    将2D UNet的权重转换并加载到3D UNet中
    Args:
        model_2d: 源2D UNet模型
        model_3d: 目标3D UNet模型
    """
    state_dict_2d = model_2d.state_dict()
    state_dict_3d = model_3d.state_dict()
    
    for k2d, k3d in zip(state_dict_2d.keys(), state_dict_3d.keys()):
        if "conv" in k2d and "weight" in k2d:
            # 对于卷积层权重进行膨胀
            state_dict_3d[k3d] = inflate_kernel(state_dict_2d[k2d], 
                                              time_dim=model_3d.time_dim)
        else:
            # 其他层直接复制
            state_dict_3d[k3d] = state_dict_2d[k2d]
    
    model_3d.load_state_dict(state_dict_3d)
    return model_3d

def verify_weight_transfer(model_2d, model_3d):
    """验证权重转换是否成功"""
    with torch.no_grad():
        # 创建测试输入
        x_2d = torch.randn(1, 3, 64, 64)
        x_3d = x_2d.unsqueeze(2).repeat(1, 1, model_3d.time_dim, 1, 1)
        
        # 获取输出
        out_2d = model_2d(x_2d)
        out_3d = model_3d(x_3d)
        
        # 验证中心帧的输出是否接近
        center_frame = out_3d[:, :, model_3d.time_dim//2, :, :]
        diff = torch.abs(out_2d - center_frame).mean()
        
        print(f"Average difference between 2D and 3D center frame: {diff.item()}")
        return diff.item() < 1e-6 