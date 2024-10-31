import os

import matplotlib.pyplot as plt
import pandas as pd

dataset_names = ['facades', 'cityscapes']
current_path = os.path.dirname(os.path.abspath(__file__))


for dataset_name in dataset_names:
    # 读取损失数据
    data = pd.read_csv(os.path.join(current_path, "logs", f'loss_history_{dataset_name}.txt'), sep='\t')

    # 创建图表
    plt.figure(figsize=(12, 6))
    plt.plot(data['Epoch'], data['Train Loss'], label='Train Loss')
    plt.plot(data['Epoch'], data['Validation Loss'], label='Validation Loss')

    # 设置图表标题和标签
    plt.title(f'Train and Validation Loss Curve of {dataset_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 显示网格
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图表
    plt.savefig(os.path.join(current_path, "pix2pix_results", f'loss_curve_{dataset_name}.png'), dpi=300, bbox_inches='tight')

    # 显示图表
    # plt.show()