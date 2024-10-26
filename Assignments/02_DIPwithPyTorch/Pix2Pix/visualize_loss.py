import matplotlib.pyplot as plt
import pandas as pd

# 读取损失数据
data = pd.read_csv('loss_history.txt', sep='\t')

# 创建图表
plt.figure(figsize=(12, 6))
plt.plot(data['Epoch'], data['Train Loss'], label='Train Loss')
plt.plot(data['Epoch'], data['Validation Loss'], label='Validation Loss')

# 设置图表标题和标签
plt.title('Train and Validation Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 显示网格
plt.grid(True, linestyle='--', alpha=0.7)

# 保存图表
plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()