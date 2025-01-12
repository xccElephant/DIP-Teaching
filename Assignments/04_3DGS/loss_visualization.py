import pandas as pd
import matplotlib.pyplot as plt

# plt.style.use('seaborn-v0_8')  # 使用更现代的样式


# 读取CSV文件
df = pd.read_csv('data/chair/checkpoints/training_loss.csv')
df = df.dropna()

# 创建图表
plt.figure(figsize=(12, 6))
plt.plot(df['epoch'], df['loss'], 'b-', linewidth=2, alpha=0.7)

# 设置图表属性
plt.title('Training Loss Curve of Chair Dataset', fontsize=14, pad=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 设置坐标轴
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 添加边距
plt.margins(x=0.02)

# 保存图片
plt.savefig('assets/chair_training_loss.png', dpi=300, bbox_inches='tight')
plt.close()
