import matplotlib.pyplot as plt
import csv
import os

# 初始化数据列表
x = []
y = []

# 读取CSV文件
with open(os.path.join(os.path.dirname(__file__), 'transformed_points.csv'), 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        x.append(float(row[0]))
        y.append(float(row[1]))

# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-', color='b')

# 添加标题和标签
plt.title('CSV Data Visualization')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示图表
plt.grid(True)
plt.show()