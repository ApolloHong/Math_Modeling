import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 折线点的坐标
x = np.array([1, 3, 5, 7, 8])
y = np.array([1, 2, 9, 16, 15])

# 插值
x_new = np.linspace(x.min(), x.max(), 300)  # 300是在最小值与最大值之间生成的点数
y_smooth = make_interp_spline(x, y)(x_new)

# 散点图
plt.scatter(x, y, c='black', alpha=0.5)  # alpha:透明度 c:颜色
# 折线图
plt.plot(x, y, linewidth=1)  # 线宽linewidth=1
# 平滑后的折线图
plt.plot(x_new, y_smooth, c='red')

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # SimHei黑体
plt.rcParams['axes.unicode_minus'] = False

plt.title("绘图", fontsize=24)  # 标题及字号
plt.xlabel("X", fontsize=24)  # X轴标题及字号
plt.ylabel("Y", fontsize=24)  # Y轴标题及字号
plt.tick_params(axis='both', labelsize=14)  # 刻度字号
# plt.xticks(x)#X轴坐标设置
# plt.yticks(y)#Y轴坐标设置
# plt.axis([0, 20, 1, 20])#设置坐标轴的取值范围
plt.show()
# plt.save('squares_plot.png'（文件名）, bbox_inches='tight'（将图表多余的空白部分剪掉）)
# 用它替换plt.show实现自动保存图表
