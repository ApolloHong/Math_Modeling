import matplotlib.pyplot as plt
import numpy as np

# 存储降雨量
data = []
# 读取文件
f = open("模拟降雨量.txt")
for line in f.readlines():
    x = line.split('\t')
    for s in x:
        data.append(int(s))
# x轴坐标
x = np.arange(0, len(data))
# 坐标轴防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 绘制直方图
plt.bar(x=x, height=data)
# 设置x轴坐标轴刻度标签
x_title = [str(i) for i in range(2009, 2020)]
myData = np.linspace(0, len(data), len(x_title))
plt.xticks([i for i in myData], x_title)
# 设置横纵轴与直方图标题
plt.xlabel('年份/年')
plt.ylabel('降水量/(' + r'$mm$' + ')')
plt.title("四川省宜宾市2009年到2019年间24小时模拟降雨量统计图")
plt.show()
