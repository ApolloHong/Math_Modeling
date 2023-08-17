import csv

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

f = open('数据.csv', 'w', encoding='utf-8')
csv_writer = csv.writer(f)

csv_writer.writerow(["Jan", "-9.9", "10.3"])
csv_writer.writerow(["Feb", "-8.6", "8.5"])
csv_writer.writerow(["Mar", "-10.2", "11.8"])
csv_writer.writerow(["Apr", "-1.7", "12.2"])
csv_writer.writerow(["May", "-0.6", "23.1"])
csv_writer.writerow(["Jun", "3.7", "25.4"])
csv_writer.writerow(["Jul", "6", "26.2"])
csv_writer.writerow(["Aug", "6.7", "21.4"])
csv_writer.writerow(["Sep", "3.5", "19.5"])
csv_writer.writerow(["Oct", "-1.3", "16"])
csv_writer.writerow(["Nov", "-8.7", "9.4"])
csv_writer.writerow(["Dec", "-9", "8.6"])

f.close()

file = pd.read_csv('数据.csv', header=None, sep=',')  # 读取数据
data = pd.DataFrame(file)

Months = data[0].sort_index(ascending=False)  # 读取月份数据
low_temp = data[1].sort_index(ascending=False)  # 读取最低温度
high_temp = data[2].sort_index(ascending=False)  # 读取最高温度

fig = plt.figure()  # 画图

plt.xlim(-15, 30)  # x轴坐标范围
plt.suptitle('Temperature variation by month', size='18')  # 图表大标题
plt.title('Observed in Vik i Sogn,Norway,2017', size='10')  # 图表小标题

low = plt.barh(Months, low_temp, color='white', left=0)  # 绘制白色的最低温水平柱状图
high = plt.barh(Months, high_temp - low_temp, color='skyblue', left=low_temp)  # 以最低温为左端点绘制紫色的温度区间水平柱状图

low_x = [x.get_width() for x in low]  # 最低温标签横坐标
high_x = high_temp  # 最高温标签横坐标

y = [x.get_y() for x in low]
h = [x.get_height() for x in low]
text_y = [y + height / 2 for y, height in zip(y, h)]  # 数字标签纵坐标

for x, y in zip(low_x, text_y):  # 标注最低温
    plt.text(x, y, str(np.round(x, 1)), horizontalalignment="right", verticalalignment='center')

for x, y in zip(high_x, text_y):  # 标注最高温
    plt.text(x, y, str(np.round(x, 1)), verticalalignment='center')

plt.xlabel('Temperature(℃)')  # 横坐标名称
plt.grid(axis='x')
plt.savefig('条形区域图.jpg')
