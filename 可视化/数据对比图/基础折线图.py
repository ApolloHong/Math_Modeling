# 导包，设置环境
# import pandas as pd
import pylab as plt

# 导入/输入原始数据
# raw_data = pd.read_csv('your_csv.csv')
# y = raw_data['val'].T.values


y = [150, 230, 224, 218, 135, 147, 260, 170, 220, 210, 231, 150, 160, 240]
y1 = y[0:int(len(y) / 2)]
y2 = y[int(len(y) / 2):len(y)]
x1 = range(0, len(y1))
x2 = range(0, len(y2))

# 绘制折线图，更多属性请查阅https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
h1 = plt.plot(x1, y1, label='Line1', linestyle='-', linewidth=1, color='blue', marker='x', markersize=4,
              markeredgecolor='green', markerfacecolor='red')
h2 = plt.plot(x2, y2, label='Line2', linestyle=':', linewidth=1, color=[1, .05, .5], marker='o', markersize=4,
              markeredgecolor='green', markerfacecolor='red')

# 其他属性设置
# 坐标轴
# plt.xlim()    # x轴范围
plt.ylim(min(min(y1), min(y2)), max(max(y1), max(y2)))  # y轴范围
plt.xticks([0, 1, 2, 3, 4, 5, 6], ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
           fontname='Times New Roman')  # x轴标签，第一个参数为欲替换的x坐标，第二个参数为对应替换坐标的标签
# plt.yticks()


# 坐标轴标签，更多属性请查阅https://matplotlib.org/api/text_api.html#matplotlib.text.Text
# plt.xlabel('Year',fontname='Times New Roman',fontsize=10)
plt.ylabel('Val')

# 图例显示
# loc--位置（1:右上，2：左上，3：左下，4：右下）
plt.legend(loc=1, prop={'size': 10})

# 网格线开关
plt.grid(x1)
# plt.grid(x2);
plt.savefig('./line_py.png')
plt.show()
