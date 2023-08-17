# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 导入/输入原始数据
# # raw_data = pd.read_csv('your_csv.csv')
# # y = raw_data['val'].T.values
#
# x = [1, 2, 3, 4, 5]
# x = np.arange(0, 4 * np.pi, 0.01)
# y1 = np.cos(x)
# x = np.linspace(0,3,10)
# y2 = 1 + 2*np.log10(x) + np.random.uniform(0.0, 0.5, len(x))
#
# fig = plt.figure()
# ax1 = plt.subplot()
# # 绘制面积图，更多属性请查阅https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
# ax1.fill_between(x, y1, alpha=0.5, linewidth=0)  #填充x,y1之间
# ax1.fill_between(x, y1, y2, alpha=.5, linewidth=0)  #填充y1,y2之间
#
#
# # 其他属性设置
# # 指定限定区域填充
# # where：定义从何处排除要填充的某些水平区域。填充区域由坐标x[其中]定义。更准确地说，如果其中[i]和其中[i+1]，则在x[i]和x[i+1]之间填充。请注意，此定义意味着where中两个假值之间的孤立真值不会导致填充。由于相邻的假值，真实位置的两侧仍保持未填充状态。
# ax1.fill_between(x, y, y2, where=(x>5)&(x<9),color='cyan', alpha=0.5)  #只针对(x>5)&(x<9)范围填充
#
#
# ax2 = plt.subplot()
# # 交点附近的填充
# # interpolate：在语义上，where通常用于y1>y2或类似的词。默认情况下，定义填充区域的多边形节点将仅放置在x阵列中的位置。这样的多边形无法描述上述靠近交点的语义。包含交叉点的x截面仅被剪裁。将“插值”设置为True将计算实际交点，并将填充区域延伸到此点。
# ax2.fill_between(x, y1, y2, where=(y1 > y2), color='C0', alpha=0.3,
#          interpolate=True)
# ax2.fill_between(x, y1, y2, where=(y1 <= y2), color='C1', alpha=0.3,
#          interpolate=True)         #交点附近也填充，但“插值”若为False，则交点附近不会填充
#
#
# # 端点处的取值
# #包含参数为三个{‘pre’,‘post’,‘mid’}，如果填充应为阶跃函数，即x之间的常数，则定义阶跃。该值确定阶跃发生的位置：
# #“pre”：y值从每个x位置持续向左，即间隔（x[i-1]，x[i]]的值为y[i]。
# #“post”：y值从每个x位置持续向右，即区间[x[i]，x[i+1]）的值为y[i]。
# #“mid”：步数出现在x位置的中间。
# plt.fill_between(a, b, 0, where = (a > 2) & (a < 5), color = 'green', step='pre')
# plt.fill_between(a, b, 0, where = (a > 2) & (a < 5), color = 'green', step='post')
# plt.fill_between(a, b, 0, where = (a > 2) & (a < 5), color = 'green', step='mid')

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # 过滤掉警告的意思
data = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCMCM培训视频上传教程\\k.xlsx")
data.head()
print(data)

# 图片显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 减号unicode编码
x = data['年份'].values.tolist()
y1 = data['资产负债率'].values.tolist()
y2 = data['营业收入增长率'].values.tolist()
y = np.vstack([y1, y2])
print(y)

fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=80)  # 设置图片大小
columns = data.columns[2:]  # 提取第三列及之后的列名
labs = columns.values.tolist()  # 设置图片显示的标签

ax = plt.gca()
ax.stackplot(x, y, labels=labs, alpha=0.8)

# 图片标题
ax.set_title('堆叠面积图', fontsize=18)

# 设置坐标轴取值范围
ax.set(ylim=[0, 1.5])
ax.legend(fontsize=10, ncol=4)
plt.xticks(x[::1], fontsize=10, horizontalalignment='center')
plt.yticks(np.arange(1, 2, 3), fontsize=10)
plt.xlim(x[0], x[-1])

plt.show()
