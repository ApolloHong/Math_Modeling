import matplotlib.pyplot as plt
import numpy as np

N = 10
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)

# 设置横坐标上的统计对象，N表示对象的个数

radii = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 设置纵坐标数据

width = np.pi / 5

# 设置扇形的宽度，可以设置成有重叠的部分，设置紧密就可以实现类似玫瑰图的效果

colors = plt.cm.inferno(radii / 10)

# 设置图中扇形颜色，“viridis”是一个数字到颜色的映射，类似的映射还有plasma，inferno等。

X = np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])

# 设置统计对象

ax = plt.subplot(projection='polar')
ax.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=1)

# alpha设置透明度

for z, x, y in zip(theta, X, radii): ax.text(z, y, x, fontsize=10, horizontalalignment='center')

plt.show()
