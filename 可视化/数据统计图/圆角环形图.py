import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

pi = math.pi
startangle = 90
colors = ['#4393E5', '#43BAE5', '#7AE6EA']
datas = [60, 75, 80]
xs, ys = [], []
for idx, data in enumerate(datas):
    xs.append((data * pi * 2) / 100)
    ys.append(idx + 1)
left = (startangle * pi * 2) / 360  # 控制起始位置

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})

# 在末尾标出线条和点来使它们变圆
for i, x in enumerate(xs):
    ax.barh(ys[i], x, left=left, height=1, color=colors[i])
    ax.scatter(x + left, ys[i], s=350, color=colors[i], zorder=2)
    ax.scatter(left, ys[i], s=350, color=colors[i], zorder=2)

plt.ylim(-4, 4)

legend_elements = [Line2D([0], [0], marker='o', color='w', label='Group A', markerfacecolor='#4393E5', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Group B',
                          markerfacecolor='#43BAE5', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Group C', markerfacecolor='#7AE6EA', markersize=10)]
ax.legend(handles=legend_elements, loc='center', frameon=False)

plt.xticks([])
plt.yticks([])
ax.spines.clear()
plt.show()
