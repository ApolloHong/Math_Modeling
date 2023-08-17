import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams.update({'font.size': 15})  # 设置字体大小

a = [np.random.randint(100, 1000, size=(20,)) for i in range(5)]  # 生成随机数据

plt.figure(figsize=(20, 10))  # 设置图表大小
plt.stackplot(range(20), *[i / np.sum(a, axis=0) for i in a], labels=["the {}".format(i + 1) for i in range(5)])  # 堆积图
plt.legend(loc=0, fontsize=10)  # 图例位置
plt.yticks([i / 10 for i in range(0, 11, 1)], labels=["{}%".format(i * 10) for i in range(11)])  # y轴坐标
plt.xticks(range(20), labels=[(lambda i: "第{}个".format(i + 1) if not i % 2 else None)(i) for i in range(20)])  # x轴坐标
plt.xlabel("x轴标签")
plt.ylabel("百分比")
# plt.savefig("输出.png")#保存图片
plt.show()  # 显示图片
