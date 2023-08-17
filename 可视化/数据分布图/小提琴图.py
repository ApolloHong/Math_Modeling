import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

plt.violinplot(dataset=np.random.normal(size=1000))
plt.show()
# 1000个标准正态分布的随机数据的小提琴图


sns.violinplot(data=[np.random.normal(size=1000), np.random.normal(size=1000), np.random.normal(size=1000)])
plt.show()
# 1000个标准正态分布的随机数据的小提琴图
