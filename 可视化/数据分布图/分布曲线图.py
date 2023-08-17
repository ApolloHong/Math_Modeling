import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

data = np.array(pd.read_excel('filename', header=None).iloc[:, 0])
kde = stats.gaussian_kde(data)
x = np.linspace(data.min(), data.max(), 100)
p = kde(x)
plt.plot(x, p)
plt.title('probability density function')  # 不支持中文，想插入中文可自行搜索解决方法
plt.savefig('test.jpg')
