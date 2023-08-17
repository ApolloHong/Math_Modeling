import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm

sns.set_theme(style="darkgrid")  # 改为黑暗模式

x = [2.5, 3.9, 2.9, 2.4, 2.9, 0.8, 9.1, 0.8, 0.7, 7.9, 1.8, 1.9, 0.8, 6.5, 1.6, 5.8, 1.3, 1.2, 2.7]
y = [211, 167, 131, 191, 220, 297, 71, 211, 300, 107, 167, 266, 277, 86, 207, 115, 285, 199, 172]

plt.plot(x, y, 'o', c='Orange', label="原始数据点")
p = np.polyfit(x, y, deg=1)  # 一次多项式
print("拟合的多项式为：{}*x+{}".format(p[0], p[1]))
plt.rc('font', size=16)
plt.rc('font', family='SimHei')
sns.lineplot(x, np.polyval(p, x), label="拟合的直线")
print("预测值为：", np.polyval(p, 8))
plt.legend()
plt.savefig("figure4_25.png", dpi=500)
plt.show()

# 基于公式的Python程序如下：
# 程序文件Pex4_25_2.py
x = [2.5, 3.9, 2.9, 2.4, 2.9, 0.8, 9.1, 0.8, 0.7, 7.9, 1.8, 1.9, 0.8, 6.5, 1.6, 5.8, 1.3, 1.2, 2.7]
y = [211, 167, 131, 191, 220, 297, 71, 211, 300, 107, 167, 266, 277, 86, 207, 115, 285, 199, 172]

df = {'x': x, 'y': y}
res = sm.formula.ols('y~x', data=df).fit()
print(res.summary(), '\n')
ypred = res.predict(dict(x=8))
print('所求的预测值为：', list(ypred))

# 基于数组的Python程序如下：
# 程序文件Pex4_25_3.py

x = np.array([2.5, 3.9, 2.9, 2.4, 2.9, 0.8, 9.1, 0.8, 0.7, 7.9, 1.8, 1.9, 0.8, 6.5, 1.6, 5.8, 1.3, 1.2, 2.7])
y = np.array([211, 167, 131, 191, 220, 297, 71, 211, 300, 107, 167, 266, 277, 86, 207, 115, 285, 199, 172])
X = sm.add_constant(x)

md = sm.OLS(y, X).fit()  # 构建合模型
print(md.params, '\n-----\n')  # 提取回归系数
print(md.summary2())
ypred = md.predict([1, 8])  # 第一列必须加1
print("预测值为：", ypred)
