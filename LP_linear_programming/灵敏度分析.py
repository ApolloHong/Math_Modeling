import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
from SALib.analyze import sobol
from SALib.sample import saltelli
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 修复中文乱码
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
np.seterr(divide='ignore', invalid='ignore')  # 消除被除数为0的警告
# 上面不用管

problem = {
    'num_vars': 4,  # 模型变量的数量
    'names': ['T', 'PT', 'Pr', 'i'],  # 模型的变量
    'bounds': [[2000, 10000],
               [0, 0.2],
               [0.1, 0.5],
               [1, 30],
               ]  # 指定变量的范围，一一对应
}


def evaluate(xx):  # 进行灵敏度分析的模型

    return np.array([x[0] * np.power(1 - (1 - np.power(1 - x[1], 1.4263 * np.power(x[1], -0.426))) * x[2],
                                     x[3] - 1) * np.power(x[1], 0.426) / 1.4263 * 5 + x[0] * np.power(
        1 - (1 - np.power(1 - x[1], 1.4263 * np.power(x[1], -0.426))) * x[2], x[3] - 1) * (
                             1 - np.power(1 - x[1], 1.4263 * np.power(x[1], -0.426))) * 16 for x in xx])


'''
    注意返回的是np.array([function for x in X]) function是函数表达式
    比如function(T,PT,Pr,i)=T+PT+Pr+i 那么function就写成x[0]+x[1]+x[2]+x[3]
    很显然，一一对应定义模型中的变量，注意列表下标从0开始
'''

# 下面不用管
samples = 128
param_values = saltelli.sample(problem, samples)
print('模型运行次数', len(param_values))
Y = evaluate(param_values)
Si = sobol.analyze(problem, Y)
print()
print('ST:', Si['ST'])  # 总灵敏度
print('S1:', Si['S1'])  # 一阶灵敏度
print("S2 Parameter:", Si['S2'][0, 1])  # 二阶灵敏度

# 一阶灵敏度与总灵敏度图片
df_sensitivity = pd.DataFrame({
    "Parameter": problem["names"],
    "一阶灵敏度": Si["S1"],
    "总灵敏度": Si["ST"]}
).set_index("Parameter")
fig, axes = plt.subplots(figsize=(10, 6))
df_sensitivity.plot(kind="bar", ax=axes, rot=45, fontsize=16)
pl.show()

# 二阶灵敏度图片
second_order = np.array(Si['S2'])
pd.DataFrame(second_order, index=problem["names"], columns=problem["names"])
figs, axes = plt.subplots(figsize=(8, 10))
ax_image = axes.matshow(second_order, vmin=-1.0, vmax=1.0, cmap="RdYlBu")
cbar = figs.colorbar(ax_image)
ax_image.axes.set_xticks(range(len(problem["names"])))
ax_image.axes.set_xticklabels(problem["names"], rotation=45, fontsize=24)
r = ax_image.axes.set_yticklabels([""] + problem["names"], fontsize=24)

plt.show()
