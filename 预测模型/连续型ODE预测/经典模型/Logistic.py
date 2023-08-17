import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Logistic: 增长率r(x) = r-sx

df = pd.DataFrame(pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\数据\\b55f057df6.xls", header=None))


# a = df.iloc[0]
# b = df.iloc[1]
# print(a,b)

def least_square_nonlinear():
    a = df.iloc[0]
    b = df.iloc[1]

    # with open("C:\\Users\\xiaohong\\Desktop\\MCM小组\\数据\\b55f057df6.xls") as f: #文件定象f
    #     s=f.read().splitlines() #返回每一行的数据
    # for i in range(0, len(s),2): #读入奇数行数据
    #     d1=s[i].split("\t")
    #     for j in range(len(d1)):
    #         if d1[j]!="":
    #             a. append(eval(d1[j]))#把非空的字符串转换为年代数据
    #
    # for i in range(1, len(s), 2): #读入偶数行数据
    #     d2=s[i].split("\t")
    #     for j in range(len(d2)):
    #         if d2[j] != "":
    #             b.append(eval(d2[j]))# 把非空的字符串转换为人口数据

    c = np.vstack((a, b))  # 构造两行的组
    np.savetxt("Pdata8_10_2.txt", c)  # 把数据保存起来供下面使用
    x = lambda t, r, xm: xm / (1 + (xm / 3.9 - 1) * np.exp(-r * (t - 1790)))
    bd = ((0, 200), (0.1, 1000))  # 约束两个参数的下界和上界
    popt, pcov = curve_fit(x, a, b, bounds=bd)
    print(popt)
    print("2010年的预测值为：", x(2010, *popt))


def least_square_linear():
    t0 = df.iloc[0]
    x0 = df.iloc[1]  # 提取年代数据及对应的人口数据
    b = np.diff(x0) / 10 / x0[:-1]  # 构造线性方程组的常数项列
    a = np.vstack([np.ones(len(x0) - 1), -x0[:-1]]).T  # 构造线性方程组系数矩阵
    rs = np.linalg.pinv(a) @ b
    r = rs[0]
    xm = r / rs[1]
    print('人口增长率r和人口最大值xm的拟合值分别为', np.round([r, xm], 4))  # np.round(data, decimal)
    xhat = xm / (1 + (xm / 3.9 - 1) * np.exp(-r * (2010 - 1790)))  # 求预测值
    print('2010年的预测值为：', np.round(xhat, 4))


least_square_linear()
least_square_nonlinear()
