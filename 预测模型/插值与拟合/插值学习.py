import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# import taichi as ti
# ti.init(arch=ti.cpu)

# df = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\数据\\b55f057df6.xls", header=None)
# x = df.iloc[0]
# y = df.iloc[1]
# print(df.max(x))


# lagrage intepolate
def inter_lagrange():
    df = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\数据\\b55f057df6.xls", header=None)
    x = df.iloc[0]
    y = df.iloc[1]
    lst = []

    def h(x, y, a):
        s = 0.0
        for i in range(len(y)):
            t = y[i]
            for j in range(len(y)):
                if i != j:
                    t *= (a - x[j]) / (x[i] - x[j])
            s += t
        return s

    x = x.tolist()
    a = x[0]
    b = x[-1]
    xhat = np.linspace(a, b, 100)
    for k in xhat:
        lst.append(h(x, y, k))

    sns.relplot(xhat, lst)
    plt.show()


def diff_quo(xi=[], fi=[]):
    '''
    计算n阶差商 f[x0,x1,.....xn]
    :param xi: 所有插值结点的数组
    :param fi: 所有插值结点函数值的数组
    :return: 返回xi的n阶差商(n=len(xi)-1)
    '''
    if len(xi) > 2 and len(fi) > 2:
        return (diff_quo(xi[:len(xi) - 1], fi[:len(fi) - 1]) -
                diff_quo(xi[:len(xi)], fi[:len(fi)])) / float(xi[0] - xi[-1])
    return (fi[0] - fi[1]) / float(xi[0] - xi[1])
