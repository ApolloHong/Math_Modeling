import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.pyplot import plot, show, rc
from numpy import polyfit, polyval, array, arange
from scipy.optimize import curve_fit
plt.style.use('seaborn') #是seaborn的暗色系
sns.set_theme(style="darkgrid")  # 改为黑暗模式

# polyfit
def ex1():
    x0 = arange(0, 1.1, 0.1)
    y0 = array([-0.447, 1.978, 3.28, 6.16, 7.08, 7.34, 7.66, 9.56, 9.48, 9.30, 11.2])
    p = polyfit(x0, y0, 2)  # 拟合二次多项式
    print("拟合二次多项式的从高次幂到低次幂系数分别为：", p)
    yhat = polyval(p, [0.25, 0.35])
    print("预测值分别为：", yhat)
    rc('font', size=16)
    plot(x0, y0, '*', x0, polyval(p, x0), '-')
    show()


# curve fit
def ex2():
    '''
    # curve fit格式
    # popt, pcov = curve_fit(func, xdata, ydata)
    # 其中func是拟合的函数， xdata是自变量的观测值，ydata是函数的观测值，返回值popt是拟合的参数，pcov是参数的协方差矩阵
    :return:
    '''
    y = lambda x, a, b, c : a * x ** 2 + b * x + c
    x0 = np.arange(0, 1.1, 0.1)
    y0 = np.array([-0.447, 1.978, 3.28, 6.16, 7.08, 7.34, 7.66, 9.56, 9.48, 9.30, 11.2])
    popt, pcov = curve_fit(y, x0, y0)
    sns.heatmap(pcov,annot=True)
    plt.show()
    print("拟合的参数值为：", popt)
    print("预测值分别为：", y(np.array([0.25, 0.35]), *popt))


def ex3():
    '''
    z = a * np.exp(b * x) + c * y ** 2
    :return:
    '''
    x0 = np.array([6, 2, 6, 7, 4, 2, 5, 9])
    y0 = np.array([4, 9, 5, 3, 8, 5, 8, 2])
    z0 = np.array([5, 2, 1, 9, 7, 4, 3, 3])
    xy0 = np.vstack((x0, y0))

    def Pfun(t, a, b, c):
        return a * np.exp(b * t[0]) + c * t[1] ** 2

    popt, pcov = curve_fit(Pfun, xy0, z0)
    print("a,b,c的拟合值为：", popt)



def ex4():
    '''
    拟合曲面
    :return:
    '''
    m = 200
    n = 300

    x = np.linspace(-6, 6, m)
    y = np.linspace(-8, 8, n)
    x2, y2 = np.meshgrid(x, y)
    x3 = np.reshape(x2, (1, -1))
    y3 = np.reshape(y2, (1, -1))
    xy = np.vstack((x3, y3))

    def Pfun(t, m1, m2, s):
        return np.exp(-((t[0] - m1) ** 2 + (t[1] - m2) ** 2) / (2 * s ** 2))

    z = Pfun(xy, 1, 2, 3)
    zr = z + 0.2 * np.random.normal(size=z.shape)  # 噪声数据
    popt, pcov = curve_fit(Pfun, xy, zr)  # 拟合参数
    print("三个参数的拟合值分别为：", popt)

    zn = Pfun(xy, *popt)  # 计算合函数的值
    zn2 = np.reshape(zn, x2.shape)
    plt.rc('font', size=16)
    ax = plt.axes(projection='3d')  # 创建一个三维坐标轴对象
    ax.plot_surface(x2, y2, zn2, cmap='coolwarm')
    plt.savefig("figure7_10.png", dpi=500)
    plt.show()

ex1()
ex2()
ex3()
ex4()
