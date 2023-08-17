import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.interpolate import interp1d, griddata, interp2d
# ‘grayscale’, ‘seaborn-bright’, ‘seaborn-muted’, ‘seaborn-notebook’, ‘seaborn-paper’,
# ‘seaborn-pastel’, ‘seaborn-poster’, ‘seaborn-talk’, ‘seaborn-ticks’, ‘seaborn-white’,
# ‘seaborn-whitegrid’, ‘seaborn’
plt.style.use('seaborn') #是seaborn的暗色系

def interpolate_1d():
    x = np.arange(0, 25, 2)
    y = np.array([12, 9, 9, 10, 18, 24, 28, 27, 25, 20, 18, 15, 13])
    xnew = np.linspace(0, 24, 500)  # 插值点

    f1 = interp1d(x, y)
    y1 = f1(xnew)
    f2 = interp1d(x, y, 'cubic')
    y2 = f2(xnew)

    y = np.vstack([y1,y2])

    plt.rc('font', size=16)
    plt.rc('font', family='SimHei')
    plt.subplot(121)
    plt.plot(xnew, y1)
    plt.xlabel("(A)分段线性插值")
    plt.subplot(122)
    plt.plot(xnew, y2)
    plt.xlabel("(B)三次样条插值")
    plt.savefig("figure7_4.png", dpi=500)
    plt.show()




def interp_2d_mesh():
    z = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\数据\\k4.xlsx")  # 加高程据
    z = z.to_numpy(dtype=None)
    x = np.arange(0, 1500, 100)
    y = np.arange(1200, -100, -100)
    f = interp2d(x, y, z, 'cubic')
    xn = np.linspace(0, 1400, 141)
    yn = np.linspace(0, 1200, 121)
    zn = f(xn, yn)
    m = len(xn)
    n = len(yn)
    s = 0

    for i in np.arange(m - 1):
        for j in np.arange(n - 1):
            p1 = np.array([xn[i], yn[j], zn[j, i]])
            p2 = np.array([xn[i + 1], yn[j], zn[j, i + 1]])
            p3 = np.array([xn[i + 1], yn[j + 1], zn[j + 1, i + 1]])
            p4 = np.array([xn[i], yn[j + 1], zn[j + 1, i]])

            p12 = norm(p1 - p2)
            p23 = norm(p3 - p2)
            p13 = norm(p3 - p1)
            p14 = norm(p4 - p1)
            p34 = norm(p4 - p3)

            L1 = (p12 + p23 + p13) / 2
            s1 = np.sqrt(L1 * (L1 - p12) * (L1 - p23) * (L1 - p13))
            L2 = (p13 + p14 + p34) / 2
            s2 = np.sqrt(L2 * (L2 - p13) * (L2 - p14) * (L2 - p34))
            s = s + s1 + s2

    print("区域的面积为：", s)
    plt.rc('font', size=16)
    plt.rc('text', usetex=True)
    plt.subplot(121)
    contr = plt.contour(xn, yn, zn)
    plt.clabel(contr)
    plt.xlabel('$x$')
    plt.ylabel('$y$', rotation=90)
    ax = plt.subplot(122, projection='3d')
    X, Y = np.meshgrid(xn, yn)
    ax.plot_surface(X, Y, zn, cmap='viridis')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    plt.savefig('figure7_5.png', dpi=500)
    plt.show()


def interp_3d_luandian():
    x = np.array([129, 140, 103.5, 88, 185.5, 195, 105, 157.5, 107.5, 77, 81, 162, 162, 117.5])
    y = np.array([7.5, 141.5, 23, 147, 22.5, 137.5, 85.5, -6.5, -81, 3, 56.5, -66.5, 84, -33.5])
    z = -np.array([4, 8, 6, 8, 6, 8, 8, 9, 9, 8, 8, 9, 4, 9])

    xy = np.vstack([x, y]).T
    xn = np.linspace(x.min(), x.max(), 100)
    yn = np.linspace(y.min(), y.max(), 100)
    xng, yng = np.meshgrid(xn, yn)  # 构网格节点

    zn = griddata(xy, z, (xng, yng), method='nearest')  # 最近邻点插值
    plt.rc('font', size=16)
    plt.rc('text', usetex=True)
    ax = plt.subplot(121, projection='3d')
    ax.plot_surface(xng, yng, zn, cmap='viridis')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    plt.subplot(122)
    c = plt.contour(xn, yn, zn, 8)
    plt.clabel(c)
    plt.savefig('figure7_6.png', dpi=500)
    plt.show()

# #test
# z = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\数据\\k4.xlsx")  # 加高程据
# z = z.to_numpy(dtype=None)


interpolate_1d()