import matplotlib.pyplot as plt
import numpy as np


def two_degree():
    N = 21
    x = np.linspace(0, 10, 11)
    y = [3.9, 4.4, 10.8, 10.3, 11.2, 13.1, 14.1, 9.9, 13.9, 15.1, 12.5]

    # fit a linear curve an estimate its y-values and their error.
    a, b, c = np.polyfit(x, y, deg=2)
    y_est = a * x ** 2 + b * x + c
    y_err = x.std() * np.sqrt(1 / len(x) +
                              (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    # 绘制拟合线
    fig, ax = plt.subplots()
    ax.plot(x, y_est, '-')
    # 填充置信带
    ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
    # 绘制数据点
    ax.plot(x, y, 'o', color='tab:brown')
    plt.show()


def one_degree():
    N = 21
    x = np.linspace(0, 10, 11)
    y = [3.9, 4.4, 10.8, 10.3, 11.2, 13.1, 14.1, 9.9, 13.9, 15.1, 12.5]

    # fit a linear curve an estimate its y-values and their error.
    a, b = np.polyfit(x, y, deg=1)
    y_est = a * x + b
    y_err = x.std() * np.sqrt(1 / len(x) +
                              (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    # 绘制拟合线
    fig, ax = plt.subplots()
    ax.plot(x, y_est, '-')
    # 填充置信带
    ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
    # 绘制数据点
    ax.plot(x, y, 'o', color='tab:brown')
    plt.show()


one_degree()
