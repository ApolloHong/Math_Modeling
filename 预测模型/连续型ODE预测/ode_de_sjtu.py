import matplotlib.pyplot as plt
import numpy as np


def ode():
    rc = 2.0  # 设置常数
    dt = 0.5  # 设置步长
    n = 1000  # 设置分割段数
    t = 0.0  # 设置初始时间
    q = 1.0  # 设置初始电量

    # 先定义三个空列表
    qt = []  # 用来盛放差分得到的q值
    qt0 = []  # 用来盛放解析得到的q值
    time = []  # 用来盛放时间值

    for i in range(n):
        t = t + dt
        q1 = q - q * dt / rc  # qn+1的近似值
        q = q - 0.5 * (q1 * dt / rc + q * dt / rc)  # 差分递推关系
        q0 = np.exp(-t / rc)  # 解析关系
        qt.append(q)  # 差分得到的q值列表
        qt0.append(q0)  # 解析得到的q值列表
        time.append(t)  # 时间列表

    plt.plot(time, qt, 'o', label='Euler-Modify')  # 差分得到的电量随时间的变化
    plt.plot(time, qt0, 'r-', label='Analytical')  # 解析得到的电量随时间的变化
    plt.xlabel('time')
    plt.ylabel('charge')
    plt.xlim(0, 20)
    plt.ylim(-0.2, 1.0)
    plt.show()
    plt.close()


def pde():
    h = 0.1  # 空间步长
    N = 30  # 空间步数
    dt = 0.0001  # 时间步长
    M = 10000  # 时间的步数
    A = dt / (h ** 2)  # lambda*tau/h^2
    U = np.zeros([N + 1, M + 1])  # 建立二维空数组
    Space = np.arange(0, (N + 1) * h, h)  # 建立空间等差数列，从0到3，公差是h

    # 边界条件
    for k in np.arange(0, M + 1):
        U[0, k] = 0.0
        U[N, k] = 0.0

    # 初始条件
    for i in np.arange(0, N):
        U[i, 0] = 4 * i * h * (3 - i * h)

    # 递推关系
    for k in np.arange(0, M):
        for i in np.arange(1, N):
            U[i, k + 1] = A * U[i + 1, k] + (1 - 2 * A) * U[i, k] + A * U[i - 1, k]

    # 不同时刻的温度随空间坐标的变化
    plt.plot(Space, U[:, 0], 'g-', label='t=0', linewidth=1.0)
    plt.plot(Space, U[:, 3000], 'b-', label='t=3/10', linewidth=1.0)
    plt.plot(Space, U[:, 6000], 'k-', label='t=6/10', linewidth=1.0)
    plt.plot(Space, U[:, 9000], 'r-', label='t=9/10', linewidth=1.0)
    plt.plot(Space, U[:, 10000], 'y-', label='t=1', linewidth=1.0)
    plt.ylabel('u(x,t)', fontsize=20)
    plt.xlabel('x', fontsize=20)
    plt.xlim(0, 3)
    plt.ylim(-2, 10)
    plt.legend(loc='upper right')
    plt.show()

    # 温度等高线随时空坐标的变化，温度越高，颜色越偏红
    extent = [0, 1, 0, 3]  # 时间和空间的取值范围
    levels = np.arange(0, 10, 0.1)  # 温度等高线的变化范围0-10，变化间隔为0.1
    plt.contourf(U, levels, origin='lower', extent=extent, cmap=plt.cm.jet)
    plt.ylabel('x', fontsize=20)
    plt.xlabel('t', fontsize=20)
    plt.show()


pde()
