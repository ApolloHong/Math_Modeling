import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import *
from scipy.integrate import solve_ivp
from sympy import diff, dsolve, simplify, Function, sin
from sympy.abc import x


# 初值微分方程
def func1():
    y = Function('y')
    eq = diff(y(x), x, 2) + 2 * diff(y(x), x) + 2 * y(x)  # 定义方程
    con = {y(0): 0, diff(y(x), x).subs(x, 0): 1}  # 定义初值条件
    y = dsolve(eq, ics=con)
    print(simplify(y))


def func2():
    y = Function('y')
    eq = diff(y(x), x, 2) + 2 * diff(y(x), x) + 2 * y(x) - sin(x)  # 定义方程
    con = {y(0): 0, diff(y(x), x).subs(x, 0): 1}  # 定义初值条件
    y = dsolve(eq, ics=con)
    print(simplify(y))


# 微分方程组带初值
def func3():
    t = sp.symbols('t')
    x1, x2, x3 = sp.symbols('x1,x2,x3', cls=sp.Function)
    eq = [x1(t).diff(t) - 2 * x1(t) + 3 * x2(t) - 3 * x3(t),
          x2(t).diff(t) - 4 * x1(t) + 5 * x2(t) - 3 * x3(t),
          x3(t).diff(t) - 4 * x1(t) + 4 * x2(t) - 2 * x3(t)]
    con = {x1(0): 1, x2(0): 2, x3(0): 3}
    s = sp.dsolve(eq, ics=con)
    print(s)


def func3_matix():
    t = sp.symbols('t')
    x1, x2, x3 = sp.symbols('x1:4', cls=sp.Function)
    x = sp.Matrix([x1(t), x2(t), x3(t)])
    A = sp.Matrix([2, -3, 3], [4, -5, 3], [4, -4, 2])
    eq = x.diff(t) - A * x
    s = sp.dsolve(eq, ics={x1(0): 1, x2(0): 2, x3(0): 3})
    print(s)


##数值解法scipy

# odeint
def ex_seconde_ordre():
    '''
    plot 符号解和解析解
    '''

    def Pfun(y, x):
        y1, y2 = y
        return np.array([y2, -2 * y1 - 2 * y2])

    x = np.arange(0, 10, 0.1)  # 创建时间点
    sol1 = odeint(Pfun, [0.0, 1.0], x)  # 求数值解
    plt.rc('font', size=16)
    plt.rc('font', family='SimHei')
    plt.plot(x, sol1[:, 0], 'r*', label="数值解")
    plt.plot(x, np.exp(-x) * np.sin(x), 'g', label="符号解曲线")
    plt.legend()
    plt.savefig("figure8_5.png")
    plt.show()


def ex_pendulum():
    def pend(y, t, b, c):
        theta, omega = y
        dydt = [omega, -b * omega - c * np.sin(theta)]
        return dydt

    b = 0.25
    c = 5.0
    y0 = [np.pi * 3 / 4, 0.0]
    t = np.linspace(0, 30, 301)
    sol = odeint(pend, y0, t, args=(b, c))
    plt.plot(t, sol[:, 0], 'b', label='theta(t)')
    plt.plot(t, sol[:, 1], 'g', label='omega(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()


# solve_ivp

def ex1():
    def exponential_decay(t, y): return -0.5 * y

    # Basic exponential decay showing automatically chosen time points.
    sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8])
    print(sol.t)
    print(sol.y)

    # Basic exponential decay showing automatically chosen time points.
    sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8],
                    t_eval=[0, 1, 2, 4, 10])
    print(sol.t)
    print(sol.y)


def ex2():
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

    def fun(t, y):
        y1 = y[0]
        y2 = y[1]
        dydt = [y2, t ** 2 - y1 + 2 * t]
        return dydt

    # 初始条件
    y0 = [0, 1]
    # 第二个位置是范围，t_eval是真正分割的点
    yy = solve_ivp(fun, np.arange(1, 50, 0.01), y0, method='Radau', t_eval=np.arange(1, 50, 0.01))
    t = yy.t
    data = yy.y
    plt.plot(t, data[0, :], c='r')
    plt.plot(t, data[1, :], c='b')
    plt.xlabel("时间s")
    plt.legend(["求解变量"])
    plt.show()


def ex3():
    '''
     Cannon fired upward with terminal event upon impact.
     The terminal and direction fields of an event are applied by monkey patching a function.
     Here y[0] is position and y[1] is velocity.
     The projectile starts at position 0 with velocity +10.
     Note that the integration never reaches t=100 because the event is terminal.

    :return:
    '''

    def upward_cannon(t, y): return [y[1], -0.5]

    def hit_ground(t, y): return y[0]

    hit_ground.terminal = True
    hit_ground.direction = -1

    def apex(t, y): return y[1]

    sol = solve_ivp(upward_cannon, [0, 100], [0, 10],
                    events=(hit_ground, apex), dense_output=True)

    print(sol.t_events)
    print(sol.t)
    print(sol.y_events)


def ex4_Lotka_Volterra():
    def lotkavolterra(t, z, a, b, c, d):
        x, y = z
        return [a * x - b * x * y, -c * y + d * x * y]

    sol = solve_ivp(lotkavolterra, [0, 15], [10, 5], args=(1.5, 1, 3, 1),
                    dense_output=True)
    t = np.linspace(0, 15, 300)
    z = sol.sol(t)
    print(z.shape)
    plt.plot(t, z.T)
    plt.xlabel('t')
    plt.legend(['x', 'y'], shadow=True)
    plt.title('Lotka-Volterra System')
    plt.savefig('plot_ode_Lotka_Volterra.png', dpi=800)
    plt.show()


def func4():
    y = Function('y')
    func = y(x).diff + 2 * y(x) - x ** 2 - 2 * x
    y0 = y(1)
    t = 10
    sol = odeint(func, y0, t)


def ex_chaos():
    def lorenz(w, t):
        sigma = 10
        rho = 28
        beta = 8 / 3
        x, y, z = w
        return np.array([sigma * (y - x), rho * x - y - x * z, x * y - beta * z])

    t = np.arange(0, 50, 0.01)  # 建时点
    sol1 = odeint(lorenz, [0.0, 1.0, 0.0], t)  # 第一个初值问题求解
    sol2 = odeint(lorenz, [0.0, 1.0001, 0.0], t)  # 第二个初值问题求解
    # plt.rc('font',size=16)
    # plt.rc('text',usetex=True)
    ax1 = plt.subplot(121, projection='3d')
    ax1.plot(sol1[:, 0], sol1[:, 1], sol1[:, 2], 'r', linewidth=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax2 = plt.subplot(122, projection='3d')
    ax2.plot(sol1[:, 0] - sol2[:, 0], sol1[:, 1] - sol2[:, 1], sol1[:, 2] - sol2[:, 2], 'g')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    plt.savefig('figure8_6.png', dpi=500)
    plt.show()
    print("sol1=", sol1, '\n\n', "sol1-sol2=", sol1 - sol2)


def ex_temperature():
    t, k = sp.var('t, k')  # 定义符变t,k
    u = sp.var('u', cls=sp.Function)  # 定义符函数
    eq = sp.diff(u(t), t) + k * (u(t) - 24)  # 定义方程
    uu = sp.dsolve(eq, ics={u(0): 150})  # 求微分方程的符号解
    print(uu)
    kk = sp.solve(uu, k)  # kk返回值是列表，可能有多个解
    k0 = kk[0].subs({t: 10.0, u(t): 100.0})
    print(kk, '\t', k0)
    u1 = uu.args[1]  # 提出符表达式
    u0 = u1.subs({t: 20, k: k0})  # 代入具体值
    print("20分钟后的温度为：", u0)






def ex_weiyuan():
    h = sp.var('h')  # 定义符变
    t = sp.var('t', cls=sp.Function)  # 定义数
    g = 9.8
    eq = t(h).diff(h) - 10000 * sp.pi / 0.62 / sp.sqrt(2 * g) * (h ** (3 / 2) - 2 * h ** (1 / 2))  # 定义方程
    t = sp.dsolve(eq, ics={t(1): 0})  # 求微分方程的符号解
    t = sp.simplify(t)
    print(t.args[0], t.args[1])
    print(t.args[1].pop_size(9))  # n(9)代表有效位数是9






def ex_traffic():
    v0 = np.array([45, 65, 80])
    T0 = 1
    L = 4.5
    I = 9
    mu = 0.7
    g = 9.8
    T = v0 / (2 * mu * g) + (I + L) / v0 + T0
    print(T)

# ex4_Lotka_Volterra()
# y = Function('y')
# eq = diff(y(x),x,2) - y(x) - (2*sp.exp(x)/(sp.exp(x)-1))
# y = sp.dsolve(eq)
# print(simplify(y))

# ex_pendulum()
# ex_chaos()
# ex_seconde_ordre()
# ex_temperature()
# ex_weiyuan()
