import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cvxopt import matrix, solvers
from numpy import ones
from scipy.optimize import minimize


def model1():
    def obj(x):
        x1, x2, x3 = x
        return (2 + x1) / (1 + x2) - 3 * x1 + 4 * x3

    LB = [0.1] * 3
    UB = [0.9] * 3
    bound = tuple(zip(LB, UB))  # 生成决策向量界限的元组
    res = minimize(obj, ones(3), bounds=bound)  # 第2个参数为初值
    print(res.fun, '\n', res.success, '\n', res.x)


def model2():
    c1 = np.array([1, 1, 3, 4, 2])
    c2 = np.array([-8, -2, -3, -1, -2])
    n = 3
    P = matrix(0., (n, n))
    P[::n + 1] = [3, 2, 1.7]
    q = matrix([3, -8.2, -1.95])
    A = matrix([[1., 0, 1], [-1, 2, 0], [0, 1, 2]]).T
    b = matrix([2., 2, 3])
    Aeq = matrix(1., (1, n))
    beq = matrix(3.)
    s = solvers.qp(P, q, A, b, Aeq, beq)
    print("最优解为：", s['x'])
    print("最优值为：", s['primal objective'])


def model3():
    c1 = np.array([1, 1, 3, 4, 2])
    c2 = np.array([-8, -2, -3, -1, -2])
    a = np.array([[1, 1, 1, 1, 1], [1, 2, 2, 1, 6], [2, 1, 6, 0, 0], [0, 0, 1, 1, 5]])
    b = np.array([400, 800, 200, 200])
    x = cp.Variable(5, integer=True)
    obj = cp.Minimize(c1 * x ** 2 + c2 * x)
    con = [0 <= x, x <= 99, a * x <= b]
    prob = cp.Problem(obj, con)
    prob.solve()
    print('最优值为:', prob.value)
    print('最优解为:\n', x.value)


def model4():
    n = 3
    P = matrix(0., (n, n))
    P[::n + 1] = [3, 2, 1.7]
    q = matrix([3, -8.2, -1.95])
    A = matrix([[1., 0, 1], [-1, 2, 0], [0, 1, 2]]).T
    b = matrix([2., 2, 3])
    Aeq = matrix(1., (1, n))
    beq = matrix(3.)
    s = solvers.qp(P, q, A, b, Aeq, beq)
    print("最优解为：", s['x'])
    print("最优值为： ", s['primal objective'])


def model5():
    x0 = np.array([150, 85, 150, 145, 130, 0])
    y0 = np.array([140, 85, 155, 50, 150, 0])
    q = np.array([243, 236, 220.5, 159, 230, 52])  # 地面参考系下飞机偏向角
    d = np.zeros((6, 6))  # 两两飞机之间的距离
    a0 = np.zeros((6, 6))  # α0两个飞机不碰撞的临界角
    b0 = np.zeros((6, 6))  # β0两两飞机的相对速度夹角
    xy0 = np.c_[x0, y0]  # 按列拼接两个数组，表示所有飞机的横纵坐标

    for i in range(6):
        for j in range(6):
            d[i, j] = np.linalg.norm(xy0[i] - xy0[j])
    d[np.where(d == 0)] = np.inf  # np.where后加条件，若不满足，则执行后面的命令(等于无穷)。
    a0 = np.arcsin(8. / d) * 180 / np.pi  # 换成角度值
    xy1 = x0 + 1j * y0  # 1j为虚数i  , xy1是位置
    xy2 = np.exp(1j * q * np.pi / 180)  # xy2是角度，弧度制

    for m in range(6):
        for n in range(6):
            if n != m:
                # 相对速度角度为地面系看到的速度差角度减去地面系相对位置角度
                # arg 相减得到 arg两项相除
                b0[m, n] = np.angle((xy2[n] - xy2[m]) / (xy1[m] - xy1[n]))
    b0 = b0 * 180 / np.pi  # 化为角度制
    f = pd.ExcelWriter('Pan6_1.xlsx')  # 创建文件对象
    pd.DataFrame(a0).to_excel(f, "sheet1", index=None)  # 把a0写入Excel文件
    pd.DataFrame(b0).to_excel(f, "sheet2", index=None)  # 把b0写入表单2
    f.save()

    plt.figure()
    sns.heatmap(a0, annot=True)
    plt.show()
    plt.figure()
    sns.heatmap(b0, annot=True)
    plt.show()

    a0 = pd.read_excel("Pan6_1.xlsx")  # 读入第1个表单
    b0 = pd.read_excel("Pan6_1.xlsx", 1)  # 读入第2个表单
    a0 = a0.values
    b0 = b0.values
    obj = lambda x: np.sum(np.abs(x))
    bd = [(-30, 30) for i in range(6)]  # 决策向量的界限
    cons = []
    for i in range(5):
        for j in range(i + 1, 6):
            # cons是默认大于零
            cons.append({'type': 'ineq', 'fun': lambda x: np.abs(b0[i, j] + (x[i] + x[j]) / 2) - a0[i, j]})

    res = minimize(obj, np.ones((1, 6)), constraints=cons, bounds=bd)
    print(res)



