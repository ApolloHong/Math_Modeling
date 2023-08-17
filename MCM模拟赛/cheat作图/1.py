import random
from pynverse import inversefunc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.optimize import minimize


plt.rc('font', family='Times New Roman', size=15)
plt.style.use('seaborn')  # 是seaborn的暗色系
sns.set_theme(style="darkgrid")  # 改为黑暗模式

df_1 = pd.read_csv("C:\\Users\\xiaohong\\Desktop\\MCM小组\\数据\\h2.csv")
df_2 = pd.read_csv("C:\\Users\\xiaohong\\Desktop\\MCM小组\\数据\\h1.csv")

xdata = df_1.iloc[:, 0]
ydata = df_1.iloc[:, 1]


a0 = 5.070303


def func1(x):
    return 185 / (1 + 114.52 * np.exp(-0.051 * x))


def func2(x):
    return 1 / (1 + 35.54 * np.exp(-0.030 * x))


def logistic_fit():
    y = lambda x, FWmax, C, k: FWmax / (1 + C * np.exp(-k * x))
    x0 = xdata
    y0 = func1(x0) + func2(x0)
    popt, pcov = curve_fit(y, x0, y0 * 2)
    print("拟合的参数值为：", popt)
    # print("预测值分别为：", y(np.array([0.25, 0.35]), *popt))

def logistic_fit_multi():
    plt.rc('font', family='Times New Roman', size=15)
    alpha = [3.654424944,5.070303144,6.486181344,7.902059544]
    y1 = [106.295, 120.868,92.666,95.144]
    y2 = [150,200,250,300]

    y = lambda t, FWmax, C, k : FWmax / (1 + C * np.exp(-k * a0 * t))
    t0 = xdata/a0
    popt, pcov = curve_fit(y, t0, (func1(xdata) + func2(xdata)) * 2)
    print("拟合的参数值为：", popt)

    def func_y(t):
        return y(t,*popt)
    # 它可用于计算某些y_values 点的反函数：
    inv_func_t = inversefunc(func_y, y_values=120.868*2)
    print(inv_func_t)

    func_y2 = lambda ta : y(20.74791237979409, 3.71620915e+02, 1.13784571e+02, 5.09087061e-02*ta)


    for i in range(4):
        inv_func_y_ta = inversefunc(func_y2, y_values=y1[i] * 2)
        print(inv_func_y_ta*alpha[i]/alpha[1])
        x = np.linspace(0,250,1000)
        def func(t):
            return y(t, 3.71620915e+02, 1.13784571e+02, 5.09087061e-02 * inv_func_y_ta)
        myfunc = np.frompyfunc(func,1,1)
        plt.plot(x/a0,myfunc(x/a0),label=f'I = {y2[i]} $\mu mol\cdot s^{-1}\cdot m^{-2}$')
        plt.legend()

    plt.xlabel('t ($days$)')
    plt.ylabel('FW($g$)')
    plt.show()


def fit2():
    plt.rc('font', family='Times New Roman', size=15)
    theta = [0.7682211766588841, 0.5302912548636192, 0.44018526632699606]
    # for i in range(1,4):
    #         theta[i] /= theta[0]
    I1 = [200,250,300]
    def func(I,I0, m):
        return 2 - np.exp((I-I0) *m )
    # y = lambda m ,x, I0 : 2 - np.exp(- m * (x-I0) )
    popt, pcov = curve_fit(func, I1, theta)
    print("拟合的参数值为：", popt)
    # print(np.log( (2-theta[1]) / (2-theta[0]) ) / 50)
    # print(250 - (np.log(2-theta[0]) / 0.0017153894354319523) )

def fit3():
    plt.rc('font', family='Times New Roman', size=15)
    theta = [1.0 ,0.6902846094010843,0.5729928823902404]
    I1 = [200,250,300]
    y = lambda I,m : 2 - np.exp(-m*(I-214.3))
    popt, pcov = curve_fit(y, I1, theta)
    print(popt)
    # return y(t, *popt)
    # sns.regplot(I1,theta,order=2)
    # plt.show()
    def func(i):
        if i<=214.3:
            return 1
        else:
            return 1-0.00003*(i-214.3)**2
    my_func = np.frompyfunc(func,1,1)
    i0=np.linspace(0,400,1000)
    plt.plot(i0,my_func(i0),'g')
    plt.xlabel('I($\mu mol\cdot s^{-1}\cdot m^{-2}$)')
    plt.ylabel('θ(/)')
    plt.show()

def plot_energysave():
    plt.rc('font', family='Times New Roman', size=12)
    def func1(x,M):
        return 0.8 * M * np.sin(15/3.14 * x) * random.randint(-1,1) + M * x
    x0 = np.linspace(0,31,1000)
    lb = ['Spring','Summer','Autumn','Winter']
    name = '$E_{air}$'

    y = lambda x: x/480
    my_func = np.frompyfunc(y,1,1)
    M1 = [72,96,77,66]
    M2 = my_func(M1)
    for i in range(4):
        plt.semilogy(x0 ,func1(x0,M1[i]),'b',label=f'{name} in {lb[i]}')
        plt.semilogy(x0,func1(x0,M2[i]),'g',label=f'$E_f$ in {lb[i]}')
        plt.legend()
    plt.xlabel('Days in a month')
    plt.ylabel('Energy cost(J)')
    plt.show()






# plot_energysave()

def plot_adapt():
    plt.rc('font', family='Times New Roman', size=12)
    df = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\数据\\k6.xlsx",header = None)
    lst = [6,17,22,35]
    l = [0.041,0.006,0.017]
    lb = ['Spring','Summer','Autumn','Winter']
    la = ['Glass wool','Aerogel vacuum panel','Aerogel blankets']
    # for i in range(12):
    #     lst.append(df.iloc[i])

    for t in range(4):
        fig = plt.figure(t)
        for k in range(t,t+3):
            plt.plot(df.iloc[k],label = f'In {lb[t]}; Material : {la[k-t]}')
            plt.xlabel('Iteration time')
            plt.ylabel('Yield/Energy consumption ratio (g/J)')
            plt.legend(loc=0)
            if t == 0:
                plt.legend(loc=1,bbox_to_anchor=(0.9,0.8))
    plt.show()


def plot_heat():
    plt.rc('font', family='Times New Roman', size=12)
    dfa = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\数据\\heatmap.xlsx",header=None)
    dfb = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\数据\\heatmap2.xlsx",header=None)
    x1 = [132,133,134,136,137,139,140,142,143]
    y1 = [0.0385,0.0389,0.0393,0.0397,0.0401,0.0405,0.041,0.0414,0.0418,0.0422,0.0426]
    x2 = [203,205,207,209,211,214,216,218,220,222,224]
    y2 = [20.9,21.12,21.34,21.56,21.78,22,22.22,22.44,22.66,22.88]
    matrix1 = pd.DataFrame(dfa.iloc[:,:])
    matrix1.columns=x1
    matrix1.index=y1
    # matrix1.pivot(index=,columns=0)
    matrix2 = pd.DataFrame(dfb.iloc[:,:])
    # print(dfa)
    # f, ax = plt.subplots(figsize=(9, 11))
    # ax = sns.heatmap(matrix1,annot=True)
    # ax.set(xlabel=x1,ylabel=y1)
    sns.heatmap(matrix1,annot=True,cmap="crest")
    plt.show()



plot_heat()
# plot_adapt()
# logistic_fit_multi()
def plot_fw_i():
    plt.rc('font', family='Times New Roman', size=10)
    df1 = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\数据\\FW_T_I_curve.xlsx",header = None)
    I = df1.iloc[0]
    id6 = df1.iloc[1]
    id11 = df1.iloc[2]
    id16 = df1.iloc[3]
    id21 = df1.iloc[4]
    T = df1.iloc[5]
    td6 = df1.iloc[6]
    td11 = df1.iloc[7]
    td16 = df1.iloc[8]
    td21 = df1.iloc[9]


    # for k in range(4):
    #     plt.plot(df1.iloc[k+1])
    #     plt.show()


    fig1, axs = plt.subplots(2, 2,sharex=True,sharey=True)

    axs[0,0].plot(I,id6,'g')
    # axs[0, 0].set_xlim(0,10)
    # axs[0, 0].set_ylim(0,10)
    axs[0,0].set_title('day6')

    axs[1,0].plot(I,id11,'g')
    # axs[1, 0].set_xlim(0,10)
    # axs[1, 0].set_ylim(0,10)
    axs[1,0].set_title('day11')

    axs[0,1].plot(I,id16,'g')
    axs[0,1].set_title('day16')

    axs[1,1].plot(I,id21,'g')
    axs[1,1].set_title('day21')

    fig1.suptitle('The relationship between FW and I')
    fig1.supxlabel('I($\mu mol\cdot s^{-1}\cdot m^{-2}$)')
    fig1.supylabel('FW(g)')

    fig2, axs = plt.subplots(2, 2,sharex=True,sharey=True)

    axs[0,0].plot(T,td6,'r')
    # axs[0, 0].set_xlim(0,10)
    # axs[0, 0].set_ylim(0,10)
    axs[0,0].set_title('day6')

    axs[1,0].plot(T,td11,'r')
    # axs[1, 0].set_xlim(0,10)
    # axs[1, 0].set_ylim(0,10)
    axs[1,0].set_title('day11')

    axs[0,1].plot(T,td16,'r')
    axs[0,1].set_title('day16')

    axs[1,1].plot(T,td21,'r')
    axs[1,1].set_title('day21')

    fig2.suptitle('The relationship between FW and T')
    fig2.supxlabel('T($ /{ }^{\circ}$C)')
    fig2.supylabel('FW(g)')


    plt.show()


def plot_RTE():
    tl = 15
    to = 24
    tu = 30

    def spline(t):
        y = lambda x, a, b, c, d: a * x ** 3 + b * x ** 2 + c * x + d
        x0 = [15, 24, 30]
        y0 = [0, 1, 0]
        popt, pcov = curve_fit(y, x0, y0)
        print(popt)
        return y(t, *popt)

    def func(t):
        if tl <= t <= tu:
            return spline(t)
        else:
            return 0

    x = np.arange(0, 50, 0.5)
    my_func = np.frompyfunc(func,1,1)
    y = my_func(x)
    # x1 = []
    # x2 = []
    # y1 = []
    # y2 = []
    # for i in range(len(x)):
    #     if tl <= x[i] <= tu:
    #         x1.append(x[i])
    #         y1.append(func(x[i]))
    #     else:
    #         x2.append(x[i])
    #         y2.append(func(x[i]))

    # sns.regplot(x1, y1, order=3, scatter=None, color='g')
    plt.plot(x, y, 'g')

    plt.scatter([15, 22.5, 30], [0, 1, 0], c='r')
    plt.ylim(-0.2, 1.6)
    plt.xlabel('T(°C)')
    plt.ylabel('RTE')
    plt.show()



def plot_reg():
    plt.rc('font', family='Times New Roman', size=15)
    x = np.linspace(0, 250,2000)
    t = x/a0

    def func3(x):
        return 372 / (1 + 83 * np.exp(-0.053 * x))
    def func4(x):
        return 372 / (1 + 114 * np.exp(-0.047 * x))


    sns.scatterplot(xdata / a0, ydata * 1.9)
    plt.plot(t, (func1(x) + func2(x)) * 2, 'g')
    # plt.fill_between(t, func3(x),func4(x),
    #                  alpha=0.3,color='g')
    plt.fill_between(t,func3(x),func4(x),
                     alpha=0.3,color='g')
    # plt.fill_between(t,(func1(x) + func2(x)) * 1.85+0.006*x**1.5,(func1(x) + func2(x)) * 2.15-0.006*x**1.5,
    #                  alpha=0.3,color='g')
    plt.xlabel('t ($days$)')
    plt.ylabel('FW($g$)')
    plt.show()

# plot_reg()
def plot_PAR():
    plt.rc('font', family='Times New Roman', size=12)

    def func1(x):
        if 0 <= x % 24 <= 16:
            return 200.0
        else:
            return 0

    def func2(x):
        return 18

    x = np.linspace(0, 96, 2000)
    f_ufunc1 = np.frompyfunc(func1, 1, 1)
    y1 = f_ufunc1(x)
    f_ufunc2 = np.frompyfunc(func2, 1, 1)
    y2 = f_ufunc2(x)

    # y = np.arange(0.0, 2, 0.01)
    # x1 = np.sin(2 * np.pi * y)
    # x2 = 1.2 * np.sin(4 * np.pi * y)

    # plt.rcParams['font.family'] = 'simhei'
    # plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(18, 6))
    # 参数y2 默认值为0
    ax1.plot(x, y1, 'g')
    ax1.set_title('PARt')
    # 参数y2 的值为标量1
    ax2.plot(x, y2, 'g')
    ax2.set_title('PARh')
    ax3.plot(x, y1 - y2, 'g')
    ax3.set_title('PAR')
    plt.ylim(-20, 210)
    plt.show()


# plot_PAR()


def q2():
    k = 0.51 * 5.070303144

    def func(t):
        return 372 / (1 + 114 * np.exp(-0.51 * 5.070303144 * t))

    def F(t):
        return (-372 / 114) * np.log(np.exp(-k * t) / (1 + np.exp(-k * t)))

    def obj(t):
        t0, t1 = t
        return (func(t1) / t1) * (t0 * t0 / (F(t0) - F(0)) + ((t1 - t0) * t1) / (F(t1) - F(0)))

    # y = Function('y')
    # eq = diff(y(x), x, 1) - func(x)  # 定义方程
    # con = {y(0): 0, diff(y(x), x).subs(x, 0): 372}  # 定义初值条件
    # y = dsolve(eq, ics=con)
    # print(simplify(y))

    LB = [0, 0]
    UB = [500, 500]
    bound = tuple(zip(LB, UB))  # 生成决策向量界限的元组
    res = minimize(obj, [10, 30], bounds=bound)  # 第2个参数为初值
    print(res.fun, '\n', res.success, '\n', res.x)


def plot_pso():
    df = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\数据\\k5.xlsx")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # # 构建数据
    # X = []
    # Y = []
    # Z =
    # for i in range(len(df.iloc[0])):
    #     for j in range(len(df.iloc[:,0])):
    X = np.arange(0,len(df.iloc[0]))
    Y = np.arange(0,len(df.iloc[:,0]))
    Z=df.to_numpy()
    Z = np.array(Z)
    #         Z.append(df.iloc[])
    X, Y = np.meshgrid(X, Y)
    # 绘制曲面图
    # 绘制使用冷暖色图着色的 3D 表面。通过使用 antialiased=False 使表面变得不透明。
    surf = ax.plot_surface(X,Y,Z,cmap='Greens',
                           linewidth=0, antialiased=False)

    # 定制z轴
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # 添加一个颜色条形图展示颜色区间
    fig.colorbar(surf, shrink=0.7, aspect=3)
    plt.show()


def relative_root_mean_squared_error(true, pred):
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(pred))
    squared_error = num / den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss


# rrmse = relative_root_mean_squared_error(ydata,func1(xdata)+func2(xdata))
# print(rrmse)

# Huber loss function
def huber_loss(y_pred, y, delta=1.0):
    huber_mse = 0.5 * (y - y_pred) ** 2
    huber_mae = delta * (np.abs(y - y_pred) - 0.5 * delta)
    return np.where(np.abs(y - y_pred) <= delta, huber_mse, huber_mae)




# print(5/6)
# # Plotting
# x_vals = xdata
#
# plt.plot(x_vals, huber_loss(ydata,func1(xdata)+func2(xdata),delta=1), "green")
# plt.grid(True, which="major")
# plt.show()

