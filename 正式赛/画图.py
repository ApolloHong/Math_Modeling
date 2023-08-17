import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LinearLocator
from scipy.optimize import curve_fit

plt.rc('font', family='Times New Roman', size=15)
plt.style.use('seaborn')  # 是seaborn的暗色系
sns.set_theme(style="darkgrid")  # 改为黑暗模式

# dp = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\T.xls")
# temporature = dp['Min(C)']
# print(temporature)
P = [1,1.6,2.2,3.9,5.1,5.3,4.3,4.7,3.7,3.1,1.8,1.4]
month = ['January', 'February', 'March', 'April',
         'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
T_max = [40,45,57,67,76,86,90,89,80,68,54,44]
T_min = [22,26,36,46,57,67,72,70,61,49,36,27]
T_mean = [31.1,35.8,46.4,56.5,66.7,76.5,81.1,79.2,70.7,58.5,45.4,35.3]


lst_plant = ['Big Bluestem','Little Bluestem','Indiangrass',
             'Western Wheatgrass','Prairie Junegrass','Blue Grama']

def plot_T():
    tl = 15
    to = 24
    tu = 30

    def spline(t):
        y = lambda x, a, b, c: a * x ** 2 + b * x ** 1 + c
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

def plot_P():
    plt.rc('font', family='Times New Roman', size=15)
    # P_min = map(lambda a,b:a-b,P,3)
    # P_max = map(lambda a,b:a+b,P,2)
    P_min = [P[i]-0.5 for i in range(len(P))]
    P_max = [P[i]+0.5 for i in range(len(P))]

    data = np.row_stack((P_min, P, P_max))
    columns = month
    rows = ['max', 'mean', 'min']

    values = np.arange(0,11,2)
    value_increment = 1.5

    # Get some pastel shades for the colors
    colors = plt.cm.BuPu(np.linspace(0.05, 0.5, len(rows)))
    n_rows = len(data)

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.6

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
        y_offset = data[row]
        cell_text.append(['%1.1f' % (x / 1.0) for x in y_offset])
    # Reverse colors and text labels to display the last value at the top.
    colors = colors[::-1]
    cell_text.reverse()

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          rowColours=colors,
                          colLabels=columns,
                          loc='bottom')

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.25)

    plt.ylabel('Precipitation')
    # plt.yticks(values * value_increment, ['%d' % val for val in values])
    plt.yticks(values)
    plt.xticks([])
    # plt.title('')

    plt.show()


def plot_spline():
    plt.rc('font', family='Times New Roman', size=15)
    T_max = [40, 45, 57, 67, 76, 86, 90, 89, 80, 68, 54, 44]
    T_min = [22, 26, 36, 46, 57, 67, 72, 70, 61, 49, 36, 27]
    T_mean = [31.1, 35.8, 46.4, 56.5, 66.7, 76.5, 81.1, 79.2, 70.7, 58.5, 45.4, 35.3]
    plt.rc('font', family='Times New Roman', size=15)
    x0 = [i for i in range(len(T_mean))]
    y0 = T_mean
    def spline(t):
        y = lambda x, a, b, c: a * x ** 2 + b * x ** 1 + c
        popt, pcov = curve_fit(y, x0, y0)
        print(popt)
        return y(t, *popt)

    # x = np.arange(0,len(temporature), 0.5)
    # my_func = np.frompyfunc(spline,1,1)
    # y = my_func(x)
    # # sns.regplot(x, y, color = 'g')
    #
    # sns.regplot(x0, y0, color='g',order=2)
    # plt.xticks(month)
    # plt.legend()
    # # plt.ylim(-0.2, 1.6)
    # plt.xlabel('Month')
    # plt.ylabel('T(°F)')
    # plt.show()

    # T_mean = map(lambda a,b:a-b,T_mean,T_min)
    # T_max = map(lambda a,b:a-b,T_max,T_mean)
    # print(T_mean,T_max)
    data = np.row_stack((T_min,T_mean,T_max))

    columns = month
    rows = ['max','mean','min']

    values = np.arange(50,100,10)
    value_increment = 1.5

    # Get some pastel shades for the colors
    colors = plt.cm.OrRd(np.linspace(0.05, 0.5, len(rows)))
    n_rows = len(data)

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.6

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
        y_offset = data[row]
        cell_text.append(['%1.1f' % (x / 1.0) for x in y_offset])
    # Reverse colors and text labels to display the last value at the top.
    colors = colors[::-1]
    cell_text.reverse()

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          rowColours=colors,
                          colLabels=columns,
                          loc='bottom')

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.25)

    plt.ylabel('Temperature')
    # plt.yticks(values * value_increment, ['%d' % val for val in values])
    plt.yticks(values*3-110,values)
    plt.xticks([])
    # plt.title('')
    # plt.savefig('T')

    plt.show()


def plot_ftp1():
    plt.rc('font', family='Times New Roman', size=15)
    dp = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\FTP_month.xlsx",header = None)
    dp = dp/2
    # data1 = dp.iloc[0:3,:]
    # data2 = dp.iloc[3:7,:]
    # data = [dp.iloc[i] for i in range(12)]
    x = np.arange(0,12)
    # print(len(dp))
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(18, 6))
    # 参数y2 默认值为0
    ax1.plot(x, dp.iloc[0], 'g',label='1')
    ax1.plot(x, dp.iloc[1], 'r',label='2')
    ax1.plot(x, dp.iloc[2], 'b',label='3')
    ax1.scatter(x, dp.iloc[0],marker='X', color='g')
    ax1.scatter(x, dp.iloc[1],marker='+', color='r')
    ax1.scatter(x, dp.iloc[2],marker='o', color='b')
    # ax1.ylabel('F(T,P)')
    ax1.set_title('Plants number from 1 to 3')
    # 参数y2 的值为标量1
    ax2.plot(x, dp.iloc[3], 'g',label='4')
    ax2.plot(x, dp.iloc[4], 'r',label='5')
    ax2.plot(x, dp.iloc[5], 'b',label='6')
    ax2.scatter(x, dp.iloc[3],marker='X', color='g')
    ax2.scatter(x, dp.iloc[4],marker='+', color='r')
    ax2.scatter(x, dp.iloc[5],marker='o', color='b')
    # plt.legend()
    # plt.ylabel('F(T,P)')
    ax2.set_title('Plants number from 4 to 6')
    plt.xticks([i for i in range(12)],month)
    # plt.ylim(0,1.2)
    plt.show()

def plot_ftp2():
    dp = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\FTP-TP.xlsx",header = None)
    # df_ = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\FTP_TP_right.xlsx")
    # df=[]
    # # 读取多个sheet
    # sheets = ['Sheet1', 'Sheet2','Sheet3','Sheet4','Sheet5','Sheet6']
    # df = pd.read_excel('file.xlsx', sheet_name=sheets)
    # df.append(pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\FTP_TP_right.xlsx",
    #                         sheet_name='Sheet1',header=None))
    # df.append(pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\FTP_TP_right.xlsx",
    #                         sheet_name='Sheet2',header=None))
    # df.append(pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\FTP_TP_right.xlsx",
    #                         sheet_name='Sheet3',header=None))
    # df.append(pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\FTP_TP_right.xlsx",
    #                         sheet_name='Sheet4',header=None))
    # df.append(pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\FTP_TP_right.xlsx",
    #                         sheet_name='Sheet5',header=None))
    # df.append(pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\FTP_TP_right.xlsx",
    #                         sheet_name='Sheet6',header=None))
    # print(df[5])
    df = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\1.xlsx")

    for i in range(6):
        data = df.iloc[51*i:51*i+50,:]
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # # 构建数据
        X = np.linspace(30,90,50)
        # X = sorted(X,reverse=True)#T
        # X = X.toarray
        # X = np.sorted(reverse=True)
        Y = np.linspace(0.2,3.8,50)#P
        Z = data.to_numpy()
        Z = np.array(Z)
        print(Z)
        #         Z.append(df.iloc[])
        X, Y = np.meshgrid(X, Y)
        # 绘制曲面图
        # 绘制使用冷暖色图着色的 3D 表面。通过使用 antialiased=False 使表面变得不透明。
        surf = ax.plot_surface(X, Y, Z, cmap='coolwarm',
                               linewidth=0, antialiased=False)

        # 定制z轴
        # ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')

        # 添加一个颜色条形图展示颜色区间
        fig.colorbar(surf, shrink=0.7, aspect=7)
        plt.show()

def plot_4():
    plt.rc('font', family='Times New Roman', size=15)
    colors = ['#0cb9c1','#f85a40','#037ef3','#ffc845','#8c95c6','#f48924']
    dp1 = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\mar1.xlsx",header = None)
    dp2 = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\mar2.xlsx",header = None)
    dp3 = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\mar3.xlsx",header = None)
    dp4 = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\mar4.xlsx",header = None)
    dp5 = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\mar5.xlsx",header = None)
    dp6 = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\mar6.xlsx",header = None)
    x = np.arange(0,600,1)

    fig, (ax1, ax2, ax3,ax4, ax5, ax6) = plt.subplots(6, 1, sharex=True, figsize=(10, 8))
    # 参数y2 默认值为0
    ax1.plot(x, dp1.iloc[0,0:600], color=colors[0],label=4)
    # ax1.set_title(4)
    # ax1.set_title('Plants number from 1 to 3')
    # 参数y2 的值为标量1
    ax2.plot(x, dp2.iloc[0,0:600], color=colors[0],label=2)
    ax2.plot(x, dp2.iloc[1,0:600], color=colors[1],label=4)
    #
    ax3.plot(x, dp3.iloc[0,0:600], color=colors[0],label=2)
    ax3.plot(x, dp3.iloc[1,0:600], color=colors[1],label=4)
    ax3.plot(x, dp3.iloc[2,0:600], color=colors[2],label=5)
    #
    ax4.plot(x, dp4.iloc[0,0:600], color=colors[0],label=1)
    ax4.plot(x, dp4.iloc[1,0:600], color=colors[1],label=2)
    ax4.plot(x, dp4.iloc[2,0:600], color=colors[2],label=3)
    ax4.plot(x, dp4.iloc[3,0:600], color=colors[3],label=5)
    #
    ax5.plot(x, dp5.iloc[0,0:600], color=colors[0],label=1)
    ax5.plot(x, dp5.iloc[1,0:600], color=colors[1],label=2)
    ax5.plot(x, dp5.iloc[2,0:600], color=colors[2],label=3)
    ax5.plot(x, dp5.iloc[3,0:600], color=colors[3],label=5)
    ax5.plot(x, dp5.iloc[4,0:600], color=colors[4],label=6)
    #
    ax6.plot(x, dp6.iloc[0,0:600], color=colors[0],label=1)
    ax6.plot(x, dp6.iloc[1,0:600], color=colors[1],label=2)
    ax6.plot(x, dp6.iloc[2,0:600], color=colors[2],label=3)
    ax6.plot(x, dp6.iloc[3,0:600], color=colors[3],label=4)
    ax6.plot(x, dp6.iloc[4,0:600], color=colors[4],label=5)
    ax6.plot(x, dp6.iloc[5,0:600], color=colors[5],label=6)
    #
    # ax2.set_title('Plants number from 4 to 6')
    # plt.xticks([i for i in range(12)],month)
    # plt.vlines()
    # plt.tight_layout()
    # plt.legend(bbox_to_anchor=(0.95, 0.1), loc=10, borderaxespad=0)
    plt.xlabel('Time(Month)')
    plt.show()

def plot_impact():
    plt.rc('font', family='Times New Roman', size=15)
    dp = []
    colors = ['#0cb9c1','#f85a40','#037ef3','#ffc845','#8c95c6','#f48924']
    x = np.arange(0,501,1)
    dp.append(pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\mar_dahan.xlsx",header=None))
    dp.append(pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\mar_xiaohan.xlsx",header=None))
    dp.append(pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\mar_dare.xlsx",header=None))
    dp.append(pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\mar_xiaore.xlsx",header=None))
    for i in range(4):
        for j in range(6):
            plt.plot(x,dp[i].iloc[j],color=colors[j],label=f'{j+1}')
            plt.xlabel('Time(month)')
            plt.ylabel('$α(g/m^2)$')
        plt.legend()
        plt.show()

def plot_radiation():
    pass


def longterm():
    plt.rc('font', family='Times New Roman', size=15)
    colors = ['#0cb9c1', '#f85a40', '#037ef3', '#ffc845', '#8c95c6', '#f48924']
    df = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\mar_100yeara.xlsx",header=None)
    x = np.arange(0,len(df.iloc[0]))

    plt.plot(x, df.iloc[0], color=colors[0],label=1,linewidth=0.5)
    plt.plot(x, df.iloc[1], color=colors[1],label=2,linewidth=0.5)
    plt.plot(x, df.iloc[2], color=colors[2],label=3,linewidth=0.5)
    plt.plot(x, df.iloc[3], color=colors[3],label=4,linewidth=0.5)
    plt.plot(x, df.iloc[4], color=colors[4],label=5,linewidth=0.5)
    plt.plot(x, df.iloc[5], color=colors[5],label=6,linewidth=0.5)

    plt.show()

def plot_disaster():
    plt.rc('font', family='Times New Roman', size=15)
    colors = ['#0cb9c1', '#f85a40', '#037ef3', '#ffc845', '#8c95c6', '#f48924']
    df = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\mar_habi.xlsx",header=None)
    x = np.arange(0,len(df.iloc[0]))

    plt.plot(x, df.iloc[0], color=colors[0],label=1,linewidth=0.7)
    plt.plot(x, df.iloc[1], color=colors[1],label=2,linewidth=0.7)
    plt.plot(x, df.iloc[2], color=colors[2],label=3,linewidth=0.7)
    plt.plot(x, df.iloc[3], color=colors[3],label=4,linewidth=0.7)
    plt.plot(x, df.iloc[4], color=colors[4],label=5,linewidth=0.7)
    plt.plot(x, df.iloc[5], color=colors[5],label=6,linewidth=0.7)
    plt.ylabel('$α_{ab}$')
    plt.xlabel('Time(Month)')

    plt.show()


# def rmse():
#     T = np.vstack(T_mean)

def plot_cd():
    plt.rc('font', family='Times New Roman', size=15)
    colors = ['#0cb9c1', '#f85a40', '#037ef3', '#ffc845', '#8c95c6', '#f48924']
    df = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\mar_polu.xlsx",header=None)
    x = np.arange(0,len(df.iloc[0]))

    plt.plot(x, df.iloc[0], color=colors[0],label=1,linewidth=0.7)
    plt.plot(x, df.iloc[1], color=colors[1],label=2,linewidth=0.7)
    plt.plot(x, df.iloc[2], color=colors[2],label=3,linewidth=0.7)
    plt.plot(x, df.iloc[3], color=colors[3],label=4,linewidth=0.7)
    plt.plot(x, df.iloc[4], color=colors[4],label=5,linewidth=0.7)
    plt.plot(x, df.iloc[5], color=colors[5],label=6,linewidth=0.7)
    plt.ylabel('$α_{ab}$')
    plt.xlabel('Time(Month)')

    plt.show()


def pairplot():
    # penguins = sns.load_dataset("penguins")
    # sns.pairplot(penguins, hue="species")
    # plt.show()
    # plt.close()

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from tensorflow import keras

    dataset_path = keras.utils.get_file("auto-mpg.data",
                                        "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                              na_values="?", comment='\t',
                              sep=" ", skipinitialspace=True)  # #sep: 指定分割符；
    # skipinitialspace忽略分隔符后的空格
    dataset = raw_dataset.copy()
    dataset = dataset.dropna()  # 数据清洗
    train_dataset = dataset.sample(frac=0.8,
                                   random_state=0)  # frac取样比例， random_state如果值为int,
    # 则为随机数生成器或numpy RandomState对象设置种子
    test_dataset = dataset.drop(train_dataset.index)  # 取train_dataset的数据下标，并drop，形成test_dataset
    sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
    # 快速查看训练集中几对列的联合分布。
    plt.show()


    column_names = ['Big Bluestem','Little Bluestem','Indiangrass',
                    'Western Wheatgrass','Prairie Junegrass','Blue Grama']
    dataset = pd.read_excel()
    dataset = dataset.dropna()  # 数据清洗

def plot_phi():
    names = ['Big Bluestem','Little Bluestem','Indiangrass',
                    'Western Wheatgrass','Prairie Junegrass','Blue Grama']
    df = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\mat_habi.xlsx",header = None)
    data1 = pd.DataFrame(df.iloc[0:6])
    data1.columns = names
    data1.index = names
    data2 = pd.DataFrame(df.iloc[7:13])
    data2.columns = names
    data2.index = names
    # data = data1 + data2
    # print(data2,data1)
    # sns.heatmap(data1,cmap = 'Greens',vmin=-1.2,vmax=1.2)
    sns.heatmap(data2,cmap = 'Greens',vmin=-1.2,vmax=1.2)
    plt.show()

def plot_multi():
    plt.rc('font', family='Times New Roman', size=15)
    colors = ['#0cb9c1', '#f85a40', '#037ef3', '#ffc845', '#8c95c6', '#f48924']
    df = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\mar_Variation.xlsx",header=None)
    x = np.arange(0,len(df.iloc[0]))
    plt.plot(x, df.iloc[0], color=colors[2],linewidth=0.7,label='group 1')
    plt.plot(x, df.iloc[2], color=colors[3],linewidth=0.7,label='group 2')
    plt.ylabel('$α_{ab}$')
    plt.xlabel('Time(Month)')
    plt.legend()
    plt.show()


def plot_matrix():
    plt.rc('font', family='Times New Roman', size=15)
    colors = ['#0cb9c1', '#f85a40', '#037ef3', '#ffc845', '#8c95c6', '#f48924']
    data = []
    df = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\data_update.xlsx",header = None)
    x = np.arange(0,len(df.iloc[0]),1)
    plt.rcParams['figure.figsize'] = [12,12]
    fig, axs = plt.subplots(6, 6)

    for i in range(36):
        a = i//6
        b = i%6
        print(a,b)
        lst = df.iloc[i]
        lst = lst.to_numpy()
        if i%6 == i//6:
            axs[a,b].plot(lst,linewidth=0.1,color='g')
        else:
            axs[a,b].plot(lst,linewidth=0.2,color='g')
        # axs[a,b].set_axis_off()
    # axs[0, 0].plot(df.iloc[0],linewidth=0.1)
    # axs[0,0].set_axis_off()
    # axs[0,0].

    # axs[0,0].pcolor(Z1)
    # axs[0,0].set_xlim(0,10)
    # axs[0,0].set_ylim(0,10)
    #
    # axs[1,1].pcolor(Z2)
    # axs[1,1].set_xlim(0,10)
    # axs[1,1].set_ylim(0,10)
    #
    #
    # axs[0,1].pcolor(M1)
    # axs[1,1].pcolor(M2)


    fig.tight_layout()
    plt.show()


def sensitivity():
    names = ['-5%','-4%','-3%','-2%','-1%','0%','1%','2%','3%','4%','5%']
    df = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\data\\senana.xlsx",header = None)
    # df = df / 10
    df.columns = names
    df.index = names
    sns.heatmap(df,cmap='RdBu',annot = True,fmt='.2f')
    plt.xlabel('$T_0$')
    plt.ylabel('$P_0$')
    plt.show()

# def F_tp():
#     return np.std(P) , np.std(T_mean)
#
# print(F_tp())

# ----------------------------------------------
# rmse()
# plot_impact()
# plot_ftp1()
# plot_ftp2()
# plot_4()
# plot_P() #fig 1
# plot_spline() #fig 2
# plot_disaster()
# pairplot()
# plot_cd()
# plot_phi()
# plot_multi()
# plot_matrix()
sensitivity()


