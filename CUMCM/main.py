import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d

plt.style.use('seaborn')  # Set the theme of seaborn


def GerOis(nc):
    d = 13
    rc = lambda i: 87+d*i
    n = lambda i:np.int(np.floor(2*np.pi*rc(i)/d))
    n_totale = np.sum(n(i+1) for i in range(nc+1))

    X = [np.float(rc(i)*np.cos((d*j)/rc(i))) for i in range(1,nc+1) for j in range(1, n(i)+1)]
    Y = [np.float(rc(i) * np.sin((d * j) / rc(i))) for i in range(1, nc + 1) for j in range(1, n(i) + 1)]

    return n_totale, X, Y

def Plotexcel1():
    plt.rc("font", family='FangSong')
    num_month = 12
    num_month = num_month+1
    ans = ['平均光学效率','平均余弦效率','平均阴影遮挡效率','平均截断效率','单位面积输出功率']
    dataFunction1 = [0.5469, 0.5671,0.5932,0.6182,0.6317,0.6363,
                     0.6316,0.6173,0.5915,0.5644,0.5457,0.5386]
    dataFunction2 = [0.7459,0.7743,0.8115,0.8499,0.8727,0.8799,
                     0.8725,0.8483,0.8095,0.7702,0.7440,0.7367]
    dataFunction3 = [0.9269,0.9274,0.9278,0.9264,0.9239,0.9237,
                     0.9239,0.9265,0.9276,0.9280,0.9275,0.9239]
    dataFunction4 = [0.8800,0.8800,0.8800,0.8800,0.8800,0.8800,
                     0.8800,0.8800,0.8800,0.8800,0.8800,0.8800]
    dataFunction5 = [0.4758,0.5348,0.5903,0.6365,0.6603,0.6679,
                     0.6600,0.6348,0.5874,0.5278,0.4706,0.4464]


    dataNumeric1 = [0.5668, 0.5904,0.6206,0.6517,0.6701,0.6761,
                    0.6699, 0.6504,0.6188,0.5872,0.5657,0.5585]
    dataNumeric2 = [0.7459,0.7743,0.8115,0.8499,0.8727,0.8799,
                    0.8725,0.8483,0.8095,0.7702,0.7440,0.7367]
    dataNumeric3 = [0.9791,0.9826,0.9834,0.9850,0.9851,0.9853,
                    0.9850,0.9849,0.9837,0.9824,0.9803,0.9765]
    dataNumeric4 = [0.8800,0.8800,0.8800,0.8800,0.8800,0.8800,
                    0.8800,0.8800,0.8800,0.8800,0.8800,0.8800]
    dataNumeric5 = [0.4932,0.5569,0.6176,0.6709,0.7004,0.7096,
                    0.7001,0.6688,0.6144,0.5492,0.4880,0.4630]

    fig = plt.figure(figsize=(15, 7))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.1)

    ax1 = fig.add_subplot(121)
    line1, = ax1.plot(np.arange(1, num_month), dataFunction2, 'g', alpha=0.5, lw=1.5, ls='-')
    ax1.scatter(np.arange(1, num_month), dataFunction2, color='g', alpha=0.5, lw=1.5, ls='-', marker='^')

    ax2 = ax1.twinx()
    line2, = ax2.plot(np.arange(1, num_month), dataFunction3, 'b', alpha=0.5, lw=1.5, ls='-', marker='+')
    ax2.scatter(np.arange(1, num_month), dataFunction3, color='b', alpha=0.5, lw=1.5, ls='-', marker='+')

    # ax3 = ax1.twinx()
    # line3, = ax3.plot(np.arange(1, num_month), dataFunction3, 'g', alpha=0.5, lw=1.5, ls='-')
    # ax3.scatter(np.arange(1, num_month), dataFunction3, color='g', alpha=0.5, lw=1.5, ls='-', marker='^')
    # ax3.legend('I')
    #
    # ax4 = ax1.twinx()
    # line4, = ax2.plot(np.arange(1, num_month), dataFunction4, 'b', alpha=0.5, lw=1.5, ls='-', marker='+')
    # ax4.scatter(np.arange(1,num_month), dataFunction4, color='b', alpha=0.5, lw=1.5, ls='-', marker='+')
    # ax4.legend('I')
    #
    # ax5 = ax1.twinx()
    # line5, = ax5.plot(np.arange(1, num_month), dataFunction5, 'g', alpha=0.5, lw=1.5, ls='-')
    # ax5.scatter(np.arange(1, num_month), dataFunction5, color='g', alpha=0.5, lw=1.5, ls='-', marker='^')
    # ax5.legend('I')

    plt.legend((line1, line2), [ans[1], ans[2]], loc=0,
               title='变量名', ncol=1, markerfirst=False,
               numpoints=2, frameon=True, fancybox=True,
               facecolor='#d3d3da', edgecolor='b', shadow=True)

    ax1.set_xlabel('月份')
    ax1.set_ylabel('使用方程解出的结果')

    # match the colors
    for tl in ax1.get_yticklabels():
        tl.set_color("g")
    for tl in ax2.get_yticklabels():
        tl.set_color("b")
    # for tl in ax3.get_yticklabels():
    #     tl.set_color("darkorange")
    # for tl in ax4.get_yticklabels():
    #     tl.set_color("y")
    # for tl in ax5.get_yticklabels():
    #     tl.set_color("r")

    ax6 = fig.add_subplot(122)
    line6, = ax6.plot(np.arange(1, num_month), dataNumeric2, 'g', alpha=0.5, lw=1.5, ls='-')
    ax6.scatter(np.arange(1, num_month), dataNumeric2, color='g', alpha=0.5, lw=1.5, ls='-', marker='^')

    ax7 = ax6.twinx()
    line7, = ax7.plot(np.arange(1, num_month), dataNumeric3, 'b', alpha=0.5, lw=1.5, ls='-', marker='+')
    ax7.scatter(np.arange(1, num_month), dataNumeric3, color='b', alpha=0.5, lw=1.5, ls='-', marker='+')

    # ax8 = ax6.twinx()
    # line8, = ax8.plot(np.arange(1, num_month), dataNumeric3, 'g', alpha=0.5, lw=1.5, ls='-')
    # ax8.scatter(np.arange(1, num_month), dataNumeric3, color='g', alpha=0.5, lw=1.5, ls='-', marker='^')
    # ax8.legend('I')
    #
    # ax9= ax6.twinx()
    # line9, = ax9.plot(np.arange(1, num_month), dataNumeric4, 'b', alpha=0.5, lw=1.5, ls='-', marker='+')
    # ax9.scatter(np.arange(1,num_month), dataNumeric4, color='b', alpha=0.5, lw=1.5, ls='-', marker='+')
    # ax9.legend('I')
    #
    # ax10 = ax6.twinx()
    # line10, = ax10.plot(np.arange(1, num_month), dataNumeric5, 'g', alpha=0.5, lw=1.5, ls='-')
    # ax10.scatter(np.arange(1, num_month), dataNumeric5, color='g', alpha=0.5, lw=1.5, ls='-', marker='^')
    # ax10.legend('I')

    plt.legend((line6, line7), [ans[1], ans[2]], loc=0,
               title='变量名', ncol=1, markerfirst=False,
               numpoints=2, frameon=True, fancybox=True,
               facecolor='#d3d3da', edgecolor='b', shadow=True)

    ax6.set_xlabel('月份')
    ax6.set_ylabel('使用解析解解出的结果')

    # match the colors
    for tl in ax6.get_yticklabels():
        tl.set_color("g")
    for tl in ax7.get_yticklabels():
        tl.set_color("b")
    # for tl in ax8.get_yticklabels():
    #     tl.set_color("darkorange")
    # for tl in ax9.get_yticklabels():
    #     tl.set_color("y")
    # for tl in ax10.get_yticklabels():
    #     tl.set_color("r")

    plt.savefig('问题一结果随着月份的变化.png')
    plt.show()
    plt.close(fig=fig)


def Plotdist():

    data = np.loadtxt('pathWrongTruncN.txt')

    fig = plt.figure()
    sns.distplot(a=data)
    plt.show()
    plt.close(fig=fig)

def PlotT1():
    data = pd.read_excel('fujian.xlsx')
    print(data)

    fig = plt.figure(figsize=(8,8))
    plt.scatter(data['x坐标 (m)'], data['y坐标 (m)'])
    plt.show()
    plt.close(fig=fig)

def PlotSave(nc):
    n_totale, X, Y = GerOis(nc=nc)
    print(n_totale)
    X = np.asarray(X).reshape((-1,1))
    Y = np.asarray(Y).reshape((-1,1))

    fig = plt.figure(figsize=(8,8))
    plt.scatter(X, Y)
    plt.savefig('T2with'+str(nc)+'.png')
    plt.show()
    plt.close(fig=fig)

def PlotVolonie():
    # rng = np.random.default_rng()
    # points = rng.random((10, 2))

    # data = pd.read_excel('fujian.xlsx')

    # n_totale, X, Y = GerOis(nc=1)
    # print(n_totale)
    # X = np.asarray(X).reshape((-1,1))
    # Y = np.asarray(Y).reshape((-1,1))
    points = [[_,__] for _ in range(5) for __ in range(5)]

    vor = Voronoi(points=points)

    fig = voronoi_plot_2d(vor)
    plt.show()

def PlotT3Volonoi():
    position = pd.read_excel('T3Position.xlsx', header=None, index_col=None)
    index = pd.read_excel('T3Classify.xlsx', header=None, index_col=None)
    position = np.asarray(position)
    index = np.asarray(index)
    print(position.shape,index.shape)

    # index_enumerate = [[i, index.iloc[i,:]] for i in range(len(index))]
    # print(index_enumerate)
    index1 = []
    index2 = []
    index3 = []
    index4 = []
    index5 = []
    for i in range(index.shape[0]):
        print(index[i,:])
        if index[i,:] == 1:
            index1.append(np.asarray(position[i,0:2]))
        elif index[i,:] == 2:
            index2.append(np.asarray(position[i,0:2]))
        elif index[i,:] == 3:
            index3.append(np.asarray(position[i,0:2]))
        elif index[i,:] == 4:
            index4.append(np.asarray(position[i,0:2]))
        elif index[i,:] == 5:
            index5.append(np.asarray(position[i,0:2]))
        else:
            pass

    index1 = np.asarray(index1)
    index2 = np.asarray(index2)
    index3 = np.asarray(index3)
    index4 = np.asarray(index4)
    index5 = np.asarray(index5)
    vor1 = Voronoi(points=index1)
    vor2 = Voronoi(points=index2)
    vor3 = Voronoi(points=index3)
    vor4 = Voronoi(points=index4)
    vor5 = Voronoi(points=index5)

    # fig = voronoi_plot_2d(vor1)
    # fig = voronoi_plot_2d(vor2)
    fig = voronoi_plot_2d(vor3)
    # fig = voronoi_plot_2d(vor4)
    # fig = voronoi_plot_2d(vor5)
    plt.show()
    plt.close(fig=fig)


    print(index1)

def PlotT3Fit():
    plt.rc("font", family='FangSong')
    fit_func = [0.5103,0.5191,0.5236,0.5273,0.5296,0.5333,0.5365,
                0.5394,0.5414,0.5423,0.5433,0.5447,0.5453,0.5462,
                0.5471,0.5480,0.5491,0.5500,0.5507,0.5511,0.5519,
                0.5523,0.5529,0.5537,0.5541,0.5549,0.5556,0.5561,
                0.5563,0.5565,0.5567,0.5567,0.55681,0.55685,0.55686,
                0.55686,0.55687,0.55687,0.55687,0.55689,0.5569]

    fig = plt.figure()
    plt.plot(fit_func,'bo-',alpha=0.4)
    plt.xlabel('迭代次数')
    plt.ylabel('问题3适应度函数')
    plt.show()
    plt.close(fig=fig)

def PlotT2Fit():
    plt.rc("font", family='FangSong')
    fit_func = [0.5004,0.5102,0.5123,0.5148,0.5196,0.5213,0.5243,
                0.5264,0.5284,0.5299,0.5312,0.5325,0.5353,0.5362,
                0.5371,0.5380,0.5391,0.5400,0.5407,0.5411,0.5419,
                0.5423,0.5429,0.5437,0.5441,0.5449,0.5466,0.5472,
                0.5489,0.5494,0.5499,0.5517,0.55201,0.55237,0.55298,
                0.5533,0.5536,0.5539,0.5542,0.5548,0.5550,0.5551,
                0.55511,0.55512,0.55518,0.55518,0.55518,0.55519,0.5552]

    fig = plt.figure()
    plt.plot(fit_func,'bo-',alpha=0.4)
    plt.xlabel('迭代次数')
    plt.ylabel('问题2适应度函数')
    plt.show()
    plt.close(fig=fig)

def PlotVoro():
    data = pd.read_excel('plotVoronoi.xlsx',header=None)
    # print(data)
    data1 = data[data[0]==1]
    data2 = data[data[0]==2]
    data3 = data[data[0]==3]
    data4 = data[data[0]==4]
    data5 = data[data[0]==5]


    fig = plt.figure(figsize=(8,8))
    print(data1.iloc[:,1])
    plt.scatter(data1.iloc[:,1],data1.iloc[:,2], alpha=0.4)
    plt.scatter(data2.iloc[:,1],data2.iloc[:,2], alpha=0.4)
    plt.scatter(data3.iloc[:,1],data3.iloc[:,2], alpha=0.4)
    plt.scatter(data4.iloc[:,1],data4.iloc[:,2], alpha=0.4)
    plt.scatter(data5.iloc[:,1],data5.iloc[:,2], alpha=0.4)
    plt.show()
    plt.close(fig=fig)

def getmean():
    # 3%
    ans1 = [0.5392,0.5572,0.7531,0.5875,0.5935,
            0.5953,0.5936,0.5870,0.5743,0.5548,
            0.5378,0.5316]
    ans2 = [0.7086,0.7306,0.7531,0.7734,0.7850,
            0.7887, 0.7849,0.7726, 0.7519,0.7278,
            0.6992]
    ans3 = [0.9888,0.9913,0.9920, 0.9868,0.9819,
            0.9798,0.9821,0.9871,0.9923,0.9909,
            0.9891,0.9875]
    ans4 = [0.8544,0.8544,0.8544,0.8544,0.8544,
            0.8544,0.8544]
    ans5 = [0.4601,0,.5152,
            0.6126,0.6082,0.5919,0.55900,0.5088,
            0.4300,0.4322]
    def xiaoshu(ans):
        f = np.mean(ans)
        print('%.4f'%f)

    xiaoshu(ans1)
    xiaoshu(ans2)
    xiaoshu(ans3)
    xiaoshu(ans4)
    xiaoshu(ans5)
    # print(np.mean(ans1),np.mean(ans2),np.mean(ans3),np.mean(ans4),np.mean(ans5))


if __name__ == '__main__':
    # print([_**__ for _ in range(3) for __ in [10, 11, 111]])
    PlotVoro()
