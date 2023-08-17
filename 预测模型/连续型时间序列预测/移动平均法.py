import pandas as pd
import pylab as pl
import seaborn as sns

sns.set_theme(style="darkgrid")  # 改为黑暗模式

df = pd.read_excel("C:\\Users\\xiaohong\\Desktop\\MCM小组\\数据\\b55f057df6.xls", header=None)
# original
x = df.iloc[0]
y = df.iloc[1]
sns.regplot(x=x, y=y, ci=100, order=1)
pl.show()


# n = len(x)
# print(n)

def one_mean_shift(x, y, n):
    # x = list(x[:])
    fx_one = list(y[0:n])
    sum = 0
    start = 0
    end = n
    for i in range(0, len(y) - n):
        sum = 0
        for j in range(start, end):
            sum += fx_one[j]
        start += 1
        end += 1
        fx_one.append(sum / n)
        # x.append(x[-1]+10) # 仅仅是此种情况要把年份加10
    return fx_one


def two_mean_shift(x, y, n):
    fx_two = list(y[0:n])
    M1 = one_mean_shift(x, y, n)
    M2 = one_mean_shift(x, M1, n)
    # 取周期T
    T = 4
    a = 2 * M1[len(M1) - 1] - M2[len(M2) - 1]
    b = (2 / (n - 1)) * (M1[len(M1) - 1] - M2[len(M2) - 1])
    print(M2)
    print("a:", a, "b:", b)
    # 计算 X （预测值）
    X = a + b * T
    return X


list_y = []


# 输入x为预测集、n为时间窗口、w为设置权重,m为预测时间
def weighting_shifts(x, n, w, m):
    num = 0
    sum = 0
    for i in range(n):
        num = w[i] + num
        sum = w[i] * x[m - i - 2] + sum
    y = sum / num
    return y


# for i in range(6,16):
#     list_y.append(weighting_shifts(y,5,w,i))
# y=y[5:15]

def mean_shift(list_y, y):
    sum1 = 0
    sum2 = 0
    y = list(y)
    for i in range(len(list_y)):
        sum1 = sum1 + list_y[i]
        sum2 = sum2 + y[i]
    error_mean = (1 - sum1 / sum2)
    return error_mean

# mean_shift(list_y,y)


# 有点小bug
# #
# fx_one = one_mean_shift(x,y,20)
# # print(fx_one)
# sns.regplot(x=x,y=fx_one,order=2,ci=100)
# pl.show()
#
# #
# X = two_mean_shift(x,y,20)
# print(X)
# # sns.regplot(x=x,y=X,order=2,ci=100)
# # pl.show()
