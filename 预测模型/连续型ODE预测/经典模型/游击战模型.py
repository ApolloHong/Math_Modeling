# 可能需要根据实际情况更改的部分和注意事项：
# 1.本代码一共有两个部分，一个部分是正规战争，一个部分是游击战争
# 2.数据输入部分，要有相应的格式,我们直接才用我们直接才用input的方式
# 3.可视化部分比较简略
# 4.可视化部分的具体细节根据个人需求而定
# 5.我们的模型是按照最简单的情况来的，对于具体的题目来讲，本题目当中出现的常数必然是可以有其他的内容（函数）的添加
################################################

import matplotlib.pyplot as plt
##导入库
import numpy as np

# 部分常量的定义
x = [35, 30, 25, 20, 15, 10, 5]
y = 20
rx = 0.5
px = 0.6
ry = 0.75
py = 0.4
a = rx * px  # 乙方平均每个士兵对甲方士兵的杀伤率（单位时间的杀伤数）
b = ry * py  # 甲方平均每个士兵对乙方士兵的杀伤率（单位时间的杀伤数）


# regular warfare(正规战争，不考虑后勤人员之类的，只关注双方的人数和战斗力)
# 本代码中只通过调整具体的双方人数来作图，当然你也可以通过改变a,b来作图

def unidimensionalCompute(x0, y0):
    k = a * y0 * y0 - b * x0 * x0
    return k  # 根据双方的初始人数来得到我们的常量k，k=0则我们双方战成平局，k>0则甲方胜，k<0则乙方胜


def getY(i, k):
    return np.sqrt((abs(k) + a * i * i) / b)


def getX(i, k):
    return np.sqrt((abs(k) + a * i * i) / b)


def unidimensionalPlot():
    for s in x:
        if s >= y:
            k = unidimensionalCompute(s, y)
            t1 = np.arange(0.0, 20.0, 0.05)  # 设置一个0到30的考察范围
            plt.plot(t1, getY(t1, k), 'k')
        else:
            k = unidimensionalCompute(s, y)
            t1 = np.arange(0.0, 20.0, 0.05)
            plt.plot(getX(t1, k), t1, 'k')
    plt.title("regular warfare model")
    plt.ylabel("y(t)")
    plt.xlabel("x(t)")
    plt.show()


def main1():
    unidimensionalPlot()


# if __name__ == "__main__":
#  main1()
# 下述代码是游击战
###############
Sry = 0.1  # 乙方一次射击的有效面积
Srx = 0.2  # 乙方一次射击的有效面积
Sx = 0.15  # 甲方活动面积
Sy = 0.10  # 乙方的活动面积
c = ry * Sry / Sx  # ry为命中率，本来该使用命中率，我们游击战中命中率为有效面积Sry与甲方活动面积之比Sx,同理甲方也是一样
d = rx * Srx / Sy  # 按照c站在甲方的角度理解即可


def ReturnM(x0, y0):
    return c * y0 - d * x0;


def returnY(m, x1):
    return (m + d * x1) / c


def Plot():
    for s in x:
        m = ReturnM(s, y)
        t1 = np.arange(0.0, 20.0, 0.05)  # 设置一个0到30的考察范围
        plt.plot(t1, returnY(m, t1), 'k')
    plt.title("guerrilla warfare model")
    plt.ylabel("y(t)")
    plt.xlabel("x(t)")
    plt.show()


def main2():
    Plot()


if __name__ == "__main__":
    main1()  # 注意如过想看正规战争的话，需要把这里的main2改为main1
