# 可能需要根据实际情况更改的部分和注意事项：
# 1.本代码只考虑了河流污染物处于稳态的情况，水中的污染物分布状况也稳定的
# 2.数据输入部分，要有相应的格式,我们直接才用我们直接才用input的方式
# 3.可视化部分做了一维的
# 4.可视化部分的具体细节根据个人需求而定
################################################

import matplotlib.pyplot as plt
##导入库
import numpy as np

# 部分常量的定义
cz = 30  # 污水中某种污染物的浓度,单位为mg/L
c1 = 0.5  # 河流中某种污染物的本底浓度,单位为mg/L
u = 0.3  # 断面的平均流速,单位为m/s
q = 0.15  # 排入污水的流量,单位为m^3/s
Q = 5.5  # 河流流量,单位为m^3/s
K = 0.2  # 污染物的衰减速度常数,单位为/day
D = 10  # 纵向弥散系数,单位为m^2/s


def unidimensionalCompute(x):
    C = (c1 * Q + cz * q) / (Q + q)
    return C * np.exp(u * x / (2 * D) * (1 - np.sqrt(1 + 4 * K * D / (u * u))))  # 我们解出的位置与污染物浓度方程


def unidimensionalPlot():
    t1 = np.arange(0.0, 30.0, 0.1)  # 设置一个0到30的考察范围
    plt.plot(t1, unidimensionalCompute(t1), 'k')
    plt.title("The relation curve of pollutant concentration and position")
    plt.ylabel("concentration")
    plt.xlabel("x")
    plt.show()


def main():
    unidimensionalPlot()


if __name__ == "__main__":
    main()
