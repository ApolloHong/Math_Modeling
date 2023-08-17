import matplotlib.pyplot as plt
import numpy as np

# 设置标题
plt.title('Title')

# x坐标的间隔设置和文字设置
N = 13
ind = np.arange(N)  # [ 0 1 2 3 4 5 6 7 8 9 10 11 12]
plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13'))
plt.xlabel('name of X')

# y坐标的文字设置和间隔设置
plt.ylabel('name of Y')
plt.yticks(np.arange(0, 201, 20))  # 0到201 间隔20

# 输入数据
A = (52, 49, 48, 47, 44, 43, 41, 41, 40, 38, 36, 31, 29)
B = (38, 40, 45, 42, 48, 51, 53, 54, 57, 59, 57, 64, 62)
C = (38, 24, 20, 23, 8, 47, 32, 44, 20, 28, 40, 5, 66)

# 对前两个数据进行求和,为后续绘制p3做准备
d = []
for i in range(0, N):
    sum = A[i] + B[i]
    d.append(sum)

width = 0.35  # 设置条形图一个长条的宽度
p1 = plt.bar(ind, A, width, color='blue')  # 绘制A中的数据
p2 = plt.bar(ind, B, width, bottom=A, color='green')  # 在p1的基础上绘制，底部数据就是p1的数据
p3 = plt.bar(ind, C, width, bottom=d, color='red')  # 在p1和p2的基础上绘制，底部数据就是p1和p2

# 绘制图例
plt.legend((p1[0], p2[0], p3[0]), ('A_name', 'B_name', 'C_name'), loc=2)  # loc指定了图例的位置，2表示左上角

plt.show()

import matplotlib.pyplot as plt
import numpy as np

index = np.arange(4)
data1 = np.array([1, 5, 6, 3])
data2 = np.array([1, 2, 1, 5])
data3 = np.array([4, 8, 9, 4])

a = 0.3
plt.title('multi bar chart')

plt.barh(index, data1, a, color='pink', label='a', hatch='/')
plt.barh(index, data2, a, left=data1,  # 堆叠在左边第一个上方
         color='c', label='b')
plt.barh(index, data3, a, left=(data1 + data2),  # 堆叠在左边第一个和第二个上方
         color='orange', alpha=0.5, label='c')

plt.legend()
plt.show()
