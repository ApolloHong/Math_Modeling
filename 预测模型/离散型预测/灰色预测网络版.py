# 在这里测试GM(1,1)模型
import math as mt

import matplotlib.pyplot as plt
import numpy as np

# 1.我们有一项初始序列X0
X0 = [15, 16.1, 17.3, 18.4, 18.7, 19.6, 19.9, 21.3, 22.5]

# 2.我们对其进行一次累加
X1 = [15]
add = X0[0] + X0[1]
X1.append(add)
i = 2
while i < len(X0):
    add = add + X0[i]
    X1.append(add)
    i += 1
print("X1", X1)

# 3.获得紧邻均值序列
Z = []
j = 1
while j < len(X1):
    num = (X1[j] + X1[j - 1]) / 2
    Z.append(num)
    j = j + 1
print("Z", Z)

# 4.最小二乘法计算
Y = []
x_i = 0
while x_i < len(X0) - 1:
    x_i += 1
    Y.append(X0[x_i])
Y = np.mat(Y).T
Y.reshape(-1, 1)
print("Y", Y)

B = []
b = 0
while b < len(Z):
    B.append(-Z[b])
    b += 1
print("B:", B)
B = np.mat(B)
B.reshape(-1, 1)
B = B.T
c = np.ones((len(B), 1))
B = np.hstack((B, c))
print("c", c)
print("b", B)

# 5.我们可以求出我们的参数
theat = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y)
a = theat[0][0]
b = theat[1]
did = 15.41559771 / -0.04352981
print(theat)
print(type(theat))

# 6.生成我们的预测模型
F = []
F.append(X0[0])
k = 1
while k < len(X0):
    F.append((X0[0] - did) * mt.exp(-a * k) + did)
    k += 1
print("F", F)

# 7.两者做差得到预测序列
G = []
G.append(X0[0])
g = 1
while g < len(X0):
    print(g)
    G.append(F[g] - F[g - 1])
    g += 1
print(F)

r = range(9)
t = list(r)

X0 = np.array(X0)
G = np.array(G)
e = X0 - G;
q = e / X0;  # 相对误差
s1 = np.var(X0)  # 方差
s2 = np.var(e)

c = s2 / s1  # 方差的比值

p = 0  # 小误差概率

for s in range(len(e)):
    if (abs(e[s]) < 0.6745 * s1):
        p = p + 1;
P = p / len(e)
print(c)
print(P)

plt.plot(t, X0, color='r', linestyle="--", label='true')
plt.plot(t, G, color='b', linestyle="--", label="predict")
plt.legend(loc='upper right')
plt.show()

## GM(1,N)

import numpy as np
import math as mt
import matplotlib.pyplot as plt

# 1.这里我们将 a 作为我们的特征序列 x0,x1,x2,x3作为我们的相关因素序列
a = [560823, 542386, 604834, 591248, 583031, 640636, 575688, 689637, 570790, 519574, 614677]
x0 = [104, 101.8, 105.8, 111.5, 115.97, 120.03, 113.3, 116.4, 105.1, 83.4, 73.3]
x1 = [135.6, 140.2, 140.1, 146.9, 144, 143, 133.3, 135.7, 125.8, 98.5, 99.8]
x2 = [131.6, 135.5, 142.6, 143.2, 142.2, 138.4, 138.4, 135, 122.5, 87.2, 96.5]
x3 = [54.2, 54.9, 54.8, 56.3, 54.5, 54.6, 54.9, 54.8, 49.3, 41.5, 48.9]


# 2.我们对其进行一次累加
def AGO(m):
    m_ago = [m[0]]
    add = m[0] + m[1]
    m_ago.append(add)
    i = 2
    while i < len(m):
        # print("a[",i,"]",a[i])
        add = add + m[i]
        # print("->",add)
        m_ago.append(add)
        i += 1
    return m_ago


a_ago = AGO(a)
x0_ago = AGO(x0)

x1_ago = AGO(x1)
x2_ago = AGO(x2)
x3_ago = AGO(x3)

xi = np.array([x0_ago, x1_ago, x2_ago, x3_ago])
print("xi", xi)


# 3.紧邻均值生成序列
def JingLing(m):
    Z = []
    j = 1
    while j < len(m):
        num = (m[j] + m[j - 1]) / 2
        Z.append(num)
        j = j + 1
    return Z


Z = JingLing(a_ago)
# print(Z)

# 4.求我们相关参数
Y = []
x_i = 0
while x_i < len(a) - 1:
    x_i += 1
    Y.append(a[x_i])
Y = np.mat(Y).T
Y.reshape(-1, 1)
print("Y.shape:", Y.shape)

B = []
b = 0
while b < len(Z):
    B.append(-Z[b])
    b += 1
B = np.mat(B)
B.reshape(-1, 1)
B = B.T
print("B.shape:", B.shape)
X = xi[:, 1:].T
print("X.shape:", X.shape)
B = np.hstack((B, X))
print("B-final:", B.shape)

# 可以求出我们的参数
theat = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y)
# print(theat)
al = theat[:1, :]
al = float(al)
# print("jhjhkjhjk",float(al))
b = theat[1:, :].T
print(b)
print("b.shape:", b.shape)
b = list(np.array(b).flatten())

# 6.生成我们的预测模型
U = []
k = 0
i = 0
# 计算驱动值
for k in range(11):
    sum1 = 0
    for i in range(4):
        sum1 += b[i] * xi[i][k]
        print("第", i, "行", "第", k, '列', xi[i][k])
        i += 1
    print(sum1)
    U.append(sum1)
    k += 1
print("U:", U)

# 计算完整公式的值
F = []
F.append(a[0])

f = 1
while f < len(a):
    F.append((a[0] - U[f - 1] / al) / mt.exp(al * f) + U[f - 1] / al)
    f += 1
print("F", )

# 做差序列
G = []
G.append(a[0])
g = 1
while g < len(a):
    G.append(F[g] - F[g - 1])
    g += 1
print("G:", G)

r = range(11)
t = list(r)

plt.plot(t, a, color='r', linestyle="--", label='true')
plt.plot(t, G, color='b', linestyle="--", label="predict")
plt.legend(loc='upper right')
plt.show()
