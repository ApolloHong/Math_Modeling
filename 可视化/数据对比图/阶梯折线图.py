import matplotlib.pyplot as plt
import numpy as np


def change1(x):
    tmp = []
    Len = len(x)
    for i in range(Len):
        tmp.append(x[i])
        if i != 0:
            tmp.append(x[i])
    return tmp


def change2(y):
    tmp = []
    Len = len(y)
    for i in range(Len):
        tmp.append(y[i])
        if i != Len - 1:
            tmp.append(y[i])
    return tmp


def plot_step_line(x, y):
    xsl = change1(x)
    ysl = change2(y)
    plt.figure()
    plt.grid()
    plt.plot(xsl, ysl, linewidth=1.5)
    # plt.xlabel("X")
    # plt.ylabel("Y")
    plt.show()


plot_step_line([1, 2, 3, 4, 6, 8], [10, 20, 30, 40, 25, 15])

x = np.linspace(1, 10, 10)
y = np.sin(x)

plt.step(x, y, lw=2)

plt.xlim(0, 11)
plt.xticks(np.arange(1, 11, 1))
plt.grid()
plt.ylim(-1.2, 1.2)

plt.show()
