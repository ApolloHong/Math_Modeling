import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts

if __name__ == '__main__':
    a = 0.031 / 10000 + 0.0337 / 10000
    print(0.0336 * 100 / np.sqrt(a))  # 1320.95

    r = sts.lognorm.rvs(0.954, size=1000)
    c = plt.hist(r, bins=500)
    plt.show()

    # 双对数坐标下
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_adjustable("datalim")
    ax.plot(c[1][:-1], c[0], 'o')
    ax.set_xlim(1e-1, 1e6)
    ax.set_ylim(1e-2, 1e6)
    ax.grid()
    plt.draw()
    plt.show()

    # 半对数坐标
    fig1, ax1 = plt.subplots()
    ax1.hist(r, bins=500)
    ax1.set_xscale('log')
    ax1.set_xlim(1e-1, 1e6)
    ax1.grid()
    plt.draw()
    plt.show()
