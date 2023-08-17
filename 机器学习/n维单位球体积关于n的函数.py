# 对于n维单位球，我们用matplotlib画一下体积V关于维度n的函数图。

#!/usr/bin/env python3
import matplotlib
import numpy as np
import scipy

import matplotlib.pyplot as plt

from scipy.special import gamma

# For unit sphere of dimension n, the volume is
# V_n = \frac {\pi^{\frac n 2}} {\Gamma(\frac n 2 + 1)}
t0 = -5.0
t1 = 20.0
t = np.arange(t0, t1, 0.1)
V = np.pi ** (t / 2.0) / gamma(t / 2.0 + 1.0)

n = np.arange(t0, t1, 1.0)
V_n = np.pi ** (n / 2.0) / gamma(n / 2.0 + 1.0)

figure, ax = plt.subplots()
ax.plot(t, V)
ax.plot(n, V_n, color='green', marker='o', linestyle='')

ax.set_xlabel('n (dimensionality)')
ax.set_ylabel('C_n m^n')
ax.set_title('volume of n dimensional unit sphere')
ax.grid()

figure.savefig("volume.png")
plt.show()