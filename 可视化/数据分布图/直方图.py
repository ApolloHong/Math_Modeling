import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import laplace

# `numpy.random` uses its own PRNG.
np.random.seed(444)
np.set_printoptions(precision=3)

# laplace分布 location=15 scale=3
d = np.random.laplace(loc=15, scale=3, size=500)

# np.histogram函数可以把数据自动分成histogram，默认分为10个bin
hist, bin_edges = np.histogram(d)

# draw the histogram of the data
n, bins, patches = plt.hist(x=d, bins=30, density=1, facecolor='blue', alpha=0.5)  # density用于归一化
# add a 'best fit' line
y = laplace.pdf(bins, loc=15, scale=3)
plt.plot(bins, y, 'r')
plt.xlabel('x')
plt.ylabel('Frequency')
plt.title(r'Histogram of : $\mu=15$, $b=3$')

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()
