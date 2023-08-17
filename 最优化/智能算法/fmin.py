import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import *


def cost_function(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


n = 50
x = np.linspace(-6, 6, n)
y = np.linspace(-6, 6, n)
z = np.zeros((n, n))
for i, a in enumerate(x):
    for j, b in enumerate(y):
        z[i, j] = cost_function([a, b])
xx, yy = np.meshgrid(x, y)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

centers = [[0, 0], [-1, 0], [0, -1], [-1, -1]]
for i, center in enumerate(centers):
    x_center = np.array(center)
    step = 0.5
    x0 = np.vstack((x_center, x_center + np.diag((step, step))))
    xtol, ftol = 1e-3, 1e-3
    xopt, fopt, iter, funcalls, warnflags, allvecs = fmin(cost_function, x_center, initial_simplex=x0, xtol=xtol,
                                                          ftol=ftol, disp=1, retall=1, full_output=1)
    print(xopt, fopt)

    ii, jj = i // 2, i % 2
    ax = axes[ii][jj]
    c = ax.pcolormesh(xx, yy, z.T, cmap='jet')
    fig.colorbar(c, ax=ax)

    t = np.asarray(allvecs)
    x_, y_ = t[:, 0], t[:, 1]
    ax.plot(x_, y_, 'r', x_[0], y_[0], 'go', x_[-1], y_[-1], 'y+', markersize=6)

plt.show()


##
def cost_function(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


x_center = np.array([0, 0])
step = 0.5
x0 = np.vstack((x_center, x_center + np.diag((step, step))))
xtol, ftol = 1e-3, 1e-3
xopt, fopt, iter, funcalls, warnflags, allvecs = fmin(cost_function, x_center, initial_simplex=x0, xtol=xtol, ftol=ftol,
                                                      disp=1, retall=1, full_output=1)
print(xopt, fopt)

n = 50
x = np.linspace(-6, 6, n)
y = np.linspace(-6, 6, n)
z = np.zeros((n, n))
for i, a in enumerate(x):
    for j, b in enumerate(y):
        z[i, j] = cost_function([a, b])

xx, yy = np.meshgrid(x, y)
fig, ax = plt.subplots()
c = ax.pcolormesh(xx, yy, z.T, cmap='jet')
fig.colorbar(c, ax=ax)

t = np.asarray(allvecs)
x_, y_ = t[:, 0], t[:, 1]
ax.plot(x_, y_, 'r', x_[0], y_[0], 'go', x_[-1], y_[-1], 'y+', markersize=6)

fig2 = plt.figure()
ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(212)
ax1.plot(x_)
ax1.set_title('x')
ax2.plot(y_)
ax2.set_title('y')
plt.show()
# ax3.plot(ys)
