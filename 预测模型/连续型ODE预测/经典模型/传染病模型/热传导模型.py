import numpy
from matplotlib import pyplot as plt

# Grid Generation

nx = 200
ny = 200
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)

# Time Step

nt = 50
alpha = 1
dt = 0.001

# Initial Condition (I.C) and Boundry Condition (B.C)

T = numpy.ones((nx, ny))  # I.C (U = Velocity)
x = numpy.linspace(0, 2, nx)  # B.C
y = numpy.linspace(0, 2, ny)  # B.C

Tn = numpy.empty_like(T)  # initialize a temporary array
X, Y = numpy.meshgrid(x, y)

T[0, :] = 20  # B.C
T[-1, :] = -100  # B.C
T[:, 0] = 150  # B.C
T[:, -1] = 100  # B.C
# Solver
###Run through nt timesteps

for n in range(nt + 1):
    Tn = T.copy()

    T[1:-1, 1:-1] = (Tn[1:-1, 1:-1] +
                     ((alpha * dt / dx ** 2) *
                      (Tn[1:-1, 2:] - 2 * Tn[1:-1, 1:-1] + Tn[1:-1, 0:-2])) +
                     ((alpha * dt / dy ** 2) *
                      (Tn[2:, 1: -1] - 2 * Tn[1:-1, 1:-1] + Tn[0:-2, 1:-1])))

    T[0, :] = 20  # From B.C
    T[-1, :] = -100  # From B.C
    T[:, 0] = 150  # From B.C
    T[:, -1] = 100  # From B.C

fig = plt.figure(figsize=(11, 7), dpi=100)
plt.contourf(X, Y, T, cmap=plt.cm.PuBu_r)
plt.colorbar()
plt.contour(X, Y, T)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
