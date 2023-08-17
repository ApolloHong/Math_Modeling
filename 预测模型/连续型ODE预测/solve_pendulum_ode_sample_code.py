import matplotlib.pyplot as plt
import numpy as np

# Physical  constants
g = 9.8
L = 2
mu = 0.1

THETA_0 = np.pi / 3  # 60 degrees
THETA_DOT_0 = 0  # No initial angulair velocity
N = []


# Definition of 连续型ODE预测
def get_double_dot(theta, theta_dot):
    return -mu * theta_dot - (g / L) * np.sin(theta)


# Solution to the differential equation
def get_theta(t):
    # Initialize changing values
    theta = THETA_0
    theta_dot = THETA_DOT_0
    delta_t = 0.01  # Some tine step
    for time in np.arange(0, t, delta_t):
        theta_double_dot = get_double_dot(theta, theta_dot)
        theta += theta_dot * delta_t
        theta_dot += theta_double_dot * delta_t
        N.append(theta)
    return N


def plot_theta(T):
    delta_t = 0.01  # Some tine step
    M = np.arange(0, T, delta_t)
    N = get_theta(T)
    plt.figure()
    plt.plot(M, N)
    plt.show()
    plt.close()


plot_theta(100)
