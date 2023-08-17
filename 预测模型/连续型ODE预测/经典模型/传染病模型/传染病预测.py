import numpy as np
import pylab as pl
import scipy.integrate as spi
import seaborn as sns

sns.set_theme(style="darkgrid")  # 改为黑暗模式

beta = 1.4247
"""the likelihood that the disease will be transmitted from an infected to a susceptible
individual in a unit time is 尾"""
gamma = 0
# gamma is the recovery rate and in SI model, gamma equals zero
I0 = 1e-6
# I0 is the initial fraction of infected individuals
ND = 70
# ND is the total time step
TS = 1.0
INPUT = (1.0 - I0, I0)


def diff_eqs(INP, t):
    '''The main set of equations'''
    Y = np.zeros((2))
    V = INP
    Y[0] = - beta * V[0] * V[1] + gamma * V[1]
    Y[1] = beta * V[0] * V[1] - gamma * V[1]
    return Y  # For odeint


t_start = 0.0
t_end = ND
t_inc = TS
t_range = np.arange(t_start, t_end + t_inc, t_inc)
RES = spi.odeint(diff_eqs, INPUT, t_range)
"""RES is the result of fraction of susceptibles and infectious individuals at each time step respectively"""
print(RES)

# Ploting
pl.plot(RES[:, 0], '-b', marker='$S$', label='Susceptibles')
pl.plot(RES[:, 1], '-r', marker='$I$', label='Infectious')
pl.legend(loc=0)
pl.title('SI epidemic without births or deaths')
pl.xlabel('Time')
pl.ylabel('Susceptibles and Infectious')
pl.savefig('2.5-SI-high.png', dpi=900)  # This does increase the resolution.
pl.show()

# SIS的python实现

beta = 1.4247
gamma = 0.14286
I0 = 1e-6
ND = 70
TS = 1.0
INPUT = (1.0 - I0, I0)


def diff_eqs(INP, t):
    '''The main set of equations'''
    Y = np.zeros((2))
    V = INP
    Y[0] = - beta * V[0] * V[1] + gamma * V[1]
    Y[1] = beta * V[0] * V[1] - gamma * V[1]
    return Y  # For odeint


t_start = 0.0
t_end = ND
t_inc = TS
t_range = np.arange(t_start, t_end + t_inc, t_inc)
RES = spi.odeint(diff_eqs, INPUT, t_range)

print(RES)

# Ploting
pl.plot(RES[:, 0], '-bs', marker='$S$', label='Susceptibles')
pl.plot(RES[:, 1], '-ro', marker='$I$', label='Infectious')
pl.legend(loc=0)
pl.title('SIS epidemic without births or deaths')
pl.xlabel('Time')
pl.ylabel('Susceptibles and Infectious')
pl.savefig('2.5-SIS-high.png', dpi=900)  # This does increase the resolution.
pl.show()

# SIR的python实现

beta = 1.4247
gamma = 0.14286
TS = 1.0
ND = 70.0
S0 = 1 - 1e-6
I0 = 1e-6
INPUT = (S0, I0, 0.0)


def diff_eqs(INP, t):
    '''The main set of equations'''
    Y = np.zeros((3))
    V = INP
    Y[0] = - beta * V[0] * V[1]
    Y[1] = beta * V[0] * V[1] - gamma * V[1]
    Y[2] = gamma * V[1]
    return Y  # For odeint


t_start = 0.0
t_end = ND
t_inc = TS
t_range = np.arange(t_start, t_end + t_inc, t_inc)
RES = spi.odeint(diff_eqs, INPUT, t_range)

print(RES)

# Ploting
pl.plot(RES[:, 0], '-b', marker='$S$', label='Susceptibles')  # I change -g to g-- # RES[:,0], '-g',
pl.plot(RES[:, 2], '-g', marker='$R$', label='Recovereds')  # RES[:,2], '-k',
pl.plot(RES[:, 1], '-r', marker='$I$', label='Infectious')
pl.legend(loc=0)
pl.title('SIR epidemic without births or deaths')
pl.xlabel('Time')
pl.ylabel('Susceptibles, Recovereds, and Infectious')
pl.savefig('2.1-SIR-high.png', dpi=900)  # This does, too
pl.show()
