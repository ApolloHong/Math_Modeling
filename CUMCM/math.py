import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# x,y,z,t = sp.symbols('x y z t')
# A = sp.symbols('A')
# m = sp.symbols('m')
#
# A = sp.Matrix([[m,1,1,1],[1,m,1,1],[1,1,m,1]])
# b = sp.Matrix([[1],[m],[m+1]])
#
#
# eq1 = m*x +y+z+t-1
# eq2 = x+m*y+z+t-m
# eq3 = x+y+m*z+t-1-m
#
# print(sp.solve([eq1,eq2,eq3],[x,y,z,t]))

# A = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1]])
# b = np.array([1,0,1])
#
# print(np.linalg.pinv(A)@b)


# eq1 = x+3*y+z-1
# eq2 = x-2*y+z-1
#
# print(sp.solve([eq1,eq2],[x,y,z]))

# a = [1,2,10,3,4,5,5,6]
# print([a[np.argsort(a)[_]] for _ in range(6)])


# A = np.array([[8,-5,0,1],[0,1,7,-2]]).T
#
# b = np.array([16,-7,21,-4]).T
#
data_mu1 = np.arange(180, 220).reshape(-1,1)
data_mu2 = np.repeat(1250, len(data_mu1)).reshape(-1,1)
data_mu3 = np.repeat(50, len(data_mu1)).reshape(-1,1)
data_mu4 = np.repeat(295, len(data_mu1)).reshape(-1,1)

data_mu = np.concatenate((data_mu1, data_mu2,
                          data_mu3, data_mu4), axis=1)
print(data_mu)


# plt.figure()
# plt.scatter(np.array([104, 2]), np.array([70, 100]), s=50, c='b', marker='o')
#
# plt.show()

