import pprint
from cvxopt import matrix, solvers

#简单的等式约束二次优化
def ex1():
    P = matrix([[4.0,1.0],[1.0,2.0]])
    q = matrix([1.0,1.0])
    G = matrix([[-1.0,0.0],[0.0,-1.0]])
    h = matrix([0.0,0.0])
    A = matrix([1.0,1.0],(1,2))
    b = matrix([1.0])

    result = solvers.qp(P,q,G,h,A,b)
    print('x\n',result['x'])


#简单不等式约束
from cvxopt import matrix, solvers
P = matrix([[1.0, 0.0], [0.0, 0.0]])
q = matrix([3.0, 4.0])
G = matrix([[-1.0, 0.0, -1.0, 2.0, 3.0], [0.0, -1.0, -3.0, 5.0, 4.0]])
h = matrix([0.0, 0.0, -15.0, 100.0, 80.0])
sol=solvers.qp(P, q, G, h)
print(sol['x'])
print(sol['primal objective'])