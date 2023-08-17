from pulp import *
from coptpy import *

#用scipy求解线性规划问题
from scipy.optimize import linprog

c = [-1,2,3]
A = [[-2,1,1],[3,-1,-2]]
b = [[9],[-4]]
Aeq = [[4,-2,-1]]
beq = [-6]
LB = [-10,0,None]
UB = [None]*len(c)
bound = tuple(zip(LB,UB))
print(bound)
res = linprog(c,A,b,Aeq,beq,bound)
print('目标函数最小值：',res.fun)
print('最优解为：',res.x)






def getresult(c, con):
# 设置对象
    prob = LpProblem('myPro', LpMinimize)
# 设置三个变量，并设置变量最小取值
    x1 = LpVariable("x1", lowBound=0)
    x2 = LpVariable("x2", lowBound=0)
    x3 = LpVariable("x3", lowBound=0)
    X = [x1, x2, x3]
# 目标函数
    z = 0
    for i in range(len(X)):
        z += X[i]*c[i]
    prob += z
# 载入约束变量
    prob += x1-2*x2+x3 <= con[0]# 约束条件1
    prob += -4*x1+x2+2*x3 >= con[1]
    prob += -2*x1+x3 == con[2]
    print('确认线性规划是否输入正确：',prob)
# 求解
    status = prob.solve()
    print('\naim ending is ',value(prob.objective))  # 计算结果
    for i in prob.variables():
        print('\nanswer is ',i.varValue)
if __name__ == '__main__':
    c = [-3,1,1]
    con = [11,3,1]
    getresult(c,con)