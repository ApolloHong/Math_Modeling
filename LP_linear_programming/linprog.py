# scipy标准型都是求最小值，小于等于

##基本调用格式
# res=linprog(c,A, b, Aeq, beq) #默认每个决策变量下界为0，上界为+∞
# res=linprog(c, A=None, b=None, Aeq=None, beq=None, bounds=None,method='simplex')
# print(res.fun) #显示目标函数小值
# print(res.x) #显示最优解
