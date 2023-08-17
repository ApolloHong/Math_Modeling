from gurobipy import *


#  模板
# model = Model()
#
#
# # 定义变量，(测试定义为连续性变量)
# x1 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='x1')
# x2 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='x2')
# d1_ub = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='d1_ub')
# d1_lb = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='d1_lb')
# d2_ub = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='d2_ub')
# d2_lb = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='d2_lb')
# d3_ub = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='d3_ub')
# d3_lb = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='d3_lb')


# # 添加约束
# model.addConstr(6*x1 + 8*x2 + d3_lb - d3_ub == 48)
# model.addConstr(4*x1 + 4*x2 + d2_lb- d2_ub == 36)
# model.addConstr(x1 - 2*x2 + d1_lb -d1_ub == 0)
# model.addConstr(5*x1 + 10*x2 <= 60)
#
#
# # 设置目标
# model.setObjectiveN(d1_lb, index=0, priority=9)
# model.setObjectiveN(d2_ub, index=1, priority=6)
# model.setObjectiveN(d3_lb, index=2, priority=3)
#
#
# model.optimize()
#
# # 打印变量值及目标
# for i in model.getVars():
#     print(i.varName, '=', i.x)
# print('obj = ', 6*x1.x + 8 * x2.x)


def my_work():
    model = Model()

    # 定义变量，(测试定义为连续性变量)
    x1 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='x1')
    x2 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='x2')
    # deviation lower or upper
    d1_ub = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='d1_ub')
    d1_lb = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='d1_lb')
    d2_ub = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='d2_ub')
    d2_lb = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='d2_lb')
    d3_ub = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='d3_ub')
    d3_lb = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='d3_lb')
    d4_ub = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='d4_ub')
    d4_lb = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='d4_lb')

    # 添加约束
    model.addConstr(70 * x1 + 120 * x2 + d1_lb - d1_ub == 50000)
    model.addConstr(x1 + d2_lb - d2_ub == 200)
    model.addConstr(x2 + d3_lb - d3_ub == 250)
    model.addConstr(9 * x1 + 4 * x2 + d4_lb - d4_ub == 3600)
    model.addConstr(4 * x1 + 5 * x2 <= 2000)
    model.addConstr(3 * x1 + 10 * x2 <= 3000)
    model.addConstr(x1 >= 0)
    model.addConstr(x2 >= 0)
    model.addConstr(d1_lb >= 0)
    model.addConstr(d1_ub >= 0)
    model.addConstr(d2_lb >= 0)
    model.addConstr(d2_ub >= 0)
    model.addConstr(d3_lb >= 0)
    model.addConstr(d3_ub >= 0)
    model.addConstr(d4_lb >= 0)
    model.addConstr(d4_ub >= 0)

    # 设置目标
    model.setObjectiveN(d1_lb, index=0, priority=9)
    model.setObjectiveN(7 * d2_ub + 12 * d3_lb, index=1, priority=6)
    model.setObjectiveN(d4_lb + d4_ub, index=2, priority=3)
    
    model.optimize()

    for i in model.getVars():
        print(i.varName, '=', i.x)
    print('obj = ', d1_lb.x * 9 + (7 * d2_ub.x + 12 * d3_lb.x) * 6 + (d4_lb.x + d4_ub.x) * 3)


my_work()
