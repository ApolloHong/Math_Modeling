from pulp import *

# the cite of pulp
# https://coin-or.github.io/pulp/main/basic_python_coding.html

p = LpProblem("Problem", LpMaximize)

p1 = 10
p2 = 10
p3 = 10

a1 = LpVariable("a1", lowBound=800 * p1)
a2 = LpVariable("a2", lowBound=800 * p2)
a3 = LpVariable("a3", lowBound=800 * p3)

d1 = LpVariable("d1", lowBound=900 * p1)
d2 = LpVariable("d2", lowBound=900 * p2)
d3 = LpVariable("d3", lowBound=900 * p3)

Vp1 = LpVariable("Vp1", lowBound=3000)
Vp2 = LpVariable("Vp2", lowBound=1500)
Vm1 = LpVariable("Vm1", lowBound=3000)
Vm2 = LpVariable("Vm2", lowBound=1500)

p += a1 + a2 + a3 + d1 + d2 + d3

psum = p1 + p2 + p3

p += (Vm2 + Vm1) - Vp2 <= 0, "constraint1"
p += Vp1 + Vp2 <= 2000000, "constraint2"
p += (a1 + a2 + a3 + p1 + p2 + p3) - Vp1 <= 0, "constraint3"

status = p.solve();
print("status:", LpStatus[status])
print("obj", value(p.objective))

for i in p.variables():
    print(i, " ", i.varValue)
