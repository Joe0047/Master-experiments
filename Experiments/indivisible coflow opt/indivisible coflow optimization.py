from gurobipy import*
import numpy as np
import random

mod = Model("IP_IDC")

K = 10
N = 10
I = N
J = N
M = 30

d = np.zeros((K,I,J))
for k in range(K):
    for i in range(I):
        for j in range(J):
            d[k,i,j] = random.randint(0, 50)

li = np.zeros((K,I))
for k in range(K):
    for i in range(I):
        for j in range(J):
            li[k,i] += d[k,i,j]
        
lj = np.zeros((K,J))
for k in range(K):
    for j in range(J):
        for i in range(I):
            lj[k,j] += d[k,i,j]

#x = mod.addVars(K, M, lb = 0.0, ub = 1.0, vtype = GRB.CONTINUOUS)
#T = mod.addVar(vtype = GRB.CONTINUOUS)
x = mod.addVars(K, M, vtype = GRB.BINARY)
T = mod.addVar(vtype = GRB.INTEGER)

mod.update()

mod.setObjective(T, GRB.MINIMIZE)

mod.addConstrs(quicksum(x[k,m] for m in range(M)) == 1
             for k in range(K))

mod.addConstrs(quicksum(li[k,i]*x[k,m] for k in range(K)) <= T
             for i in range(I)
             for m in range(M))

mod.addConstrs(quicksum(lj[k,j]*x[k,m] for k in range(K)) <= T
             for j in range(J)
             for m in range(M))

mod.optimize()

print('OPT: %g' % mod.objVal)
