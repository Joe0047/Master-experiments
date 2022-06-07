from gurobipy import*
import numpy as np
import random

mod = Model("LP_DC")

K = 10
N = 10
I = N
J = N
M = 10

d = np.zeros((K,I,J))
for k in range(K):
    for i in range(I):
        for j in range(J):
            d[k,i,j] = random.randint(0, 50)

'''
x = mod.addVars(K, I, J, M, lb = 0.0, ub = 1.0, vtype = GRB.CONTINUOUS)
T = mod.addVar(vtype = GRB.CONTINUOUS)
'''
x = mod.addVars(K, I, J, M, vtype = GRB.BINARY)
T = mod.addVar(vtype = GRB.INTEGER)


mod.update()

mod.setObjective(T, GRB.MINIMIZE)

mod.addConstrs(quicksum(x[k,i,j,m] for m in range(M)) == 1   
             for i in range(I)
             for j in range(J)
             for k in range(K))

mod.addConstrs(quicksum(d[k,i,j]*x[k,i,j,m] for j in range(J) for k in range(K)) <= T
             for i in range(I)
             for m in range(M))

mod.addConstrs(quicksum(d[k,i,j]*x[k,i,j,m] for i in range(I) for k in range(K)) <= T
             for j in range(J)
             for m in range(M))

mod.optimize()
'''
for v in mod.getVars():
    print('%s %f' % (v.varName, v.x))
'''
print('OPT: %f' % mod.objVal)
