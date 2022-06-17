from gurobipy import*
import numpy as np
import random

mod = Model("IP_DC")

K = 10
N = 10
I = N
J = N
M = 20

flowlist = []


d = np.zeros((K,I,J))
for k in range(K):
    for i in range(I):
        for j in range(J):
            d[k,i,j] = random.randint(0, 50)
            flowlist.append((d[k,i,j], i, j))
            
flowlist.sort(key = lambda f: f[0], reverse = True)

'''IP'''
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

'''FLPT'''
loadI = np.zeros((M,I))
loadO = np.zeros((M,J))
A = [[] for m in range(M)]

for f in flowlist:
    m_star = -1
    minload = float("inf")
    for m in range(M):
        if loadI[m][f[1]] + loadO[m][f[2]] < minload:
            m_star = m
            minload = loadI[m][f[1]] + loadO[m][f[2]]
    A[m_star].append(f)
    loadI[m_star][f[1]] += f[0]
    loadO[m_star][f[2]] += f[0]

makespan = float("-inf")

for m in range(M):
    for i in range(I):
        if loadI[m][i] > makespan:
            makespan = loadI[m][i]
    for j in range(J):
        if loadO[m][j] > makespan:
            makespan = loadO[m][j]

print('OPT: %f' % mod.objVal)
print('FLPT: %f' % makespan)
print(makespan / mod.objVal)