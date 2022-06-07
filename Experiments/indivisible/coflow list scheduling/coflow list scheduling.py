from gurobipy import*
import numpy as np
import random

mod = Model("IP_IDC")

K = 30
N = 10
I = N
J = N
M = 10

coflowlist = []

d = np.zeros((K,I,J))
for k in range(K):
    for i in range(I):
        for j in range(J):
            d[k,i,j] = random.randint(0, 50)

li = np.zeros((K,I))
lj = np.zeros((K,J))

for k in range(K):
    coflowI = []
    for i in range(I):
        for j in range(J):
            li[k,i] += d[k,i,j]
        coflowI.append(li[k,i])
    
    coflowO = []
    for j in range(J):
        for i in range(I):
            lj[k,j] += d[k,i,j]
        coflowO.append(lj[k,j])
    
    coflowlist.append((coflowI, coflowO))

'''IP'''
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

'''CLS'''
loadI = np.zeros((M,I))
loadO = np.zeros((M,J))
A = [[] for m in range(M)]

for k in coflowlist:
    m_star = -1
    minload = float("inf")
    for m in range(M):
        maxload = float("-inf")
        for i in range(I):
            for j in range(J):
                if loadI[m][i] + loadO[m][j] + k[0][i] + k[1][j] > maxload:
                    maxload = loadI[m][i] + loadO[m][j] + k[0][i] + k[1][j]
        if maxload < minload:
            m_star = m
            minload = maxload
    A[m_star].append(k)
    
    for i in range(I):
        loadI[m_star][i] += k[0][i]
    for j in range(J):
        loadO[m_star][j] += k[1][j]

CLS_makespan = float("-inf")

for m in range(M):
    for i in range(I):
        if loadI[m][i] > CLS_makespan:
            CLS_makespan = loadI[m][i]
    for j in range(J):
        if loadO[m][j] > CLS_makespan:
            CLS_makespan = loadO[m][j]

print('CLS: %f\n' % CLS_makespan)
print('CLS/OPT: %f\n' % (CLS_makespan / mod.objVal))