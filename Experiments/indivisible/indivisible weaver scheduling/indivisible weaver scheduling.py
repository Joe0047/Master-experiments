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

'''Weaver'''
loadI = np.zeros((M,I))
loadO = np.zeros((M,J))
A = [[] for m in range(M)]

flowlist.sort(key = lambda f: f[0], reverse = True)

for f in flowlist:
    m_star = -1
    minload = float("inf")
    
    for m in range(M):
        loadI[m][f[1]] += f[0]
        loadO[m][f[2]] += f[0]
        
        maxload1 = float("-inf")
        for i in range(I):
            if loadI[m][i] > maxload1:
                maxload1 = loadI[m][i]
        for j in range(J):
            if loadO[m][j] > maxload1:
                maxload1 = loadO[m][j]
        
        loadI[m][f[1]] -= f[0]
        loadO[m][f[2]] -= f[0]
        
        maxload2 = float("-inf")
        for i in range(I):
            if loadI[m][i] > maxload2:
                maxload2 = loadI[m][i]
        for j in range(J):
            if loadO[m][j] > maxload2:
                maxload2 = loadO[m][j]
        
        if (maxload1 > maxload2) and (maxload1 < minload):
            m_star = m
            minload = maxload1
    
    if m_star == -1:
        minload = float("inf")
        for m in range(M):
            loadI[m][f[1]] += f[0]
            loadO[m][f[2]] += f[0]
            
            maxload1 = max(loadI[m][f[1]], loadO[m][f[2]])
            
            loadI[m][f[1]] -= f[0]
            loadO[m][f[2]] -= f[0]
            
            if maxload1 < minload:
                m_star = m
                minload = maxload1
    
    A[m_star].append(f)
    loadI[m_star][f[1]] += f[0]
    loadO[m_star][f[2]] += f[0]

Weaver_makespan = float("-inf")

for m in range(M):
    for i in range(I):
        if loadI[m][i] > Weaver_makespan:
            Weaver_makespan = loadI[m][i]
    for j in range(J):
        if loadO[m][j] > Weaver_makespan:
            Weaver_makespan = loadO[m][j]

print('Weaver: %f\n' % Weaver_makespan)
print('Weaver/OPT: %f\n' % (Weaver_makespan / mod.objVal))
