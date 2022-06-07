from gurobipy import*
import numpy as np
import random

mod = Model("IP_DC")

K = 10
N = 50
I = N
J = N
M = 30

flowlist = []

d = np.zeros((K,I,J))
for k in range(K):
    for i in range(I):
        for j in range(J):
            d[k,i,j] = random.randint(0, 50)
            flowlist.append((d[k,i,j], i, j))

'''IP'''
'''
x = mod.addVars(K, I, J, M, vtype = GRB.BINARY)
T = mod.addVar(vtype = GRB.INTEGER)
'''
x = mod.addVars(K, I, J, M, lb = 0.0, ub = 1.0, vtype = GRB.CONTINUOUS)
T = mod.addVar(vtype = GRB.CONTINUOUS)

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

print('OPT: %f\n' % mod.objVal)

'''FLS'''
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

FLS_makespan = float("-inf")

for m in range(M):
    for i in range(I):
        if loadI[m][i] > FLS_makespan:
            FLS_makespan = loadI[m][i]
    for j in range(J):
        if loadO[m][j] > FLS_makespan:
            FLS_makespan = loadO[m][j]

print('FLS: %f\n' % FLS_makespan)
print('FLS/OPT: %f\n' % (FLS_makespan / mod.objVal))

'''FLPT'''
loadI = np.zeros((M,I))
loadO = np.zeros((M,J))
A = [[] for m in range(M)]

flowlist.sort(key = lambda f: f[0], reverse = True)

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

FLPT_makespan = float("-inf")

for m in range(M):
    for i in range(I):
        if loadI[m][i] > FLPT_makespan:
            FLPT_makespan = loadI[m][i]
    for j in range(J):
        if loadO[m][j] > FLPT_makespan:
            FLPT_makespan = loadO[m][j]


print('FLPT: %f\n' % FLPT_makespan)
print('FLPT/OPT: %f\n' % (FLPT_makespan / mod.objVal))

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
            


