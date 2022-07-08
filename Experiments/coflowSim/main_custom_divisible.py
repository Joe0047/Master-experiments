from gurobipy import *
from traceProducer.traceProducer import *
from traceProducer.jobClassDescription import *
from datastructures.jobCollection import *
from simulator.simulator import *
import numpy as np

numRacks = 100
numJobs = 100
randomSeed = 13

jobClassDescs = [JobClassDescription(1, 5, 1, 10),
                 JobClassDescription(1, 5, 10, 1000),
                 JobClassDescription(5, numRacks, 1, 10),
                 JobClassDescription(5, numRacks, 10, 1000)]

fracsOfClasses = [41,
                  29,
                  9,
                  21]

tr = CustomTraceProducer(numRacks, numJobs, jobClassDescs, fracsOfClasses, randomSeed)
tr.prepareTrace()

sim = Simulator(tr)

K = tr.getNumJobs()
N = tr.getNumRacks()
I = N
J = N
M = 10

d = np.zeros((K,I,J))
flowlist = []

for k in range(K):
    for t in tr.jobs.elementAt(k).tasks:
        if t.taskType != TaskType.REDUCER:
            continue
        
        for f in t.flows:
            # Convert machine to rack. (Subtracting because machine IDs start from 1)
            i = f.getMapper().getPlacement() - 1
            j = f.getReducer().getPlacement() - 1
            
            d[k,i,j] = f.getFlowSize()
            flowlist.append((d[k,i,j], i, j, f))
            
# LP_DC
mod = Model("LP_DC")

x = mod.addVars(K, I, J, M, lb = 0.0, ub = 1.0, vtype = GRB.CONTINUOUS)
T = mod.addVar(vtype = GRB.CONTINUOUS)

#x = mod.addVars(K, I, J, M, vtype = GRB.BINARY)
#T = mod.addVar(vtype = GRB.INTEGER)


mod.update()

mod.setObjective(T, GRB.MINIMIZE)

mod.addConstrs(quicksum(x[k,i,j,h] for h in range(M)) == 1   
             for i in range(I)
             for j in range(J)
             for k in range(K))

mod.addConstrs(quicksum(d[k,i,j]*x[k,i,j,h] for j in range(J) for k in range(K)) <= T
             for i in range(I)
             for h in range(M))

mod.addConstrs(quicksum(d[k,i,j]*x[k,i,j,h] for i in range(I) for k in range(K)) <= T
             for j in range(J)
             for h in range(M))

mod.optimize()

'''
# FLS
loadI = np.zeros((M,I))
loadO = np.zeros((M,J))
A = [[] for h in range(M)]

for f in flowlist:
    h_star = -1
    minload = float("inf")
    
    for h in range(M):
        if loadI[h][f[1]] + loadO[h][f[2]] < minload:
            h_star = h
            minload = loadI[h][f[1]] + loadO[h][f[2]]
            
    A[h_star].append(f[3])
    loadI[h_star][f[1]] += f[0]
    loadO[h_star][f[2]] += f[0]


# FLPT
loadI = np.zeros((M,I))
loadO = np.zeros((M,J))
A = [[] for h in range(M)]

flowlist.sort(key = lambda f: f[0], reverse = True)

for f in flowlist:
    h_star = -1
    minload = float("inf")
    for h in range(M):
        if loadI[h][f[1]] + loadO[h][f[2]] < minload:
            h_star = h
            minload = loadI[h][f[1]] + loadO[h][f[2]]
            
    A[h_star].append(f[3])
    loadI[h_star][f[1]] += f[0]
    loadO[h_star][f[2]] += f[0]


# Weaver
loadI = np.zeros((M,I))
loadO = np.zeros((M,J))
A = [[] for h in range(M)]

flowlist.sort(key = lambda f: f[0], reverse = True)

for f in flowlist:
    h_star = -1
    minload = float("inf")
    
    for h in range(M):
        loadI[h][f[1]] += f[0]
        loadO[h][f[2]] += f[0]
        
        maxload1 = float("-inf")
        for i in range(I):
            if loadI[h][i] > maxload1:
                maxload1 = loadI[h][i]
        for j in range(J):
            if loadO[h][j] > maxload1:
                maxload1 = loadO[h][j]
        
        loadI[h][f[1]] -= f[0]
        loadO[h][f[2]] -= f[0]
        
        maxload2 = float("-inf")
        for i in range(I):
            if loadI[h][i] > maxload2:
                maxload2 = loadI[h][i]
        for j in range(J):
            if loadO[h][j] > maxload2:
                maxload2 = loadO[h][j]
        
        if (maxload1 > maxload2) and (maxload1 < minload):
            h_star = h
            minload = maxload1
    
    if h_star == -1:
        minload = float("inf")
        for h in range(M):
            loadI[h][f[1]] += f[0]
            loadO[h][f[2]] += f[0]
            
            maxload1 = max(loadI[h][f[1]], loadO[h][f[2]])
            
            loadI[h][f[1]] -= f[0]
            loadO[h][f[2]] -= f[0]
            
            if maxload1 < minload:
                h_star = h
                minload = maxload1
    
    A[h_star].append(f[3])
    loadI[h_star][f[1]] += f[0]
    loadO[h_star][f[2]] += f[0]



#print('OPT: %f' % mod.objVal)
#print('FLS: %f' % makespan)
#print(makespan / mod.objVal)
'''
    
    


