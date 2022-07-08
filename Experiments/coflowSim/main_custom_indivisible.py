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
for k in range(K):
    for t in tr.jobs.elementAt(k).tasks:
        if t.taskType != TaskType.REDUCER:
            continue
        
        for f in t.flows:
            # Convert machine to rack. (Subtracting because machine IDs start from 1)
            i = f.getMapper().getPlacement() - 1
            j = f.getReducer().getPlacement() - 1
            
            d[k,i,j] = f.getFlowSize()

li = np.zeros((K,I))
lj = np.zeros((K,J))
coflowlist = []
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
    
    coflowlist.append((coflowI, coflowO, tr.jobs.elementAt(k)))

# LP_IDC
mod = Model("LP_IDC")
      
x = mod.addVars(K, M, lb = 0.0, ub = 1.0, vtype = GRB.CONTINUOUS)
T = mod.addVar(vtype = GRB.CONTINUOUS)
#x = mod.addVars(K, M, vtype = GRB.BINARY)
#T = mod.addVar(vtype = GRB.INTEGER)

mod.update()

mod.setObjective(T, GRB.MINIMIZE)

mod.addConstrs(quicksum(x[k,h] for h in range(M)) == 1
             for k in range(K))

mod.addConstrs(quicksum(li[k,i]*x[k,h] for k in range(K)) <= T
             for i in range(I)
             for h in range(M))

mod.addConstrs(quicksum(lj[k,j]*x[k,h] for k in range(K)) <= T
             for j in range(J)
             for h in range(M))

mod.optimize()

'''
# CLS
loadI = np.zeros((M,I))
loadO = np.zeros((M,J))
A = [[] for h in range(M)]

for k in coflowlist:
    h_star = -1
    minload = float("inf")
    for h in range(M):
        maxload = float("-inf")
        for i in range(I):
            for j in range(J):
                if loadI[h][i] + loadO[h][j] + k[0][i] + k[1][j] > maxload:
                    maxload = loadI[h][i] + loadO[h][j] + k[0][i] + k[1][j]
        if maxload < minload:
            h_star = h
            minload = maxload
            
    A[h_star].append(k[2])
    for i in range(I):
        loadI[h_star][i] += k[0][i]
    for j in range(J):
        loadO[h_star][j] += k[1][j]
'''
    
    


