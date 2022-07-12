from gurobipy import *
from traceProducer.traceProducer import *
from traceProducer.jobClassDescription import *
from datastructures.jobCollection import *
from simulator.simulator import *
import numpy as np

numRacks = 150
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

li, lj, coflowlist = tr.produceCoflowSizeAndList()

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

EPOCH_IN_MILLIS = Constants.SIMULATION_QUANTA

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
    
    for t in k[2].tasks:
        if t.taskType != TaskType.REDUCER:
            continue
        
        for f in t.flows:
            A[h_star].append(f)
    
    for i in range(I):
        loadI[h_star][i] += k[0][i]
    for j in range(J):
        loadO[h_star][j] += k[1][j]

makespan_CLS = float("-inf")

for h in range(M):
    finishedTimeOfCore = sim.simulate(A[h], EPOCH_IN_MILLIS)
    if finishedTimeOfCore > makespan_CLS:
        makespan_CLS = finishedTimeOfCore
        
# Test whether if some flows are not finished 
for k in range(K):
    for t in tr.jobs.elementAt(k).tasks:
        if t.taskType != TaskType.REDUCER:
            continue
        
        for f in t.flows:
            if f.bytesRemaining > 0:
                print("some flows are not finished!!!")

print("========================================================")
print('OPT: %f' % mod.objVal)
print('CLS: %f' % makespan_CLS)
print(makespan_CLS / mod.objVal)
print("========================================================")
    


