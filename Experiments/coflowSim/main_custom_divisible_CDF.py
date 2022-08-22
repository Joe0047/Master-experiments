from gurobipy import *
from traceProducer.traceProducer import *
from traceProducer.jobClassDescription import *
from datastructures.jobCollection import *
from simulator.simulator import *
import numpy as np
import matplotlib.pyplot as plt

completionTimeOfCoreFLS = []
completionTimeOfCoreFLPT = []
completionTimeOfCoreWeaver = []
completionTimeOfCore = []
cdfOfFLS = []
cdfOfFLPT = []
cdfOfWeaver = []

numRacks = 50
numJobs = 120
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
M = 50

d, flowlist = tr.produceFlowSizeAndList()
            
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

EPOCH_IN_MILLIS = Constants.SIMULATION_QUANTA

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

makespan_FLS = float("-inf")

for h in range(M):
    finishedTimeOfCore = sim.simulate(A[h], EPOCH_IN_MILLIS)
    completionTimeOfCoreFLS.append(finishedTimeOfCore)
    if finishedTimeOfCore > makespan_FLS:
        makespan_FLS = finishedTimeOfCore

completionTimeOfCoreFLS.sort()

print("========================================================")
print('OPT: %f' % mod.objVal)
print('FLS: %f' % makespan_FLS)
print(makespan_FLS / mod.objVal)
print("========================================================")


# Initialize the remaining bytes of flows
tr.initFlowRemainingBytes()


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

makespan_FLPT = float("-inf")

for h in range(M):
    finishedTimeOfCore = sim.simulate(A[h], EPOCH_IN_MILLIS)
    completionTimeOfCoreFLPT.append(finishedTimeOfCore)
    if finishedTimeOfCore > makespan_FLPT:
        makespan_FLPT = finishedTimeOfCore

completionTimeOfCoreFLPT.sort()

print("========================================================")
print('OPT: %f' % mod.objVal)
print('FLPT: %f' % makespan_FLPT)
print(makespan_FLPT / mod.objVal)
print("========================================================")


# Initialize the remaining bytes of flows
tr.initFlowRemainingBytes()


# Weaver
loadI = np.zeros((M,I))
loadO = np.zeros((M,J))
L_add = [0 for h in range(M)]
L = [0 for h in range(M)]
A = [[] for h in range(M)]

flowlist.sort(key = lambda f: f[0], reverse = True)

for f in flowlist:
    h_star = -1
    minload = float("inf")
    
    for h in range(M):
        loadI[h][f[1]] += f[0]
        loadO[h][f[2]] += f[0]
        
        if loadI[h][f[1]] > L_add[h]:
            L_add[h] = loadI[h][f[1]]
        if loadO[h][f[2]] > L_add[h]:
            L_add[h] = loadO[h][f[2]]
        
        loadI[h][f[1]] -= f[0]
        loadO[h][f[2]] -= f[0]
        
        if (L_add[h] > L[h]) and (L_add[h] < minload):
            h_star = h
            minload = L_add[h]
    
    if h_star == -1:
        minload = float("inf")
        for h in range(M):
            loadI[h][f[1]] += f[0]
            loadO[h][f[2]] += f[0]
            
            maxload = max(loadI[h][f[1]], loadO[h][f[2]])
            
            loadI[h][f[1]] -= f[0]
            loadO[h][f[2]] -= f[0]
            
            if maxload < minload:
                h_star = h
                minload = maxload
    
    A[h_star].append(f[3])
    loadI[h_star][f[1]] += f[0]
    loadO[h_star][f[2]] += f[0]
    
    if loadI[h_star][f[1]] > L[h_star]:
        L[h_star] = loadI[h_star][f[1]]
    if loadO[h_star][f[2]] > L[h_star]:
        L[h_star] = loadO[h_star][f[2]]
    
    L_add = L.copy()

makespan_Weaver = float("-inf")

for h in range(M):
    finishedTimeOfCore = sim.simulate(A[h], EPOCH_IN_MILLIS)
    completionTimeOfCoreWeaver.append(finishedTimeOfCore)
    if finishedTimeOfCore > makespan_Weaver:
        makespan_Weaver = finishedTimeOfCore

completionTimeOfCoreWeaver.sort()

print("========================================================")
print('OPT: %f' % mod.objVal)
print('Weaver: %f' % makespan_Weaver)
print(makespan_Weaver / mod.objVal)
print("========================================================")

completionTimeOfCore.extend(completionTimeOfCoreFLS)
completionTimeOfCore.extend(completionTimeOfCoreFLPT)
completionTimeOfCore.extend(completionTimeOfCoreWeaver)

completionTimeOfCore.sort()

for i in range(len(completionTimeOfCore)):
    j = 0
    while(j < len(completionTimeOfCoreFLS)):
        if completionTimeOfCoreFLS[j] > completionTimeOfCore[i]:
            cdfOfFLS.append(j/M)
            break
        
        if j == len(completionTimeOfCoreFLS) - 1:
            cdfOfFLS.append((j+1)/M)
        
        j += 1
        
for i in range(len(completionTimeOfCore)):
    j = 0
    while(j < len(completionTimeOfCoreFLPT)):
        if completionTimeOfCoreFLPT[j] > completionTimeOfCore[i]:
            cdfOfFLPT.append(j/M)
            break
        
        if j == len(completionTimeOfCoreFLPT) - 1:
            cdfOfFLPT.append((j+1)/M)
        
        j += 1

for i in range(len(completionTimeOfCore)):
    j = 0
    while(j < len(completionTimeOfCoreWeaver)):
        if completionTimeOfCoreWeaver[j] > completionTimeOfCore[i]:
            cdfOfWeaver.append(j/M)
            break
        
        if j == len(completionTimeOfCoreWeaver) - 1:
            cdfOfWeaver.append((j+1)/M)
        
        j += 1
        
algo = {'cdfOfFLS': cdfOfFLS, 'cdfOfFLPT': cdfOfFLPT, 'cdfOfWeaver': cdfOfWeaver}

file = open('../result/custom_divisible_CDF/custom_divisible_CDF.txt','w')

file.write('completionTimeOfCore ' + str(len(completionTimeOfCore)))
for c in completionTimeOfCore:
    file.write(' ' + str(c))
file.write('\n')

for key, values in algo.items():
    file.write(key + ' ' + str(len(values)))
    
    for value in values:
        file.write(' ' + str(value))
        
    file.write('\n')

file.close()

# 設定圖片大小為長15、寬10

plt.figure(figsize=(15,10),dpi=100,linewidth = 2)


plt.plot(completionTimeOfCore,cdfOfFLS,'s-',color = 'r', label="FLS")


plt.plot(completionTimeOfCore,cdfOfFLPT,'o-',color = 'g', label="FLPT")


plt.plot(completionTimeOfCore,cdfOfWeaver,'^-',color = 'b', label="Weaver")


# 設定圖片標題，以及指定字型設定，x代表與圖案最左側的距離，y代表與圖片的距離

plt.title("Divisible coflows from custom", size=40, x=0.5, y=1.03)

# 設置刻度字體大小

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

# 設置數值不使用科學記號表示

plt.ticklabel_format(style='plain')

# 標示x軸(labelpad代表與圖片的距離)

plt.xlabel("Completion time of core (ms)", fontsize=30, labelpad = 15)

# 標示y軸(labelpad代表與圖片的距離)

plt.ylabel("CDF", fontsize=30, labelpad = 20)

# 顯示出線條標記位置

plt.legend(loc = "best", fontsize=20)

# 畫出圖片

plt.show()