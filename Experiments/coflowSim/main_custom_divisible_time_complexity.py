from gurobipy import *
from traceProducer.traceProducer import *
from traceProducer.jobClassDescription import *
from datastructures.jobCollection import *
from simulator.simulator import *
import numpy as np
import matplotlib.pyplot as plt
import time

instanceOfAlgo = ["FLS", "FLPT", "Weaver"]
rawFLS = []
rawFLPT = []
rawWeaver = []
FLS = []
FLPT = []
Weaver = []

rseed = 13
turn = 100
listOfTurnsFLS = []
average_FLS = 0
listOfTurnsFLPT = []
average_FLPT = 0
listOfTurnsWeaver = []
average_Weaver = 0

while(turn > 0):
    print(turn)
    # set the number of ports
    numRacks = 50
    
    # set the number of coflows
    numJobs = 100
    
    randomSeed = rseed
    
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
    
    # set the number of cores
    M = 10
    
    d, flowlist = tr.produceFlowSizeAndList()
 
    start = time.time()
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
    
    end = time.time()
    
    executionTimeOfFLS = end - start
    
    print("========================================================")
    print('execution time of FLS: %f' % executionTimeOfFLS)
    print("========================================================")
    
    listOfTurnsFLS.append(executionTimeOfFLS)
    
    start = time.time()
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
    
    end = time.time()
    
    executionTimeOfFLPT = end - start
    
    print("========================================================")
    print('execution time of FLPT: %f' % executionTimeOfFLPT)
    print("========================================================")
    
    listOfTurnsFLPT.append(executionTimeOfFLPT)
    
    start = time.time()
    # Weaver
    loadI = np.zeros((M,I))
    loadO = np.zeros((M,J))
    L = [0 for h in range(M)]
    A = [[] for h in range(M)]
    
    flowlist.sort(key = lambda f: f[0], reverse = True)
    
    for f in flowlist:
        h_star = -1
        minload = float("inf")
        flag = -1
        for h in range(M):
            if loadI[h][f[1]]+f[0] > L[h]:
                flag = 1
            if loadO[h][f[2]]+f[0] > L[h]:
                flag = 1
        
        if flag == 1:
            for h in range(M):
                maxload = max(max(loadI[h][f[1]]+f[0], loadO[h][f[2]]+f[0]), L[h])
                if maxload < minload:
                    h_star = h
                    minload = maxload
        
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
    
    end = time.time()
    
    executionTimeOfWeaver = end - start
    
    print("========================================================")
    print('execution time of Weaver: %f' % executionTimeOfWeaver)
    print("========================================================")
    
    listOfTurnsWeaver.append(executionTimeOfWeaver)
    
    rseed += 1
    turn -= 1

for f in listOfTurnsFLS:
    average_FLS += f
average_FLS /= len(listOfTurnsFLS)
FLS.append(average_FLS)

rawFLS.append(listOfTurnsFLS)

for f in listOfTurnsFLPT:
    average_FLPT += f
average_FLPT /= len(listOfTurnsFLPT)
FLPT.append(average_FLPT)

rawFLPT.append(listOfTurnsFLPT)

for w in listOfTurnsWeaver:
    average_Weaver += w
average_Weaver /= len(listOfTurnsWeaver)
Weaver.append(average_Weaver)

rawWeaver.append(listOfTurnsWeaver)


raw = {'rawFLS': rawFLS, 'rawFLPT': rawFLPT, 'rawWeaver': rawWeaver}
algo = {'FLS': FLS, 'FLPT': FLPT, 'Weaver': Weaver}

file = open('../result/custom_divisible_time_complexity/custom_divisible_time_complexity.txt','w')

for key, values in raw.items():
    file.write(key + ' ' + str(len(values)))
    
    for value in values:
        file.write(' ' + str(len(value)))
        for v in value:
            file.write(' ' + str(v))
        
    file.write('\n')
    
for key, values in algo.items():
    file.write(key + ' ' + str(len(values)))
    
    for value in values:
        file.write(' ' + str(value))
        
    file.write('\n')

file.close()

# 設定圖片大小為長15、寬10

plt.figure(figsize=(15,10),dpi=100,linewidth = 2)

x = np.arange(len(instanceOfAlgo))

width = 0.3

plt.bar(x[0],FLS,width,color = 'r', label="FLS")

plt.bar(x[1],FLPT,width,color = 'g', label="FLPT")

plt.bar(x[2],Weaver,width,color = 'b', label="Weaver")

# 設定圖片標題，以及指定字型設定，x代表與圖案最左側的距離，y代表與圖片的距離

plt.title("Divisible coflows from custom", size=40, x=0.5, y=1.03)

# 設置刻度字體大小

plt.xticks(x,instanceOfAlgo,fontsize=20)

plt.yticks(fontsize=20)

# 標示x軸(labelpad代表與圖片的距離)

plt.xlabel("Algorithms", fontsize=30, labelpad = 15)

# 標示y軸(labelpad代表與圖片的距離)

plt.ylabel("Execution time (s)", fontsize=30, labelpad = 20)

# 顯示出線條標記位置

plt.legend(loc = "best", fontsize=20)

# 畫出圖片

plt.show()