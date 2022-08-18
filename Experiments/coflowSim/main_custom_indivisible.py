from gurobipy import *
from traceProducer.traceProducer import *
from traceProducer.jobClassDescription import *
from datastructures.jobCollection import *
from simulator.simulator import *
import numpy as np
import matplotlib.pyplot as plt

numOfCoflows = []
CLS = []

curNumCoflows = 30
lastNumCoflows = 120
stepSize = 10

while(curNumCoflows <= lastNumCoflows):
    print(curNumCoflows)
    numOfCoflows.append(curNumCoflows)
    
    rseed = 13
    turn = 5
    listOfTurnsCLS = []
    average_CLS = 0
    
    while(turn > 0):
        numRacks = 50
        numJobs = curNumCoflows
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
        
        print("========================================================")
        print('OPT: %f' % mod.objVal)
        print('CLS: %f' % makespan_CLS)
        print(makespan_CLS / mod.objVal)
        print("========================================================")
        
        listOfTurnsCLS.append(makespan_CLS / mod.objVal)
        
        rseed += 1
        turn -= 1
    
    for c in listOfTurnsCLS:
        average_CLS += c
    average_CLS /= len(listOfTurnsCLS)
    CLS.append(average_CLS)
    
    curNumCoflows += stepSize
    
algo = {'CLS': CLS}

file = open('../result/custom_indivisible/custom_indivisible.txt','w')
for key, values in algo.items():
    file.write(key + ' ' + str(len(values)))
    
    for value in values:
        file.write(' ' + str(value))
        
    file.write('\n')

file.close()

# 設定圖片大小為長15、寬10

plt.figure(figsize=(15,10),dpi=100,linewidth = 2)


plt.plot(numOfCoflows,CLS,'s-',color = 'r', label="CLS")


# 設定圖片標題，以及指定字型設定，x代表與圖案最左側的距離，y代表與圖片的距離

plt.title("Indivisible coflows from custom", size=40, x=0.5, y=1.03)

# 設置刻度字體大小

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

# 標示x軸(labelpad代表與圖片的距離)

plt.xlabel("Number of coflows", fontsize=30, labelpad = 15)

# 標示y軸(labelpad代表與圖片的距離)

plt.ylabel("Approximation ratio", fontsize=30, labelpad = 20)

# 顯示出線條標記位置

plt.legend(loc = "best", fontsize=20)

# 畫出圖片

plt.show()

