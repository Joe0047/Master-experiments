from gurobipy import *
from traceProducer.traceProducer import *
from traceProducer.jobClassDescription import *
from datastructures.jobCollection import *
from simulator.simulator import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

coreSpeedDistribution = [[[1,9], [2,8], [3,7], [4,6], [5,5]], 
     [[1,1,8], [1,2,7], [1,3,6], [1,4,5], [2,2,6], [2,3,5], [2,4,4], [3,3,4]],
     [[1,1,1,7], [1,1,2,6], [1,1,3,5], [1,1,4,4], [1,2,2,5], [1,2,3,4], [1,3,3,3], [2,2,2,4], [2,2,3,3]]]


for speedListOfCores in coreSpeedDistribution:
    curNumCores = len(speedListOfCores[0])
    
    rawFLPT = []
    rawWeaver = []
    FLPT = []
    Weaver = []
    
    indexOfCores = []
    for i in range(len(speedListOfCores)):
        indexOfCores.append(i+1)
    
    index = 1
    for speedIndex in speedListOfCores:
        rseed = 13
        turn = 100
        
        listOfTurnsFLPT = []
        average_FLPT = 0
        listOfTurnsWeaver = []
        average_Weaver = 0
        
        while(turn > 0):
            print(curNumCores)
            print(index)
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
            M = curNumCores
            
            d, flowlist = tr.produceFlowSizeAndList()
                        
            # Relaxed Linear Program of Divisible Coflows
            mod = Model("LP_DC")
            
            x = mod.addVars(K, I, J, M, lb = 0.0, ub = 1.0, vtype = GRB.CONTINUOUS)
            T = mod.addVar(vtype = GRB.CONTINUOUS)
            
            mod.update()
            
            mod.setObjective(T, GRB.MINIMIZE)
            
            mod.addConstrs(quicksum(x[k,i,j,h] for h in range(M)) == 1   
                         for i in range(I)
                         for j in range(J)
                         for k in range(K))
            
            mod.addConstrs(quicksum(d[k,i,j]*x[k,i,j,h]/speedIndex[h] for j in range(J) for k in range(K)) <= T
                         for i in range(I)
                         for h in range(M))
            
            mod.addConstrs(quicksum(d[k,i,j]*x[k,i,j,h]/speedIndex[h] for i in range(I) for k in range(K)) <= T
                         for j in range(J)
                         for h in range(M))
            
            mod.optimize()
            
            # set timestep
            EPOCH_IN_MILLIS = Constants.SIMULATION_QUANTA
            
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
                loadI[h_star][f[1]] += f[0] / speedIndex[h_star]
                loadO[h_star][f[2]] += f[0] / speedIndex[h_star]
            
            makespan_FLPT = float("-inf")
            
            for h in range(M):
                finishedTimeOfCore = sim.simulate(A[h], EPOCH_IN_MILLIS) / speedIndex[h]
                if finishedTimeOfCore > makespan_FLPT:
                    makespan_FLPT = finishedTimeOfCore
            
            print("========================================================")
            print('OPT: %f' % mod.objVal)
            print('FLPT: %f' % makespan_FLPT)
            print(makespan_FLPT / mod.objVal)
            print("========================================================")
            
            listOfTurnsFLPT.append(makespan_FLPT / mod.objVal)
            
            # Initialize the remaining bytes of flows
            tr.initFlowRemainingBytes()
            
            
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
                        maxload = max(max(loadI[h][f[1]]+f[0], loadO[h][f[2]]+f[0]), L[h]) / speedIndex[h]
                        if maxload < minload:
                            h_star = h
                            minload = maxload
                
                if h_star == -1:
                    minload = float("inf")
                    for h in range(M):
                        loadI[h][f[1]] += f[0]
                        loadO[h][f[2]] += f[0]
                        
                        maxload = max(loadI[h][f[1]], loadO[h][f[2]]) / speedIndex[h]
                        
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
            
            makespan_Weaver = float("-inf")
            
            for h in range(M):
                finishedTimeOfCore = sim.simulate(A[h], EPOCH_IN_MILLIS) / speedIndex[h]
                if finishedTimeOfCore > makespan_Weaver:
                    makespan_Weaver = finishedTimeOfCore
                    
            print("========================================================")
            print('OPT: %f' % mod.objVal)
            print('Weaver: %f' % makespan_Weaver)
            print(makespan_Weaver / mod.objVal)
            print("========================================================")
            
            listOfTurnsWeaver.append(makespan_Weaver / mod.objVal)
            
            rseed += 1
            turn -= 1
    
        
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
        
        index += 1
    
    raw = {'rawFLPT': rawFLPT, 'rawWeaver': rawWeaver}
    algo = {'FLPT': FLPT, 'Weaver': Weaver}

    file = open('../result/custom_divisible_heterogeneous/custom_divisible_heterogeneous_core' + str(curNumCores) + '.txt','w')
    
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
    
    
    plt.plot(indexOfCores,FLPT,'o-',color = 'g', label="FLPT")
    
    
    plt.plot(indexOfCores,Weaver,'^-',color = 'b', label="Weaver")
    
    
    # 設定圖片標題，以及指定字型設定，x代表與圖案最左側的距離，y代表與圖片的距離
    
    plt.title("Divisible coflows from custom", size=40, x=0.5, y=1.03)
    
    # 設置刻度字體大小
    
    plt.xticks(fontsize=20)
    
    plt.yticks(fontsize=20)
    
    # 標示x軸(labelpad代表與圖片的距離)
    
    plt.xlabel("Configuration index", fontsize=30, labelpad = 15)
    
    # x軸只顯示整數刻度
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer = True))
    
    # 標示y軸(labelpad代表與圖片的距離)
    
    plt.ylabel("Approximation ratio", fontsize=30, labelpad = 20)
    
    # 顯示出線條標記位置
    
    plt.legend(loc = "best", fontsize=20)
    
    # 畫出圖片
    
    plt.show()
    
    # 清空圖片
    
    plt.clf()
    
