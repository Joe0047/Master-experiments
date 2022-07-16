from gurobipy import *
from traceProducer.traceProducer import *
from traceProducer.jobClassDescription import *
from datastructures.jobCollection import *
from simulator.simulator import *
import math
import numpy as np
import matplotlib.pyplot as plt

instanceOfCoflows = ["Dense", "Combined"]
FLS = []
FLPT = []
Weaver = []

rseed = 13
turn = 5
listOfTurnsDenseFLS = []
average_DenseFLS = 0
listOfTurnsDenseFLPT = []
average_DenseFLPT = 0
listOfTurnsDenseWeaver = []
average_DenseWeaver = 0

while(turn > 0):
    numRacks = 25
    numJobs = 120
    randomSeed = rseed
    
    jobClassDescs = [JobClassDescription(int(math.sqrt(numRacks)), numRacks, 1, 100)]
    
    fracsOfClasses = [1]
    
    tr = CustomTraceProducer(numRacks, numJobs, jobClassDescs, fracsOfClasses, randomSeed)
    tr.prepareTrace()
    
    sim = Simulator(tr)
    
    K = tr.getNumJobs()
    N = tr.getNumRacks()
    I = N
    J = N
    M = 10
    
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
        if finishedTimeOfCore > makespan_FLS:
            makespan_FLS = finishedTimeOfCore
            
    print("========================================================")
    print('OPT: %f' % mod.objVal)
    print('FLS: %f' % makespan_FLS)
    print(makespan_FLS / mod.objVal)
    print("========================================================")
    
    listOfTurnsDenseFLS.append(makespan_FLS / mod.objVal)
    
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
        if finishedTimeOfCore > makespan_FLPT:
            makespan_FLPT = finishedTimeOfCore
    
    print("========================================================")
    print('OPT: %f' % mod.objVal)
    print('FLPT: %f' % makespan_FLPT)
    print(makespan_FLPT / mod.objVal)
    print("========================================================")
    
    listOfTurnsDenseFLPT.append(makespan_FLPT / mod.objVal)
    
    # Initialize the remaining bytes of flows
    tr.initFlowRemainingBytes()
    
    
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
    
    makespan_Weaver = float("-inf")
    
    for h in range(M):
        finishedTimeOfCore = sim.simulate(A[h], EPOCH_IN_MILLIS)
        if finishedTimeOfCore > makespan_Weaver:
            makespan_Weaver = finishedTimeOfCore
            
    print("========================================================")
    print('OPT: %f' % mod.objVal)
    print('Weaver: %f' % makespan_Weaver)
    print(makespan_Weaver / mod.objVal)
    print("========================================================")
    
    listOfTurnsDenseWeaver.append(makespan_Weaver / mod.objVal)
    
    rseed += 1
    turn -= 1

for f in listOfTurnsDenseFLS:
    average_DenseFLS += f
average_DenseFLS /= len(listOfTurnsDenseFLS)
FLS.append(average_DenseFLS)

for f in listOfTurnsDenseFLPT:
    average_DenseFLPT += f
average_DenseFLPT /= len(listOfTurnsDenseFLPT)
FLPT.append(average_DenseFLPT)

for w in listOfTurnsDenseWeaver:
    average_DenseWeaver += w
average_DenseWeaver /= len(listOfTurnsDenseWeaver)
Weaver.append(average_DenseWeaver)

rseed = 13
turn = 5
listOfTurnsCombinedFLS = []
average_CombinedFLS = 0
listOfTurnsCombinedFLPT = []
average_CombinedFLPT = 0
listOfTurnsCombinedWeaver = []
average_CombinedWeaver = 0

while(turn > 0):
    numRacks = 25
    numJobs = 120
    randomSeed = rseed
    random.seed(randomSeed)
    
    jobClassDescs = []
    for i in range(numJobs):
        coin = random.randint(0, 1)
        if coin == 0:
            jobClassDescs.append(JobClassDescription(int(math.sqrt(numRacks)), numRacks, 1, 100))
        else:
            jobClassDescs.append(JobClassDescription(1, int(math.sqrt(numRacks)), 1, 100))
    
    fracsOfClasses = []
    for i in range(numJobs):
        fracsOfClasses.append(1)
    
    tr = CustomTraceProducer(numRacks, numJobs, jobClassDescs, fracsOfClasses, randomSeed)
    tr.prepareTrace()
    
    sim = Simulator(tr)
    
    K = tr.getNumJobs()
    N = tr.getNumRacks()
    I = N
    J = N
    M = 10
    
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
        if finishedTimeOfCore > makespan_FLS:
            makespan_FLS = finishedTimeOfCore
            
    print("========================================================")
    print('OPT: %f' % mod.objVal)
    print('FLS: %f' % makespan_FLS)
    print(makespan_FLS / mod.objVal)
    print("========================================================")
    
    listOfTurnsCombinedFLS.append(makespan_FLS / mod.objVal)
    
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
        if finishedTimeOfCore > makespan_FLPT:
            makespan_FLPT = finishedTimeOfCore
    
    print("========================================================")
    print('OPT: %f' % mod.objVal)
    print('FLPT: %f' % makespan_FLPT)
    print(makespan_FLPT / mod.objVal)
    print("========================================================")
    
    listOfTurnsCombinedFLPT.append(makespan_FLPT / mod.objVal)
    
    # Initialize the remaining bytes of flows
    tr.initFlowRemainingBytes()
    
    
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
    
    makespan_Weaver = float("-inf")
    
    for h in range(M):
        finishedTimeOfCore = sim.simulate(A[h], EPOCH_IN_MILLIS)
        if finishedTimeOfCore > makespan_Weaver:
            makespan_Weaver = finishedTimeOfCore
            
    print("========================================================")
    print('OPT: %f' % mod.objVal)
    print('Weaver: %f' % makespan_Weaver)
    print(makespan_Weaver / mod.objVal)
    print("========================================================")
    
    listOfTurnsCombinedWeaver.append(makespan_Weaver / mod.objVal)
    
    rseed += 1
    turn -= 1

for f in listOfTurnsCombinedFLS:
    average_CombinedFLS += f
average_CombinedFLS /= len(listOfTurnsCombinedFLS)
FLS.append(average_CombinedFLS)

for f in listOfTurnsCombinedFLPT:
    average_CombinedFLPT += f
average_CombinedFLPT /= len(listOfTurnsCombinedFLPT)
FLPT.append(average_CombinedFLPT)

for w in listOfTurnsCombinedWeaver:
    average_CombinedWeaver += w
average_CombinedWeaver /= len(listOfTurnsCombinedWeaver)
Weaver.append(average_CombinedWeaver)

# 設定圖片大小為長15、寬10

plt.figure(figsize=(15,10),dpi=100,linewidth = 2)

x = np.arange(len(instanceOfCoflows))

width = 0.3

plt.bar(x,FLS,width,color = 'r', label="FLS")

plt.bar(x+width,FLPT,width,color = 'g', label="FLPT")

plt.bar(x+width+width,Weaver,width,color = 'b', label="Weaver")


# 設定圖片標題，以及指定字型設定，x代表與圖案最左側的距離，y代表與圖片的距離

plt.title("Indivisible coflows from custom", size=40, x=0.5, y=1.03)

# 設置刻度字體大小

plt.xticks(x+width,instanceOfCoflows,fontsize=20)

plt.yticks(fontsize=20)

# 標示x軸(labelpad代表與圖片的距離)

plt.xlabel("Coflow instance", fontsize=30, labelpad = 15)

# 標示y軸(labelpad代表與圖片的距離)

plt.ylabel("Approximation ratio", fontsize=30, labelpad = 20)

# 顯示出線條標記位置

plt.legend(loc = "best", fontsize=20)

# 畫出圖片

plt.show()