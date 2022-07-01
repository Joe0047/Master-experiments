from task import *

class Job:
    def __init__(self, jobName, jobID):
        self.jobName = jobName
        self.jobID = jobID
        self.actualStartTime = float("inf")
        self.simulatedStartTime = 0
        self.simulatedFinishTime = 0
        self.tasks = []
        self.tasksInRacks = None
        self.shuffleBytesPerRack = None
        self.numMappersInRacks = None
        self.numMappers = 0
        self.numReducers = 0
        self.totalShuffleBytes = 0
    
    def addTask(self, task):
        self.tasks.append(task)
        if task.actualStartTime < self.actualStartTime:
            self.actualStartTime = task.actualStartTime
        
        if task.taskType == TaskType.MAPPER:
            self.numMappers += 1
        elif task.taskType == TaskType.REDUCER:
            self.numReducers += 1
            self.totalShuffleBytes += task.shuffleBytes
            
    def convertMachineToRack(self, machine, machinesPerRack):
        # Subtracting because machine IDs start from 1
        return int((machine - 1) / machinesPerRack)
    
    def addAscending(self, coll, t):
        index = 0
        while index < len(coll):
            if coll[index].shuffleBytesLeft > t.shuffleBytesLeft:
                break
            index += 1
        coll.insert(index, t)
            
    def arrangeTasks(self, numRacks, machinesPerRack):
        if self.numMappersInRacks == None:
            self.numMappersInRacks = []
            for i in range(numRacks):
                self.numMappersInRacks.append(0)
        
        if self.tasksInRacks == None:
            self.tasksInRacks = []
            for i in range(numRacks):
                self.tasksInRacks.append([])
            
            self.shuffleBytesPerRack = []
            for i in range(numRacks):
                self.shuffleBytesPerRack.append(0)
        
        for t in self.tasks:
            if t.taskType == TaskType.MAPPER:
                fromRack = self.convertMachineToRack(t.getPlacement(), machinesPerRack)
                self.numMappersInRacks[fromRack] += 1
            
            if t.taskType == TaskType.REDUCER:
                toRack = self.convertMachineToRack(t.getPlacement(), machinesPerRack)
                self.addAscending(self.tasksInRacks[toRack], t)
                self.shuffleBytesPerRack[toRack] += t.shuffleBytes
        
        #self.coalesceMappers(numRacks)
        #self.coalesceReducers(numRacks)
    
    def coalesceMappers(self, numRacks):
        newMappers = []
        for t in self.tasks:
            if t.taskType == TaskType.MAPPER:
                newMappers.append(t)
        
        
        
        
                
                
        
        
      
                
            
        
    
    
    
            
        
        
        
    
