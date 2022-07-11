class Simulator:
    def __init__(self, traceProducer):
        self.NUM_RACKS = traceProducer.getNumRacks()
        self.MACHINES_PER_RACK = traceProducer.getMachinesPerRack()
        self.jobs = None
        self.traceProducer = traceProducer
        
        self.initialize(traceProducer)
    
    def initialize(self, traceProducer):
        self.jobs = traceProducer.jobs
        
        self.mergeTasksByRack()
    
    
    '''
        Merges all tasks in the same rack to a single one to form a non-blocking switch.
    '''
    def mergeTasksByRack(self):
        for j in self.jobs.listOfJobs:
            j.arrangeTasks(self.NUM_RACKS, self.MACHINES_PER_RACK)
        
    '''
    Event loop of the simulator that proceed epoch by epoch.
     * Simulate the time steps in each epoch, where each time step (8ms) is as long as it takes to
       transfer 1MB through each link.
     * In each time step take appropriate scheduling decision using {@link #onSchedule(long)}.
     * If any job/coflow has completed, update relavant data structures using
       {@link #afterJobDeparture(long)}.
     * Repeat.
    '''
    def simulate(self, flowsInThisCore, EPOCH_IN_MILLIS):
        CURRENT_TIME = 0
        readyFlows = []
        activeFlows = []
        
        rackMapperInfoTable = []
        rackReducerInfoTable = []
        for i in range(self.NUM_RACKS):
            rackMapperInfoTable.append(False)
            rackReducerInfoTable.append(False)
        
        while(len(flowsInThisCore) > 0 or len(activeFlows) > 0):
            newReadyFlows = []
            
            for flow in flowsInThisCore:
                flowArrivalTime = min(flow.getMapper().getArrivalTime(), flow.getReducer().getArrivalTime())
                
                if flowArrivalTime > CURRENT_TIME + EPOCH_IN_MILLIS:
                    continue
                
                # One flow added
                newReadyFlows.append(flow)
                
            for flow in newReadyFlows:
                readyFlows.append(flow)
                flowsInThisCore.remove(flow)
            
            for flow in readyFlows:
                # Convert machine to rack. (Subtracting because machine IDs start from 1)
                i = flow.getMapper().getPlacement() - 1
                j = flow.getReducer().getPlacement() - 1
                
                # If link (i,j) is idle, assign flow to it
                if rackMapperInfoTable[i] == False and rackReducerInfoTable[j] == False:
                    activeFlows.append(flow)
                    readyFlows.remove(flow)
                    
            
                
        
        
        
