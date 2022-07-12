from utils.constants import *

class Simulator:
    def __init__(self, traceProducer):
        self.NUM_RACKS = traceProducer.getNumRacks()
        self.MACHINES_PER_RACK = traceProducer.getMachinesPerRack()
        self.jobs = None
        
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
     * Repeat.
    '''
    def simulate(self, flowsInThisCore, EPOCH_IN_MILLIS):
        CURRENT_TIME = 0
        readyFlows = []
        activeFlows = []
        
        rackMapperInfoTable = []
        rackReducerInfoTable = []
        finishedTimeOfRackReducer = []
        for i in range(self.NUM_RACKS):
            rackMapperInfoTable.append(False)
            rackReducerInfoTable.append(False)
            finishedTimeOfRackReducer.append(0)
        
        while(len(flowsInThisCore) > 0 or len(readyFlows) > 0 or len(activeFlows) > 0):
            
            newReadyFlows = []
            for flow in flowsInThisCore:
                flowArrivalTime = min(flow.getMapper().getArrivalTime(), flow.getReducer().getArrivalTime())
                
                # If flow is not arrived, do not add flow to ready flows
                if flowArrivalTime > CURRENT_TIME + EPOCH_IN_MILLIS:
                    continue
                
                # One flow added to ready flows
                newReadyFlows.append(flow)
                
            for flow in newReadyFlows:
                readyFlows.append(flow)
                flowsInThisCore.remove(flow)
            
            newActiveFlows = []
            for flow in readyFlows:
                # Convert machine to rack. (Subtracting because machine IDs start from 1)
                i = flow.getMapper().getPlacement() - 1
                j = flow.getReducer().getPlacement() - 1
                
                # If link (i,j) is not idle, do not assign flow to active flows
                if rackMapperInfoTable[i] == True or rackReducerInfoTable[j] == True:
                    continue
                
                # Update the mapper and reducer rack table
                rackMapperInfoTable[i] = True
                rackReducerInfoTable[j] = True
                
                # One flow added to active flows
                newActiveFlows.append(flow)
            
            for flow in newActiveFlows:
                activeFlows.append(flow)
                readyFlows.remove(flow)
            
            finishedFlows = []
            for flow in activeFlows:
                # Convert machine to rack. (Subtracting because machine IDs start from 1)
                i = flow.getMapper().getPlacement() - 1
                j = flow.getReducer().getPlacement() - 1
                
                flow.bytesRemaining -= EPOCH_IN_MILLIS * Constants.RACK_BYTES_PER_MILLISEC
                
                # Finished flow
                if flow.bytesRemaining <= 0:
                    finishedFlows.append(flow)
                    
                    # Update the mapper and reducer rack table
                    rackMapperInfoTable[i] = False
                    rackReducerInfoTable[j] = False
                    
                    # Update the finished time of rack reducer
                    finishedTimeOfRackReducer[j] = CURRENT_TIME + EPOCH_IN_MILLIS
            
            for flow in finishedFlows:
                activeFlows.remove(flow)
                
            CURRENT_TIME += EPOCH_IN_MILLIS
                
        maxFinishedTime = float("-inf")
        for i in range(self.NUM_RACKS):
            if finishedTimeOfRackReducer[i] > maxFinishedTime:
                maxFinishedTime = finishedTimeOfRackReducer[i]
            
        return maxFinishedTime
        
        
        
