class Simulator:
    def __init__(self, traceProducer):
        self.NUM_RACKS = traceProducer.getNumRacks()
        self.MACHINES_PER_RACK = traceProducer.getMachinesPerRack()
        self.jobs = None
        self.flowsInRacks = None
        self.activeJobs = None
        self.CURRENT_TIME = 0
        self.numActiveTasks = 0
        
        self.initialize(traceProducer)
    
    def initialize(self, traceProducer):
        self.jobs = traceProducer.jobs
        self.jobs.sortByStartTime()
        
        self.flowsInRacks = []
        for i in range(self.NUM_RACKS):
            self.flowsInRacks.append([])
        
        self.activeJobs = {}
        
        self.mergeTasksByRack()
    
    
    '''
        Merges all tasks in the same rack to a single one to form a non-blocking switch.
    '''
    def mergeTasksByRack(self):
        for j in self.jobs.listOfJobs:
            j.arrangeTasks(self.NUM_RACKS, self.MACHINES_PER_RACK)
        
        
