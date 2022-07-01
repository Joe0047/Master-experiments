from task import *
from machine import *
from constants import *
from jobCollection import *

class TraceProducer:
    def __init__(self):
        self.jobs = JobCollection()
    
    def prepareTrace(self):
        return
    
    def getNumRacks(self):
        return
    
    def getMachinesPerRack(self):
        return
        
'''
Reads a trace from the <a href="https://github.com/coflow/coflow-benchmark">coflow-benchmark</a>
project.

Expected trace format:
 * Line 1: Number of Racks; Number of Jobs;
 * Line i: Job ID; Job Arrival Time; Number of Mappers; Location of each Mapper;
           Number of Reducers; Location:ShuffleMB of each Reducer;
           
Characteristics of the generated trace:
 * Each rack has at most one mapper and at most one reducer. Historically, this was because
   production clusters at Facebook and Microsoft are oversubscribed in core-rack links; essentially,
   simulating rack-level was enough for them. For full-bisection bandwidth networks, setting to the
   number of machines should result in desired outcome.
 * All tasks of a phase are known when that phase starts, meaning all mappers start together and
   all reducers do the same.
 * Mapper arrival times are ignored because they are assumed to be over before reducers start;
   i.e., shuffle start time is job arrival time.
 * Each reducer's shuffle is equally divided across mappers; i.e., reduce-side skew is intact,
   while map-side skew is lost. This is because shuffle size is logged only at the reducer end.
 * All times are in milliseconds.
'''

class CoflowBenchmarkTraceProducer(TraceProducer):
    def __init__(self, pathToCoflowBenchmarkTraceFile):
        super().__init__()
        self.NUM_RACKS = None
        self.MACHINES_PER_RACK = 1
        self.numJobs = None
        self.pathToCoflowBenchmarkTraceFile = pathToCoflowBenchmarkTraceFile
        
    def prepareTrace(self):
        f = None
        try:
            f = open(self.pathToCoflowBenchmarkTraceFile, 'r')
            line = f.readline()
            splits = line.split(" ")
            self.NUM_RACKS = int(splits[0])
            self.numJobs = int(splits[1])
            
            # Read numJobs jobs from the trace file
            for j in range(self.numJobs):
                line = f.readline()
                splits = line.split(" ")
                lIndex = 0
                
                jobName = "JOB-" + splits[lIndex]
                lIndex += 1
                job = self.jobs.getOrAddJob(jobName)
                
                jobArrivalTime = int(splits[lIndex])
                lIndex += 1
                
                # Create mappers
                numMappers = int(splits[lIndex])
                lIndex += 1
                for mID in range(numMappers):
                    taskName = "MAPPER-" + str(mID)
                    taskID = mID
                
                    # 1 <= rackIndex <= NUM_RACKS
                    rackIndex = int(splits[lIndex]) + 1
                    lIndex += 1
                    
                    # Create map task
                    task = MapTask(taskName, taskID, job, jobArrivalTime, Machine(rackIndex))
                    
                    # Add task to corresponding job
                    job.addTask(task)
                
                # Create reducers
                numReducers = int(splits[lIndex])
                lIndex += 1
                for rID in range(numReducers):
                    taskName = "REDUCER-" + str(rID)
                    taskID = rID
                    
                    # 1 <= rackIndex <= NUM_RACKS
                    rack_MB = splits[lIndex]
                    lIndex += 1
                    
                    rackIndex = int(rack_MB.split(":")[0]) + 1
                    shuffleBytes = float(rack_MB.split(":")[1]) * 1048576.0
                    
                    # Create reduce task
                    task = ReduceTask(taskName, taskID, job, jobArrivalTime, Machine(rackIndex), shuffleBytes)
                    
                    # Add task to corresponding job
                    job.addTask(task)

        except IOError:
            print("Error: cannot find " + self.pathToCoflowBenchmarkTraceFile)
            if f:
                f.close()

        finally:
            if f:
                f.close()

    def getNumRacks(self):
        return self.NUM_RACKS
    
    def getMachinesPerRack(self):
        return self.MACHINES_PER_RACK
