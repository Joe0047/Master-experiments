from traceProducer import *
from jobCollection import *
from simulator import *

pathToCoflowBenchmarkTraceFile = "./coflow-benchmark-master/FB2010-1Hr-150-0.txt"

t=CoflowBenchmarkTraceProducer(pathToCoflowBenchmarkTraceFile)
t.prepareTrace()

s = Simulator(t)
