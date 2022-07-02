'''
Constructor for Flow.
* mapper: flow source.
* reducer: flow destination.
* totalBytes: size in bytes.
'''

class Flow:
    def __init__(self, mapper, reducer, totalBytes):
        self.mapper = mapper
        self.reducer = reducer
        self.totalBytes = totalBytes
        self.bytesRemaining = totalBytes
    
    def toString(self):
        return "FLOW-" + str(self.mapper) + "-->" + str(self.reducer) + " | " + str(self.bytesRemaining)

    def getFlowSize(self):
        return self.totalBytes