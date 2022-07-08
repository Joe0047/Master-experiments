class Utils:
    
    @staticmethod
    def sumArray(array):
        total = 0
        for i in range(len(array)):
            total += array[i]
        return total