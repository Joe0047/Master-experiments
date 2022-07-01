class Constants:
    # Constant for values we are not sure about
    VALUE_UNKNOWN = -1
    
    # Constant for values we don't care about
    VALUE_IGNORED = -2
    
    # Capacity constraint of a rack in bps
    RACK_BITS_PER_SEC = 1.0 * 1024 * 1048576
    
    # Capacity constraint of a rack in Bps
    RACK_BYTES_PER_SEC = RACK_BITS_PER_SEC / 8.0
    
    # Number of milliseconds in a second of Simulator
    # An epoch of Simulator
    SIMULATION_SECOND_MILLIS = 1024
    
    # Time step of Simulator
    SIMULATION_QUANTA = SIMULATION_SECOND_MILLIS / (int) (RACK_BYTES_PER_SEC / 1048576)
    