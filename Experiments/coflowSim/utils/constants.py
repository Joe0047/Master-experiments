class Constants:
    # Capacity constraint of a rack in bps (1 Mb per ms)
    RACK_BITS_PER_SEC = 1.0 * 1024 * 1048576
    
    # Capacity constraint of a rack in Bps (1/8 MB per ms)
    RACK_BYTES_PER_SEC = RACK_BITS_PER_SEC / 8.0
    
    # Number of milliseconds in a second of Simulator
    # An epoch of Simulator (1024 ms per s)
    SIMULATION_SECOND_MILLIS = 1024
    
    # Time step of Simulator (transfer 1M needs 8ms)
    SIMULATION_QUANTA = SIMULATION_SECOND_MILLIS / (int) (RACK_BYTES_PER_SEC / 1048576)
    