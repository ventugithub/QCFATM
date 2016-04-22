import numpy as np
import cython

def detectConflicts(flightIndices, times, lat, lon):
    """ Detect conflicts

    Arguments:
    flightIndices: Array of consecutive flight indices starting from 0
    times: array of trajectory times (same length as flightIndices)
    lat: array of trajectory latitudes (same length as flightIndices)
    lon: array of trajectory longitudes (same length as flightIndices)
    """
    flights = np.array(flightIndices, dtype=int)
    times = np.array(times, dtype=int)
    lat = np.array(lat)
    lon = np.array(lon)

    flightNumbers = np.unique(flights)
    NFlights = len(flightNumbers)

    cdef int n = 0
    cdef int i
    ib = 0
    flightIndexLimits = np.ndarray(NFlights + 1, dtype=int)
    for i in range(len(flights)):
        if (flights[i] == flights[ib] + 1):
            flightIndexLimits[flights[i]] = i
            ib = i
    print flightIndexLimits

    for i in range(NFlights):
        for j in range(i + 1, NFlights):
            if n % 10000 == 0:
                #print i, j
                pass
            n = n + 1
