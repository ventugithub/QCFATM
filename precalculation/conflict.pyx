import numpy as np
import cython
import math
from libc.math cimport sin, cos, acos, fabs

cdef getPointConflict(float lat1, float lon1, float time1, float lat2, float lon2, float time2, float spaceThreshold=0.5, float timeThreshold=10.0):
    earth_radius = 6367.0
    CosD = sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * (cos(lon1) * cos(lon2) + sin(lon1) * sin(lon2))
    CosD = min(CosD, 1)
    CosD = max(CosD, -1)
    spatialDistance = earth_radius * acos(CosD)
    temporalDistance = fabs(time1 - time2)
    if spatialDistance < spaceThreshold and temporalDistance < timeThreshold:
        return True
    else:
        return False

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

    N = len(flightIndices)
    flightNumbers = np.unique(flights)
    NFlights = len(flightNumbers)

    # get the separation of flight indices
    cdef int i
    ib = 0
    flightIndexLimits = np.ndarray(NFlights + 1, dtype=int)
    flightIndexLimits[0] = 0
    flightIndexLimits[NFlights] = len(flights)
    for i in range(1, len(flights)):
        if (flights[i] == flights[ib] + 1):
            flightIndexLimits[flights[i]] = i
            ib = i

    # mapping to coarse grid of 0.5 degree in longitude and latitude
    latRound = np.round(2*lat)/2
    lonRound = np.round(2*lon)/2

    cdef int n = 0
    cdef int m = 0
    cdef int j
    cdef int l
    cdef int k
    cdef int lmin
    cdef int lmax
    cdef int kmin
    cdef int kmax
    cdef int c = 0
    f = open('pointConflicts', 'w')
    f.write('# conflictIndex, flightIndex1, flightIndex2, lat1, lon1, time1, lat2, lon2, time2\n')
    for i in range(NFlights):
        for j in range(i + 1, NFlights):
            if n % 1 == 0:
                print n, " of ", NFlights*(NFlights-1)/2.0, " : ", (flightIndexLimits[i + 1] - flightIndexLimits[i]) * (flightIndexLimits[j + 1] - flightIndexLimits[j])
            n = n + 1

            lmin= flightIndexLimits[i]
            lmax = flightIndexLimits[i + 1]
            kmin = flightIndexLimits[j]
            kmax = flightIndexLimits[j + 1]
            for l in range(lmin, lmax):
                for k in range(kmin, kmax):
                    #if m % 10 == 0:
                        #print m, " of ", (flightIndexLimits[i + 1] - flightIndexLimits[i]) * (flightIndexLimits[j + 1] - flightIndexLimits[j])
                        #pass
                    #m = m + 1
                    isConflict = getPointConflict(lat[l], lon[l], times[l], lat[k], lon[k], times[k])
                    if isConflict:
                        f.write('%i, %i, %i, %f, %f, %f, %f, %f, %f\n' % (c, i, j, lat[l], lon[l], times[l], lat[k], lon[k], times[k]))
                        c = c + 1

