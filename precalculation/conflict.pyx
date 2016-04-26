import numpy as np
import cython
import math
from libc.math cimport sin, cos, acos, fabs
from libcpp.vector cimport vector

cdef mapToCoarseGrid(float lat, float lon, float time, float latMin, float lonMin, float timeMin, float deltaLat=0.5, float deltaLon=0.5, float deltaTime=60.0):
    cdef int I = int((lat - latMin) / deltaLat)
    cdef int J = int((lon - lonMin) / deltaLon)
    cdef int T = int((time - timeMin) / deltaTime)
    return I, J, T

cdef getCoarsePointConflict(vector[vector[vector[vector[vector[float]]]]] & coarseTraj, int I, int J, int K):
    rangeI = np.array((I - 1, I, I + 1), dtype=int)
    rangeJ = np.array((J - 1, J, J + 1), dtype=int)
    rangeK = np.array((K - 1, K, K + 1), dtype=int)
    cdef int i
    cdef int j
    cdef int k
    conflicts = []
    for i in rangeI:
        for j in rangeJ:
            for k in rangeK:
                if coarseTraj[i][j][k][0].size() != 0:
                    conflicts.append((i, j, k))
    return conflicts




cdef getPointConflict(float lat1, float lon1, float time1, float lat2, float lon2, float time2, float spaceThreshold=55.56, float timeThreshold=60.0, earthRadius=6000):
    CosD = sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * (cos(lon1) * cos(lon2) + sin(lon1) * sin(lon2))
    CosD = min(CosD, 1)
    CosD = max(CosD, -1)
    spatialDistance = earthRadius * acos(CosD)
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
    lat = np.array(lat, dtype=float)
    lon = np.array(lon, dtype=float)
    cdef float latMin = lat.min()
    cdef float latMax = lat.max()
    cdef float lonMin = lon.min()
    cdef float lonMax = lon.max()
    cdef float timeMin = times.min()
    cdef float timeMax = times.max()
    cdef float deltaLat = 0.5
    cdef float deltaLon = 0.5
    cdef int deltaTime = 60
    cdef int Nlat = int((latMax - latMin) / deltaLat) + 1
    cdef int Nlon = int((lonMax - lonMin) / deltaLon) + 1
    cdef int Ntime = int((timeMax - timeMin) / deltaTime) + 1

    # earth radius in kilometers
    cdef float earthRadius = 6367.0
    # nautic mile in kilometers
    cdef float nautic = 1.852
    # minimal acceptable distance in kilometers
    cdef float spaceThreshold = 30.0 * nautic
    # minimal acceptable time difference in minutes
    cdef float temporalThreshold = 60

    cdef int i
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
    cdef int flight1
    cdef int flight2
    print "prepare coarse trajectory container"
    #coarseTraj = np.empty((Nlat, Nlon, Ntime, 0))
    cdef vector[vector[vector[vector[vector[float]]]]] coarseTraj
    coarseTraj.resize(Nlat)
    for i in range(0, Nlat):
        coarseTraj[i].resize(Nlon)
        for j in range(0, Nlon):
            coarseTraj[i][j].resize(Ntime)
            for k in range(0, Ntime):
                coarseTraj[i][j][k].resize(5)

    N = len(flightIndices)
    flightNumbers = np.unique(flights)
    NFlights = len(flightNumbers)

    # get the separation of flight indices
    cdef int I
    cdef int J
    cdef int K
    cdef int Ip
    cdef int Jp
    cdef int Kp
    ib = 0
    flightIndexLimits = np.ndarray(NFlights + 1, dtype=int)
    flightIndexLimits[0] = 0
    flightIndexLimits[NFlights] = len(flights)
    print "mapping to coarse trajectory container"
    print Nlat, Nlon, Ntime
    for i in range(1, len(flights)):
        if i % 1000 == 0:
            print i, " of ",  len(flights)
        # mapping to coarse trajectory container
        I, J, K = mapToCoarseGrid(lat[i], lon[i], times[i], latMin, lonMin, timeMin, deltaLat, deltaLon, deltaTime)
        coarseTraj[I][J][K][0].push_back(flights[i])
        coarseTraj[I][J][K][1].push_back(i)
        coarseTraj[I][J][K][2].push_back(lat[i])
        coarseTraj[I][J][K][3].push_back(lon[i])
        coarseTraj[I][J][K][4].push_back(times[i])

        # get flight index limits
        if (flights[i] == flights[ib] + 1):
            flightIndexLimits[flights[i]] = i
            ib = i

    f = open('pointConflicts.dat', 'w')
    f.write('# conflictIndex, flightIndex1, flightIndex2, lat1, lon1, time1, lat2, lon2, time2\n')
    n = 0

    cdef float lat1
    cdef float lon1
    cdef float time1
    cdef float lat2
    cdef float lon2
    cdef float time2
    c = 0
    for I in range(1, Nlat - 1):
        for J in range(1, Nlon - 1):
            for K in range(1, Ntime - 1):
                if n % 100 == 0:
                    print n, " of ", (Nlat - 1) * (Nlon - 1) * (Ntime - 1)
                n = n + 1
                conflicts = getCoarsePointConflict(coarseTraj, I, J, K)
                if conflicts:
                    for l in range(coarseTraj[I][J][K][0].size()):
                        flight1 = int (coarseTraj[I][J][K][0][l])
                        lat1 = coarseTraj[I][J][K][2][l]
                        lon1 = coarseTraj[I][J][K][3][l]
                        time1 = coarseTraj[I][J][K][4][l]
                        m = 0
                        for con in conflicts:
                            Ip = con[0]
                            Jp = con[1]
                            Kp = con[2]
                            flight2 = int(coarseTraj[Ip][Jp][Kp][0][m])
                            lat2 = coarseTraj[Ip][Jp][Kp][2][m]
                            lon2 = coarseTraj[Ip][Jp][Kp][3][m]
                            time2 = coarseTraj[Ip][Jp][Kp][4][m]
                            if flight1 != flight2:
                                isConflict = getPointConflict(lat1, lon1, time1, lat2, lon2, time2, spaceThreshold, temporalThreshold, earthRadius)
                                m = m + 1
                                if isConflict:
                                    f.write('%i, %i, %i, %f, %f, %f, %f, %f, %f\n' % (c, coarseTraj[I][J][K][0][l], coarseTraj[Ip][Jp][Kp][0][m], lat1, lon1, time1, lat2, lon2, time2))
                                    c = c + 1
