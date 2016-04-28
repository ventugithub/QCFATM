import numpy as np
import pandas as pd
import cython
import math
from libc.math cimport sin, cos, acos, fabs
from libcpp.vector cimport vector
import progressbar


cdef mapToCoarseGrid(float lat, float lon, float time, float latMin, float lonMin, float timeMin, float deltaLat=0.5, float deltaLon=0.5, float deltaTime=60.0):
    """ map trajectory point to coarse grid

    Arguments:
    lat: latitude of the input trajectory point
    lon: longitude of the input trajectory point
    time: time of the input trajectory point
    latMin: minimum latitude of all trajectory points
    lonMin: minimum longitude of all trajectory points
    timeMin: minimum time of all trajectory points
    deltaLat: coarse grid step in latitude direction in degrees
    deltaLon: coarse grid step in longitude direction in degrees
    deltaTime: coarse grid step in in time

    returns: (I, J, K)
    I: coarse grid index in latitude direction
    J: coarse grid index in longitude direction
    K: coarse grid index in time direction
    """
    cdef int I = int((lat - latMin) / deltaLat)
    cdef int J = int((lon - lonMin) / deltaLon)
    cdef int T = int((time - timeMin) / deltaTime)
    return I, J, T

cdef getCoarsePointConflict(vector[vector[vector[vector[vector[float]]]]] & coarseTraj, int I, int J, int K):
    """
    Given a coarse grid cell (I, J, K), return all neighboring
    coarse grid cells which contain trajactory points

    Arguments:
    coarseTraj: a mapping from each trajectory point to
                a coarse grid. It is designed as follows:
                Dimension | Content
                    1     | coarse grid index in latitude direction
                    2     | coarse grid index in longitude direction
                    3     | coarse grid index in time direction
                    4     | 5-dimensional trajectory point information
                          |  1. consecutive flight index
                          |  2. row index of the original trajectory data
                          |  3. exact latitude
                          |  4. exact longitude
                          |  5. exact time
                    5     | index indicating the different trajectory points
                          | inside a coarse grid cell
                Example: coarseTraj(3, 4, 5, 2, 15) is the exact latitude of
                the 15th trajectory point in the coarse grid cell (3, 4, 5).
                I.e. with latitude index 3, longitude index 4, and time index 5

    I: coarse grid index in latitude direction
    J: coarse grid index in longitude direction
    K: coarse grid index in time direction
    """
    rangeI = np.array((I - 1, I, I + 1), dtype=int)
    rangeJ = np.array((J - 1, J, J + 1), dtype=int)
    rangeK = np.array((K - 1, K, K + 1), dtype=int)
    cdef int i
    cdef int j
    cdef int k
    cdef vector[vector[int]] conflicts
    conflicts.resize(3)
    for i in rangeI:
        for j in rangeJ:
            for k in rangeK:
                if coarseTraj[i][j][k][0].size() != 0:
                    conflicts[0].push_back(i)
                    conflicts[1].push_back(j)
                    conflicts[2].push_back(k)
    return conflicts

cdef getPointConflict(float lat1, float lon1, float time1, float lat2, float lon2, float time2, float spaceThreshold=55.56, float timeThreshold=60.0, float earthRadius=6000):
    """ Given two trajectory points (lat1, lon1, time1) and (lat2, lon2, time2)
    calculate if there is a conflict

    Arguments:

    lat1: latitude of the first trajectory point
    lon1: longitude of the first trajectory point
    time1: time of the first trajectory point
    lat2: latitude of the second trajectory point
    lon2: longitude of the second trajectory point
    time2: time of the second trajectory point
    spaceThreshold: minimal distance in kilometer to avoid conflict
    timeThreshold: minimal time difference in minutes to avoid conflict
    earthRadius: earth radius in kilometer
    """

    CosD = sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * (cos(lon1) * cos(lon2) + sin(lon1) * sin(lon2))
    CosD = min(CosD, 1)
    CosD = max(CosD, -1)
    spatialDistance = earthRadius * acos(CosD)
    temporalDistance = fabs(time1 - time2)
    if spatialDistance < spaceThreshold and temporalDistance < timeThreshold:
        return True
    else:
        return False

def detectConflicts(flightIndices, times, lat, lon, pointConflictFile, mindistance, mintime):
    """ Detect conflicts

    Arguments:
    flightIndices: Array of consecutive flight indices starting from 0
    times: array of trajectory times (same length as flightIndices)
    lat: array of trajectory latitudes (same length as flightIndices)
    lon: array of trajectory longitudes (same length as flightIndices)
    pointConflictFile: output file name
    mindistance: minimum distance in nautic miles to qualify as a conflict
    mintime: minimum time difference in minutes to qualify as a conflict
    """
    #######################################################
    # constants
    #######################################################
    # grid discretization for latitude
    cdef float deltaLat = 0.5
    # grid discretization for longitude
    cdef float deltaLon = 0.5
    # grid discretizaiton for time
    cdef int deltaTime = 60
    # earth radius in kilometers
    cdef float earthRadius = 6367.0
    # minimal acceptable distance in kilometers
    cdef float spaceThreshold = mindistance
    # minimal acceptable time difference in minutes
    cdef float temporalThreshold = mintime


    # consecutive flight indices
    flights = np.array(flightIndices, dtype=int)
    # times in minutes
    times = np.array(times, dtype=int)
    # latitude
    lat = np.array(lat, dtype=float)
    # longitude
    lon = np.array(lon, dtype=float)
    # minimal latitude value
    cdef float latMin = lat.min()
    # maximal latitude value
    cdef float latMax = lat.max()
    # minimal longitude value
    cdef float lonMin = lon.min()
    # maximal longitude value
    cdef float lonMax = lon.max()
    # minimal time value
    cdef float timeMin = times.min()
    # maximal time value
    cdef float timeMax = times.max()
    # number of grid points in latitude direction
    cdef unsigned int Nlat = int((latMax - latMin) / deltaLat) + 1
    # number of grid points in longitude direction
    cdef unsigned int Nlon = int((lonMax - lonMin) / deltaLon) + 1
    # number of grid points in time
    cdef unsigned int Ntime = int((timeMax - timeMin) / deltaTime) + 1
    # number of trajectory points
    cdef unsigned int N = len(flightIndices)

    # cython typed variable definitions (for speed)
    cdef unsigned int i
    cdef unsigned int n = 0
    cdef unsigned int m = 0
    cdef unsigned int j
    cdef unsigned int l
    cdef unsigned int k
    cdef unsigned int lmin
    cdef unsigned int lmax
    cdef unsigned int kmin
    cdef unsigned int kmax
    cdef unsigned int c = 0
    cdef unsigned int flight1
    cdef unsigned int flight2
    cdef unsigned int I
    cdef unsigned int J
    cdef unsigned int K
    cdef unsigned int Ip
    cdef unsigned int Jp
    cdef unsigned int Kp
    cdef float lat1
    cdef float lon1
    cdef float time1
    cdef float lat2
    cdef float lon2
    cdef float time2
    cdef vector[vector[int]] conflicts

    #########################################################
    ##### prepare coarse trajectory grid container ##########
    #########################################################
    # coarseTraj is a mapping from each trajectory point to
    # a coarse grid. It is designed as follows:
    # Dimension | Content
    #     1     | coarse grid index in latitude direction
    #     2     | coarse grid index in longitude direction
    #     3     | coarse grid index in time direction
    #     4     | 5-dimensional trajectory point information
    #           |  1. consecutive flight index
    #           |  2. row index of the original trajectory data
    #           |  3. exact latitude
    #           |  4. exact longitude
    #           |  5. exact time
    #     5     | index indicating the different trajectory points
    #           | inside a coarse grid cell
    #
    #
    # Example: coarseTraj(3, 4, 5, 2, 15) is the exact latitude of
    # the 15th trajectory point in the coarse grid cell (3, 4, 5).
    # I.e. with latitude index 3, longitude index 4, and time index 5
    print "Prepare coarse trajectory container ..."
    cdef vector[vector[vector[vector[vector[float]]]]] coarseTraj
    coarseTraj.resize(Nlat)
    for i in range(0, Nlat):
        coarseTraj[i].resize(Nlon)
        for j in range(0, Nlon):
            coarseTraj[i][j].resize(Ntime)
            for k in range(0, Ntime):
                coarseTraj[i][j][k].resize(5)


    ###################################################
    #### mapping to coarse trajectory container #######
    ###################################################
    print "Map trajectory point to coarse grid ..."
    for i in range(1, len(flights)):
        #if i % 1000 == 0:
            #print i, " of ",  len(flights)
        # mapping to coarse trajectory container
        I, J, K = mapToCoarseGrid(lat[i], lon[i], times[i], latMin, lonMin, timeMin, deltaLat, deltaLon, deltaTime)
        coarseTraj[I][J][K][0].push_back(flights[i])
        coarseTraj[I][J][K][1].push_back(i)
        coarseTraj[I][J][K][2].push_back(lat[i])
        coarseTraj[I][J][K][3].push_back(lon[i])
        coarseTraj[I][J][K][4].push_back(times[i])

    ##################################################
    ### calculate point conflicts ####################
    ##################################################
    cdef vector[int] pcIndex
    cdef vector[int] pcFlight1
    cdef vector[int] pcFlight2
    cdef vector[float] pcLat1
    cdef vector[float] pcLon1
    cdef vector[float] pcTime1
    cdef vector[float] pcLat2
    cdef vector[float] pcLon2
    cdef vector[float] pcTime2
    # conflict number
    c = 0
    # progress bar
    Nloops = (Nlat - 1) * (Nlon - 1) * (Ntime - 1)
    print 'Calculate point conflicts'
    pbar = progressbar.ProgressBar().start()
    pbar.maxval = Nloops
    # looping over all coarse grid cells
    n = 0
    for I in range(1, Nlat - 1):
        for J in range(1, Nlon - 1):
            for K in range(1, Ntime - 1):
                if n % 10000 == 0:
                    pbar.update(n)
                n = n + 1
                # get all coarse grid cells in vicinity of the
                # current coarse grid cell (I, J, K) which contain
                # trajectory point as a list of 3-tuples (i, j, k)
                conflicts = getCoarsePointConflict(coarseTraj, I, J, K)
                if conflicts[0].size() != 0:
                    # loop over all trajectory point in the current coarse grid cell
                    for l in range(coarseTraj[I][J][K][0].size()):
                        flight1 = int (coarseTraj[I][J][K][0][l])
                        lat1 = coarseTraj[I][J][K][2][l]
                        lon1 = coarseTraj[I][J][K][3][l]
                        time1 = coarseTraj[I][J][K][4][l]
                        # loop over all trajectory points in the neigboring coarse gri cells
                        for i in range(conflicts[0].size()):
                            Ip = conflicts[0][i]
                            Jp = conflicts[1][i]
                            Kp = conflicts[2][i]
                            for m in range(coarseTraj[Ip][Jp][Kp][0].size()):
                                flight2 = int(coarseTraj[Ip][Jp][Kp][0][m])
                                lat2 = coarseTraj[Ip][Jp][Kp][2][m]
                                lon2 = coarseTraj[Ip][Jp][Kp][3][m]
                                time2 = coarseTraj[Ip][Jp][Kp][4][m]
                                # if the flight number is different, check if there is a point conflict and write the information to a text file
                                if flight1 != flight2:
                                    isConflict = getPointConflict(lat1, lon1, time1, lat2, lon2, time2, spaceThreshold, temporalThreshold, earthRadius)
                                    if isConflict:
                                        pcIndex.push_back(c)
                                        pcFlight1.push_back(flight1)
                                        pcFlight2.push_back(flight2)
                                        pcLat1.push_back(lat1)
                                        pcLon1.push_back(lon1)
                                        pcTime1.push_back(time1)
                                        pcLat2.push_back(lat2)
                                        pcLon2.push_back(lon2)
                                        pcTime2.push_back(time2)
                                        c = c + 1
    pbar.finish()
    np.array(pcIndex)
    pointConflicts = pd.DataFrame({'conflictIndex': np.array(pcIndex),
                                   'flight1': np.array(pcFlight1),
                                   'flight2': np.array(pcFlight2),
                                   'lat1': np.array(pcLat1),
                                   'lon1': np.array(pcLon1),
                                   'time1': np.array(pcTime1),
                                   'lat2': np.array(pcLat2),
                                   'lon2': np.array(pcLon2),
                                   'time2': np.array(pcTime2)
                                   })
    pointConflicts = pointConflicts.set_index('conflictIndex')
    # remove duplicates
    pointConflictsSwap = pointConflicts.copy()
    pointConflictsSwap.columns=['flight2', 'flight1', 'lat2', 'lat1', 'lon2', 'lon1', 'time2', 'time1']
    pointConflicts = pd.concat([pointConflicts, pointConflictsSwap])
    pointConflicts.drop_duplicates(inplace=True)
    # save to csv file
    pointConflicts.to_csv(pointConflictFile, mode='w')
    print "Point conflict data written to", pointConflictFile
    return pointConflicts
