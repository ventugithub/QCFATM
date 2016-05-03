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
    rangeI = np.array((I, I + 1), dtype=int)
    rangeJ = np.array((J, J + 1), dtype=int)
    rangeK = np.array((K, K + 1), dtype=int)
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

def detectConflicts(flightIndices, times, lat, lon, mindistance, mintime):
    """ Detect conflicts

    Arguments:
    flightIndices: Array of consecutive flight indices starting from 0
    times: array of trajectory times (same length as flightIndices)
    lat: array of trajectory latitudes (same length as flightIndices)
    lon: array of trajectory longitudes (same length as flightIndices)
    mindistance: minimum distance in nautic miles to qualify as a conflict
    mintime: minimum time difference in minutes to qualify as a conflict

    returns: Pandas DataFrame containing the raw point conflicts with columns
            1. conflictIndex
            2. flight1
            3. flight2
            4. latitude1
            5. latitude2
            6. longitude1
            7. longitude2
            8. time1
            9. time2
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
    pointConflicts.drop_duplicates(inplace=True)

    # enforce flight1 < flight2 and remove duplicated
    pc1 = pointConflicts[pointConflicts['flight1'] > pointConflicts['flight2']]
    pc1.columns = ['flight2', 'flight1', 'lat2', 'lat1', 'lon2', 'lon1', 'time2', 'time1']
    pc2 = pointConflicts[pointConflicts['flight1'] <= pointConflicts['flight2']]
    pointConflicts = pd.concat([pc1, pc2])
    pointConflicts.drop_duplicates(inplace=True)
    # reset conflict index
    pointConflicts.reset_index(drop=True, inplace=True)
    pointConflicts.index.rename('conflictIndex', inplace=True)
    pointConflicts.sort(['flight1', 'flight2', 'time1', 'time2'], inplace=True)
    return pointConflicts


def parsePointConflicts(rawPointConflicts, deltaT=1):
    """ Given the raw point conflicts, group conflicts

    Arguments:
        rawPointConflicts: Pandas DataFrame with columns:
            1. conflictIndex
            2. flight1
            3. flight2
            4. latitude1
            5. latitude2
            6. longitude1
            7. longitude2
            8. time1
            9. time2

        deltaT: time difference determining if two consecutive raw point conflicts
                belong to the same parallel conflict

    Returns: Tuple of two Pandas DataFrame
        pointConflicts: Pandas DataFrame with columns:
            1. conflictIndex
            2. flight1
            3. flight2
            4. latitude1
            5. latitude2
            6. longitude1
            7. longitude2
            8. time1
            9. time2

        parallelConflicts: Pandas DataFrame with columns:
            1. parallelConflict index
            2. flight1
            3. flight2
            4. latitude1
            5. latitude2
            6. longitude1
            7. longitude2
            8. time1
            9. time2
    """
    # consecutive flight indices
    flight1 = np.array(rawPointConflicts['flight1'], dtype=int)
    flight2 = np.array(rawPointConflicts['flight2'], dtype=int)
    # times in minutes
    time1 = np.array(rawPointConflicts['time1'], dtype=int)
    time2 = np.array(rawPointConflicts['time2'], dtype=int)
    # latitude
    lat1 = np.array(rawPointConflicts['lat1'], dtype=float)
    lat2 = np.array(rawPointConflicts['lat2'], dtype=float)
    # longitude
    lon1 = np.array(rawPointConflicts['lon1'], dtype=float)
    lon2 = np.array(rawPointConflicts['lon2'], dtype=float)

    parallelConflict = np.empty_like(flight1)

    cdef int i
    cdef int j
    cdef int f1
    cdef int f2
    i = 0
    active = False
    cdef int pc = 0
    while i < len(flight1) - 1:
        # check for parallel conflicts
        f1Same = flight1[i + 1] == flight1[i]
        f2Same = flight2[i + 1] == flight2[i]
        t1Subsequent = time1[i + 1] - time1[i] <= deltaT
        t2Subsequent = time2[i + 1] - time2[i] <= deltaT
        if (f1Same and f2Same and t1Subsequent and t2Subsequent):
            if not active:
                pc = pc + 1
            parallelConflict[i] = pc
            i = i + 1
            parallelConflict[i] = pc
            active = True
        elif active:
            parallelConflict[i] = pc
            active = False
            i = i + 1
        else:
            parallelConflict[i] = 0
            i = i + 1
    # check the last row
    i = len(flight1) - 1
    if active:
        parallelConflict[i] = pc
        active = False
    else:
        parallelConflict[i] = 0

    rawPointConflicts['parallelConflict'] = parallelConflict

    # get all potential point conflicts
    pointConflicts = rawPointConflicts[rawPointConflicts['parallelConflict'] == 0]
    pointConflicts.reset_index(drop=True, inplace=True)
    pointConflicts = pointConflicts.drop('parallelConflict', axis=1)
    pointConflicts.index.rename('conflictIndex', inplace=True)
    parallelConflicts = rawPointConflicts[rawPointConflicts['parallelConflict'] != 0]
    parallelConflicts.loc[:, 'parallelConflict'] = parallelConflicts['parallelConflict'].apply(lambda x: x - 1)
    parallelConflicts.set_index('parallelConflict', drop=True, inplace=True)
    return pointConflicts, parallelConflicts


def getFlightConflicts(pointConflicts, parallelConflicts):
    """ Given the point and parallel conflicts, calculate
    the conflicts for each flight and order them by their
    temporal appearance

    Arguments:
        pointConflicts: Pandas DataFrame with columns:
            1. conflictIndex
            2. flight1
            3. flight2
            4. latitude1
            5. latitude2
            6. longitude1
            7. longitude2
            8. time1
            9. time2

        parallelConflicts: Pandas DataFrame with columns:
            1. parallelConflict index
            2. flight1
            3. flight2
            4. latitude1
            5. latitude2
            6. longitude1
            7. longitude2
            8. time1
            9. time2
    Returns:
        Pandas panel containing the mapping from the flight index
        to the conflicts (in temporal order)
        first dimension: flight indices
        second and third dimension: Pandas DataFrame with columns
            1. consecutive conflict index
            2. arrival time
            3. partner flight
    """
    pointConflictIndex = np.array(pointConflicts.index, dtype=int)
    flight1 = np.array(pointConflicts['flight1'], dtype=int)
    flight2 = np.array(pointConflicts['flight2'], dtype=int)
    time1 = np.array(pointConflicts['time1'], dtype=int)
    time2 = np.array(pointConflicts['time2'], dtype=int)
    parallelConflictIndex = np.array(parallelConflicts.index, dtype=int)
    pflight1 = np.array(parallelConflicts['flight1'], dtype=int)
    pflight2 = np.array(parallelConflicts['flight2'], dtype=int)
    ptime1 = np.array(parallelConflicts['time1'], dtype=int)
    ptime2 = np.array(parallelConflicts['time2'], dtype=int)

    cdef int i
    cdef int N1 = len(flight1)
    cdef int N2 = len(pflight1)
    pFlightsUnique = pd.concat([parallelConflicts['flight1'], parallelConflicts['flight2']]).unique()
    flightsUnique = np.unique(np.append(pd.concat([pointConflicts['flight1'], pointConflicts['flight2']]).unique(), pFlightsUnique))
    cdef int N = max(flight1.max(), pflight1.max())
    cdef int M = max(flight2.max(), pflight2.max())
    N = max(N, M) + 1

    cdef vector[vector[vector[int]]] conflicts
    conflicts.resize(N)
    for i in range(N):
        conflicts[i].resize(4)

    print 'Calculate mapping from flight index to point conflicts ...'
    for i in range(N1):
        conflicts[flight1[i]][0].push_back(pointConflictIndex[i])
        conflicts[flight1[i]][1].push_back(time1[i])
        conflicts[flight1[i]][2].push_back(flight2[i])
        conflicts[flight1[i]][3].push_back(False)
        conflicts[flight2[i]][0].push_back(pointConflictIndex[i])
        conflicts[flight2[i]][1].push_back(time2[i])
        conflicts[flight2[i]][2].push_back(flight1[i])
        conflicts[flight2[i]][3].push_back(False)

    print 'Calculate mapping from flight index to parallel conflicts ...'
    conflicts[pflight1[0]][0].push_back(parallelConflictIndex[0])
    conflicts[pflight1[0]][1].push_back(time1[0])
    conflicts[pflight1[0]][2].push_back(flight2[0])
    conflicts[pflight1[0]][3].push_back(True)
    conflicts[pflight2[0]][0].push_back(parallelConflictIndex[0])
    conflicts[pflight2[0]][1].push_back(time2[0])
    conflicts[pflight2[0]][2].push_back(flight1[0])
    conflicts[pflight2[0]][3].push_back(True)
    for i in range(1, N2):
        if (parallelConflictIndex[i] != parallelConflictIndex[i - 1]):
            conflicts[pflight1[i]][0].push_back(parallelConflictIndex[i])
            conflicts[pflight1[i]][1].push_back(ptime1[i])
            conflicts[pflight1[i]][2].push_back(pflight2[i])
            conflicts[pflight1[i]][3].push_back(True)
            conflicts[pflight2[i]][0].push_back(parallelConflictIndex[i])
            conflicts[pflight2[i]][1].push_back(ptime2[i])
            conflicts[pflight2[i]][2].push_back(pflight1[i])
            conflicts[pflight2[i]][3].push_back(True)

    print 'Convert mapping from flight index to parallel conflicts to data frame ...'
    pbar = progressbar.ProgressBar().start()
    pbar.maxval = len(flightsUnique)
    cdef int n = 0
    flight2Conflict = {}
    for flight in flightsUnique:
        print flight
        #pbar.update(n)
        n = n + 1
        con = pd.DataFrame({'conflictIndex': np.array(conflicts[flight][0], dtype=int) + np.array(conflicts[flight][3], dtype=int) * N1,
                            'arrivalTime': np.array(conflicts[flight][1], dtype=int),
                            'partnerFlight': np.array(conflicts[flight][2], dtype=int)},
                             columns=('conflictIndex', 'arrivalTime', 'partnerFlight'),
                            dtype=int
                           )
        flight2Conflict[flight] = con
    pbar.finish()

    f2c = pd.Panel.from_dict(flight2Conflict)
    return f2c
