import numpy as np
import pandas as pd
import cython
import math
from libc.math cimport sin, cos, acos, fabs
from libcpp.vector cimport vector
from libcpp cimport bool
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

cdef getCoarseRawPointConflict(vector[vector[vector[vector[vector[float]]]]] & coarseTraj, int I, int J, int K, int Imax, int Jmax, int Kmax):
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
    Imax: Maximum coarse grid index in latitude direction + 1
    Jmax: Maximum coarse grid index in longitude direction + 1
    Kmax: Maximum coarse grid index in time direction + 1
    """
    rangeI = np.arange(max(I - 1, 0), min(I + 2, Imax - 1), dtype=int)
    rangeJ = np.arange(max(J - 1, 0), min(J + 2, Jmax - 1), dtype=int)
    rangeK = np.arange(max(K - 1, 0), min(K + 2, Kmax - 1), dtype=int)
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

cdef getRawPointConflict(float lat1, float lon1, float alt1, float time1,
                      float lat2, float lon2, float alt2, float time2,
                      float spaceThreshold, float timeThreshold,
                      float altitudeThreshold, float earthRadius,
                      float maxDelay1=0, float maxDelay2=0):
    """ Given two trajectory points (lat1, lon1, alt1, time1) and (lat2, lon2, alt2, time2)
    calculate if there is a conflict.

    Arguments:

    lat1: latitude of the first trajectory point
    lon1: longitude of the first trajectory point
    alt1: altitude of the first trajectory point
    time1: time of the first trajectory point
    lat2: latitude of the second trajectory point
    lon2: longitude of the second trajectory point
    alt2: altitude of the second trajectory point
    time2: time of the second trajectory point
    spaceThreshold: minimal distance in kilometer to avoid conflict
    timeThreshold: temporal threshold below which a conflict is considered real
    earthRadius: earth radius in kilometer
    maxDelay1: maximal delay of first flight
    maxDelay2: maximal delay of second flight
    """

    CosD = sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * (cos(lon1) * cos(lon2) + sin(lon1) * sin(lon2))
    CosD = min(CosD, 1)
    CosD = max(CosD, -1)
    spatialDistance = earthRadius * acos(CosD)
    altitudeDistance = fabs(alt1 - alt2)

    if spatialDistance < spaceThreshold and altitudeDistance < altitudeThreshold:
        if time1 >= time2:
            if time1 - time2 - maxDelay1 < timeThreshold:
                return True
            else:
                return False
        else:
            if time2 - time1 - maxDelay2 < timeThreshold:
                return True
            else:
                return False
    else:
        return False

def detectRawConflicts(flightIndices, times, lat, lon, alt, mindistance, mintime):
    """ Detect conflicts

    Arguments:
    flightIndices: Array of consecutive flight indices starting from 0
    times: array of trajectory times (same length as flightIndices)
    lat: array of trajectory latitudes (same length as flightIndices)
    lon: array of trajectory longitudes (same length as flightIndices)
    alt: array of trajectory altitudes (same length as flightIndices)
         We assume only a few distinct values of the altitude
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
            8. alt1
            9. alt2
            10. time1
            11. time2
    """
    #######################################################
    # constants
    #######################################################
    # grid discretization for latitude
    cdef float deltaLat = np.radians(2)
    # grid discretization for longitude
    cdef float deltaLon = np.radians(2)
    # grid discretizaiton for time
    cdef int deltaTime = 60
    # earth radius in kilometers
    cdef float earthRadius = 6371.21
    # minimal acceptable altitude difference in feet
    cdef float altitudeThreshold = 1000.0
    # minimal acceptable distance in kilometers
    cdef float spaceThreshold = mindistance
    # minimal acceptable time difference in minutes
    cdef float temporalThreshold = float(mintime)


    # consecutive flight indices
    flights = np.array(flightIndices, dtype=int)
    # times in minutes
    times = np.array(times, dtype=int)
    # latitude
    lat = np.array(lat, dtype=float)
    # longitude
    lon = np.array(lon, dtype=float)
    # altitude
    alt = np.array(alt, dtype=float)
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
    cdef float alt1
    cdef float time1
    cdef float lat2
    cdef float lon2
    cdef float alt2
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
                coarseTraj[i][j][k].resize(6)


    ###################################################
    #### mapping to coarse trajectory container #######
    ###################################################
    print "Map trajectory point to coarse grid ..."
    for i in range(0, len(flights)):
        # mapping to coarse trajectory container
        I, J, K = mapToCoarseGrid(lat[i], lon[i], times[i], latMin, lonMin, timeMin, deltaLat, deltaLon, deltaTime)
        coarseTraj[I][J][K][0].push_back(flights[i])
        coarseTraj[I][J][K][1].push_back(i)
        coarseTraj[I][J][K][2].push_back(lat[i])
        coarseTraj[I][J][K][3].push_back(lon[i])
        coarseTraj[I][J][K][4].push_back(alt[i])
        coarseTraj[I][J][K][5].push_back(times[i])

    ##################################################
    ### calculate point conflicts ####################
    ##################################################
    cdef vector[int] pcIndex
    cdef vector[int] pcFlight1
    cdef vector[int] pcFlight2
    cdef vector[float] pcLat1
    cdef vector[float] pcLon1
    cdef vector[float] pcAlt1
    cdef vector[float] pcTime1
    cdef vector[float] pcLat2
    cdef vector[float] pcLon2
    cdef vector[float] pcAlt2
    cdef vector[float] pcTime2
    # conflict number
    c = 0
    # progress bar
    Nloops = Nlat * Nlon * Ntime
    print 'Calculate point conflicts'
    pbar = progressbar.ProgressBar().start()
    pbar.maxval = Nloops
    pbarstep = Nloops/100
    # looping over all coarse grid cells
    n = 0
    for I in range(0, Nlat):
        for J in range(0, Nlon):
            for K in range(0, Ntime):
                if n % pbarstep == 0:
                    pbar.update(n)
                n = n + 1
                # get all coarse grid cells in vicinity of the
                # current coarse grid cell (I, J, K) which contain
                # trajectory point as a list of 3-tuples (i, j, k)
                conflicts = getCoarseRawPointConflict(coarseTraj, I, J, K, Nlat, Nlon, Ntime)
                # loop over all trajectory point in the current coarse grid cell
                for l in range(coarseTraj[I][J][K][0].size()):
                    flight1 = int(coarseTraj[I][J][K][0][l])
                    lat1 = coarseTraj[I][J][K][2][l]
                    lon1 = coarseTraj[I][J][K][3][l]
                    alt1 = coarseTraj[I][J][K][4][l]
                    time1 = coarseTraj[I][J][K][5][l]
                    # loop over all trajectory points in the neigboring coarse gri cells
                    for i in range(conflicts[0].size()):
                        Ip = conflicts[0][i]
                        Jp = conflicts[1][i]
                        Kp = conflicts[2][i]
                        for m in range(coarseTraj[Ip][Jp][Kp][0].size()):
                            flight2 = int(coarseTraj[Ip][Jp][Kp][0][m])
                            lat2 = coarseTraj[Ip][Jp][Kp][2][m]
                            lon2 = coarseTraj[Ip][Jp][Kp][3][m]
                            alt2 = coarseTraj[Ip][Jp][Kp][4][m]
                            time2 = coarseTraj[Ip][Jp][Kp][5][m]
                            # if the flight number is different, check if there is a point conflict and write the information to a text file
                            if flight1 != flight2:
                                isConflict = getRawPointConflict(lat1, lon1, alt1, time1, lat2, lon2, alt2, time2,
                                                                    spaceThreshold=spaceThreshold, timeThreshold=temporalThreshold,
                                                                    altitudeThreshold=altitudeThreshold, earthRadius=earthRadius)
                                if isConflict:
                                    if flight1 < flight2:
                                        pcIndex.push_back(c)
                                        pcFlight1.push_back(flight1)
                                        pcFlight2.push_back(flight2)
                                        pcLat1.push_back(lat1)
                                        pcLon1.push_back(lon1)
                                        pcAlt1.push_back(alt1)
                                        pcTime1.push_back(time1)
                                        pcLat2.push_back(lat2)
                                        pcLon2.push_back(lon2)
                                        pcAlt2.push_back(alt2)
                                        pcTime2.push_back(time2)
                                    else:
                                        pcIndex.push_back(c)
                                        pcFlight1.push_back(flight2)
                                        pcFlight2.push_back(flight1)
                                        pcLat1.push_back(lat2)
                                        pcLon1.push_back(lon2)
                                        pcAlt1.push_back(alt2)
                                        pcTime1.push_back(time2)
                                        pcLat2.push_back(lat1)
                                        pcLon2.push_back(lon1)
                                        pcAlt2.push_back(alt1)
                                        pcTime2.push_back(time1)
                                    c = c + 1
    pbar.finish()
    np.array(pcIndex)
    pointConflicts = pd.DataFrame({'conflictIndex': np.array(pcIndex),
                                   'flight1': np.array(pcFlight1),
                                   'flight2': np.array(pcFlight2),
                                   'lat1': np.array(pcLat1),
                                   'lon1': np.array(pcLon1),
                                   'alt1': np.array(pcAlt1),
                                   'time1': np.array(pcTime1),
                                   'lat2': np.array(pcLat2),
                                   'lon2': np.array(pcLon2),
                                   'alt2': np.array(pcAlt2),
                                   'time2': np.array(pcTime2)
                                   })
    pointConflicts = pointConflicts.set_index('conflictIndex')
    # remove duplicates
    pointConflicts.drop_duplicates(subset=['flight1', 'flight2', 'time1', 'time2'], inplace=True)

    pointConflicts.index.rename('conflictIndex', inplace=True)
    pointConflicts.sort_values(by=['flight1', 'flight2', 'time1', 'time2'], inplace=True)
    return pointConflicts


def parseRawPointConflicts(rawPointConflicts, deltaT=1):
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
            10. alt1
            11. alt2

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
            10. alt1
            11. alt2

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
            10. alt1
            11. alt2
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
    pointConflicts = rawPointConflicts[rawPointConflicts['parallelConflict'] == 0].copy()
    pointConflicts.reset_index(drop=True, inplace=True)
    pointConflicts = pointConflicts.drop('parallelConflict', axis=1)
    pointConflicts.index.rename('conflictIndex', inplace=True)
    parallelConflicts = rawPointConflicts[rawPointConflicts['parallelConflict'] != 0].copy()
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
            10. alt1
            11. alt2

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
            10. alt1
            11. alt2
    Returns:
        Pandas panel containing the mapping from the flight index
        to the conflicts (in temporal order)
        first dimension: flight indices
        second and third dimension: Pandas DataFrame with columns
            1. arrival time
            2. arrival time of the partner flight
            3. partner flight
            4. consecutive conflict index
            5. minimum time difference with partner, i.e. min(arrival times - arrival times partner)
            6. maximum time difference with partner, i.e. max(arrival times - arrival times partner)
    """
    pointConflictIndex = np.array(pointConflicts.index, dtype=int)
    flight1 = np.array(pointConflicts['flight1'], dtype=int)
    flight2 = np.array(pointConflicts['flight2'], dtype=int)
    time1 = np.array(pointConflicts['time1'], dtype=int)
    time2 = np.array(pointConflicts['time2'], dtype=int)
    timediff = np.array(pointConflicts['time1'] - pointConflicts['time2'], dtype=int)
    parallelConflictIndex = np.array(parallelConflicts.index, dtype=int)
    pflight1 = np.array(parallelConflicts['flight1'], dtype=int)
    pflight2 = np.array(parallelConflicts['flight2'], dtype=int)
    ptime1 = np.array(parallelConflicts['time1'], dtype=int)
    ptime2 = np.array(parallelConflicts['time2'], dtype=int)

    parallelConflicts['timediff'] = parallelConflicts['time1'] - parallelConflicts['time2']
    ptimediff = np.array(parallelConflicts['timediff'], dtype=int)

    cdef int i
    cdef int N1 = len(flight1)
    cdef int N2 = len(pflight1)
    pFlightsUnique = pd.concat([parallelConflicts['flight1'], parallelConflicts['flight2']]).unique()
    flightsUnique = np.unique(np.append(pd.concat([pointConflicts['flight1'], pointConflicts['flight2']]).unique(), pFlightsUnique))
    cdef int N = max(flight1.max() if flight1.shape[0] else 0, pflight1.max() if pflight1.shape[0] else 0)
    cdef int M = max(flight2.max() if flight2.shape[0] else 0, pflight2.max() if pflight2.shape[0] else 0)
    N = max(N, M) + 1

    cdef vector[vector[vector[int]]] conflicts
    conflicts.resize(N)
    for i in range(N):
        conflicts[i].resize(6)

    print 'Calculate mapping from flight index to point conflicts ...'
    for i in range(N1):
        conflicts[flight1[i]][0].push_back(pointConflictIndex[i])
        conflicts[flight1[i]][1].push_back(time1[i])
        conflicts[flight1[i]][2].push_back(flight2[i])
        conflicts[flight1[i]][3].push_back(time2[i])
        conflicts[flight1[i]][4].push_back(timediff[i])
        conflicts[flight1[i]][5].push_back(timediff[i])
        conflicts[flight2[i]][0].push_back(pointConflictIndex[i])
        conflicts[flight2[i]][1].push_back(time2[i])
        conflicts[flight2[i]][2].push_back(flight1[i])
        conflicts[flight2[i]][3].push_back(time1[i])
        conflicts[flight2[i]][4].push_back(-timediff[i])
        conflicts[flight2[i]][5].push_back(-timediff[i])

    print 'Calculate mapping from flight index to parallel conflicts ...'
    if N2 > 0:
        conflicts[pflight1[0]][0].push_back(parallelConflictIndex[0] + N1)
        conflicts[pflight1[0]][1].push_back(ptime1[0])
        conflicts[pflight1[0]][2].push_back(pflight2[0])
        conflicts[pflight1[0]][3].push_back(ptime2[0])
        conflicts[pflight1[0]][4].push_back(parallelConflicts.loc[0]['timediff'].min())
        conflicts[pflight1[0]][5].push_back(parallelConflicts.loc[0]['timediff'].max())
        conflicts[pflight2[0]][0].push_back(parallelConflictIndex[0] + N1)
        conflicts[pflight2[0]][1].push_back(ptime2[0])
        conflicts[pflight2[0]][2].push_back(pflight1[0])
        conflicts[pflight2[0]][3].push_back(ptime1[0])
        conflicts[pflight2[0]][4].push_back(-parallelConflicts.loc[0]['timediff'].min())
        conflicts[pflight2[0]][5].push_back(-parallelConflicts.loc[0]['timediff'].max())
    for i in range(1, N2):
        if (parallelConflictIndex[i] != parallelConflictIndex[i - 1]):
            conflicts[pflight1[i]][0].push_back(parallelConflictIndex[i] + N1)
            conflicts[pflight1[i]][1].push_back(ptime1[i])
            conflicts[pflight1[i]][2].push_back(pflight2[i])
            conflicts[pflight1[i]][3].push_back(ptime2[i])
            conflicts[pflight1[i]][4].push_back(parallelConflicts.loc[parallelConflictIndex[i]]['timediff'].min())
            conflicts[pflight1[i]][5].push_back(parallelConflicts.loc[parallelConflictIndex[i]]['timediff'].max())
            conflicts[pflight2[i]][0].push_back(parallelConflictIndex[i] + N1)
            conflicts[pflight2[i]][1].push_back(ptime2[i])
            conflicts[pflight2[i]][2].push_back(pflight1[i])
            conflicts[pflight2[i]][3].push_back(ptime1[i])
            conflicts[pflight2[i]][4].push_back(-parallelConflicts.loc[parallelConflictIndex[i]]['timediff'].min())
            conflicts[pflight2[i]][5].push_back(-parallelConflicts.loc[parallelConflictIndex[i]]['timediff'].max())

    print 'Convert mapping from flight index to parallel conflicts to data frame ...'
    pbar = progressbar.ProgressBar().start()
    pbar.maxval = len(flightsUnique)
    cdef int n = 0
    flight2Conflict = {}
    for flight in flightsUnique:
        pbar.update(n)
        n = n + 1
        con = pd.DataFrame({'arrivalTime': np.array(conflicts[flight][1], dtype=int),
                            'arrivalTimePartner': np.array(conflicts[flight][3], dtype=int),
                            'conflictIndex': np.array(conflicts[flight][0], dtype=int),
                            'partnerFlight': np.array(conflicts[flight][2], dtype=int),
                            'minTimeDiffWithPartner': np.array(conflicts[flight][4], dtype=int),
                            'maxTimeDiffWithPartner': np.array(conflicts[flight][5], dtype=int)},
                             columns=('arrivalTime', 'arrivalTimePartner', 'conflictIndex', 'partnerFlight', 'minTimeDiffWithPartner', 'maxTimeDiffWithPartner'),
                            dtype=int
                           )
        con.sort_values(by=['arrivalTime', 'arrivalTimePartner'], inplace=True)
        con.reset_index(drop=True, inplace=True)
        flight2Conflict[flight] = con
    pbar.finish()

    f2c = pd.Panel.from_dict(flight2Conflict)
    return f2c

def reindexParallelConflicts(p):
    """ reset index after dropping rows from the parallel
    conflicts data frame. e.g.

    0 0 2 2 2 4 4 5 5 6 6 6
    ->
    0 0 1 1 1 2 2 3 3 4 4 4
    """
    pac = p.copy()
    pac['index'] = pac.index
    g =pac['index'].groupby(pac['index'])
    counts = g.count()
    newindex = np.zeros(len(pac.index), dtype=int)
    n = 0
    index = 0
    for i in counts:
        for j in range(i):
            newindex[n] = index
            n += 1
        index += 1
    pac['newindex'] = newindex
    pac.set_index('newindex', inplace=True)
    pac.drop('index', axis=1, inplace=True)
    pac.index.name = p.index.name
    return pac

def reduceConflicts(flight2Conflict, pointConflicts, parallelConflicts, delayPerConflictAvoidance=3, dthreshold=3, maxDepartDelay = 10):
    """ Reduce the number of conflicts by considering the maximal delay of
    each flight, when it reaches a certain conflict.

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
            10. alt1
            11. alt2

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
            10. alt1
            11. alt2
        flight2Conflict:
            Pandas panel containing the mapping from the flight index
            to the conflicts (in temporal order)
            first dimension: flight indices
            second and third dimension: Pandas DataFrame with columns
                1. arrival time
                2. arrival time of the partner flight
                3. partner flight
                4. consecutive conflict index
                5. minimum time difference with partner, i.e. min(arrival times - arrival times partner)
                6. maximum time difference with partner, i.e. max(arrival times - arrival times partner)

        delayPerConflictAvoidance: Delay introduced by each conflict avoiding maneuver
        dthreshold: temporal threshold below which a conflict is considered real
        maxDepartDelay: maximal delay at departure time
    Returns:
        pointConflicts and parallelConflicts with dropped conflicts
    """
    cdef int N1 = len(pointConflicts)

    print 'Reduce potential conflicts by considering maximal delay ...'
    pbar = progressbar.ProgressBar().start()
    pbar.maxval = len(flight2Conflict)
    cdef int n = 0
    dropPointConflicts = []
    dropParallelConflicts = []
    for flight, f2c in flight2Conflict.iteritems():
        pbar.update(n)
        n = n + 1
        df = f2c.dropna()
        for i, row in df.iterrows():
            minTimeDiffWithPartner = row['minTimeDiffWithPartner']
            maxTimeDiffWithPartner = row['maxTimeDiffWithPartner']
            partnerFlight = row['partnerFlight']
            conflictIndex = row['conflictIndex']
            dfp = flight2Conflict.loc[int(partnerFlight)].dropna()
            indexList = dfp.loc[dfp['conflictIndex']==conflictIndex].index.tolist()
            assert len(indexList) == 1
            NConflictsBeforePartner = indexList[0]
            NConflictsBefore = i
            keep = False
            # conflict possible if dthreshold - max(t1 - t2) < d_i - d_j < dthreshold - min(t1 - t2)
            # d_i in [0, delayPerConflictAvoidance * NConflictsBefore + maxDepartDelay]
            # d_j in [0, delayPerConflictAvoidance * NConflictsBeforePartner + maxDepartDelay]
            # => a conflict is possible if
            # dthreshold - max(t1, t2) < delayPerConflictAvoidance * NConflictsBefore + maxDepartDelay
            # - delayPerConflictAvoidance * NConflictsBeforePartner - maxDelay1 < dthreshold - min(t1, t2)
            if dthreshold - maxTimeDiffWithPartner < delayPerConflictAvoidance * NConflictsBefore + maxDepartDelay:
                keep = True
            elif -delayPerConflictAvoidance * NConflictsBeforePartner - maxDepartDelay < dthreshold - minTimeDiffWithPartner:
                keep = True
            if not keep:
                if conflictIndex < N1:
                    dropPointConflicts.append(conflictIndex)
                else:
                    dropParallelConflicts.append(conflictIndex - N1)
    pbar.finish()
    # drop conflicts
    poc = pointConflicts.drop(dropPointConflicts)
    pac = parallelConflicts.drop(dropParallelConflicts)
    poc.reset_index(drop=True, inplace=True)
    poc.index.rename('conflictIndex', inplace=True)
    pac = reindexParallelConflicts(pac)

    return poc, pac

def getMultiConflicts(pointConflicts, parallelConflicts, flight2Conflict, mindistance, delayPerConflictAvoidance=3, dthreshold=3, maxDepartDelay = 10):
    """ Calculate all non-pairwise conflicts. This is done in the folllowing way:
    The point conflicts as well as the parallel conflicts are composed of raw
    point conflicts, i.e. two conflicting trajectory points. A non-pairwise conflict
    can be detected by checking if the two trajectory points of a given conflict are
    in conflict with the two trajectory points of another conflict. By checking all
    combinations of point conflicts and parallel conflicts, we get all non-pairwise
    conflicts.

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
            10. alt1
            11. alt2

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
            10. alt1
            11. alt2

        flight2Conflict:
            Pandas panel containing the mapping from the flight index
            to the conflicts (in temporal order)
            first dimension: flight indices
            second and third dimension: Pandas DataFrame with columns
            1. arrival time
            2. arrival time of the partner flight
            3. partner flight
            4. consecutive conflict index

        mindistance: minimum distance in nautic miles to qualify as a conflict
        delayPerConflictAvoidance: Delay introduced by each conflict avoiding maneuver
        dthreshold: temporal threshold below which a conflict is considered real
        maxDepartDelay: maximal delay at departure time

    Returns:
        Pandas DataFrame with columns
        1. multiConflictIndex : consecutive index for the non-pairwise conflicts
        2. conflict1: first conflict index
        3. conflict2: second conflict index
        4. conflictType1: True for parallel conflict, false for point conflict
        5. conflictType2: True for parallel conflict, false for point conflict
        6. isConflict11: True if there is a point conflict between the 1st flight of the 1st conflict and the 1st flight of the 2nd conflict
        7. isConflict12: True if there is a point conflict between the 1st flight of the 1st conflict and the 2nd flight of the 2nd conflict
        8. isConflict21: True if there is a point conflict between the 2nd flight of the 1st conflict and the 1st flight of the 2nd conflict
        9. isConflict22: True if there is a point conflict between the 2nd flight of the 1st conflict and the 2nd flight of the 2nd conflict
        10. deltaTMin': Minimum time difference of all contributing point conflicts
        11. multiConflictType': 0 if both conflicts are point conflicts, 1 if both conflicts are parallel conflicts, 0.5 otherwise
    """

    #######################################################
    # constants
    #######################################################
    # earth radius in kilometers
    cdef float earthRadius = 6371.21
    # minimal acceptable altitude difference in feet
    cdef float altitudeThreshold = 1000.0
    # minimal acceptable distance in kilometers
    cdef float spaceThreshold = mindistance
    cdef float temporalThreshold = dthreshold

    # merge point and parallel conflicts to one data frame
    cdef int N1 = len(pointConflicts)
    parallelConflicts.index = parallelConflicts.index + N1
    parallelConflicts.index.name = 'conflictIndex'
    pointConflicts.index.name = 'conflictIndex'
    conflicts = pd.concat([pointConflicts, parallelConflicts])
    cdef int N = len(conflicts)

    # conflict data
    cdef vector[int] conflictIndex = conflicts.index.values
    cdef vector[int] flight1 = conflicts.flight1.values
    cdef vector[int] flight2 = conflicts.flight2.values
    cdef vector[int] time1 = conflicts.time1.values
    cdef vector[int] time2 = conflicts.time2.values
    cdef vector[int] lat1 = conflicts.lat1.values
    cdef vector[int] lat2 = conflicts.lat2.values
    cdef vector[int] lon1 = conflicts.lon1.values
    cdef vector[int] lon2 = conflicts.lon2.values
    cdef vector[int] alt1 = conflicts.alt1.values
    cdef vector[int] alt2 = conflicts.alt2.values

    # get maximal delay introduced by previous conflict avoiding maneuvers
    cdef vector[float] maxDelay1
    cdef vector[float] maxDelay2
    maxDelay1.resize(N)
    maxDelay2.resize(N)

    print "Get number of previous conflicts for each conflict and flight"
    cdef int i
    pbar = progressbar.ProgressBar().start()
    pbar.maxval = N
    for i in range(N):
        if i % 100 == 0:
            pbar.update(i)
        df = flight2Conflict.loc[flight1[i],:,:].dropna()
        indexList = df.loc[df['conflictIndex']==conflictIndex[i]].index.tolist()
        assert len(indexList) == 1
        NConflictsBefore = indexList[0]
        maxDelay1[i] = delayPerConflictAvoidance * NConflictsBefore + maxDepartDelay
        df = flight2Conflict.loc[flight2[i],:,:].dropna()
        indexList = df.loc[df['conflictIndex']==conflictIndex[i]].index.tolist()
        assert len(indexList) == 1
        NConflictsBefore = indexList[0]
        maxDelay2[i] = delayPerConflictAvoidance * NConflictsBefore + maxDepartDelay
    pbar.finish()

    # calculate multi conflicts
    cdef vector[int] multiConflictsFirst
    cdef vector[int] multiConflictsSecond
    cdef vector[bool] conflict11
    cdef vector[bool] conflict12
    cdef vector[bool] conflict21
    cdef vector[bool] conflict22
    cdef vector[float] deltaTMinVec
    cdef vector[float] conflictType
    cdef vector[float] conflictTypeFirst
    cdef vector[float] conflictTypeSecond
    print "Calculate conflict involving more than two flights"
    pbar = progressbar.ProgressBar().start()
    pbar.maxval = N
    cdef bool isConflict11
    cdef bool isConflict12
    cdef bool isConflict21
    cdef bool isConflict22
    cdef float deltaTMin
    cdef float deltaT
    for i in range(N):
        for j in range(N):
            if i % 100 == 0:
                pbar.update(i)
            if conflictIndex[i] != conflictIndex[j]:
                isConflict11 = getRawPointConflict(lat1[i], lon1[i], alt1[i], time1[i], lat1[j], lon1[j], alt1[j], time1[j],
                                                   spaceThreshold=spaceThreshold, timeThreshold=temporalThreshold,
                                                   altitudeThreshold=altitudeThreshold, earthRadius=earthRadius,
                                                   maxDelay1=maxDelay1[i], maxDelay2=maxDelay1[j])
                isConflict12 = getRawPointConflict(lat1[i], lon1[i], alt1[i], time1[i], lat2[j], lon2[j], alt2[j], time2[j],
                                                   spaceThreshold=spaceThreshold, timeThreshold=temporalThreshold,
                                                   altitudeThreshold=altitudeThreshold, earthRadius=earthRadius,
                                                   maxDelay1=maxDelay1[i], maxDelay2=maxDelay2[j])
                isConflict21 = getRawPointConflict(lat2[i], lon2[i], alt2[i], time2[i], lat1[j], lon1[j], alt1[j], time1[j],
                                                   spaceThreshold=spaceThreshold, timeThreshold=temporalThreshold,
                                                   altitudeThreshold=altitudeThreshold, earthRadius=earthRadius,
                                                   maxDelay1=maxDelay2[i], maxDelay2=maxDelay1[j])
                isConflict22 = getRawPointConflict(lat2[i], lon2[i], alt2[i], time2[i], lat2[j], lon2[j], alt2[j], time2[j],
                                                   spaceThreshold=spaceThreshold, timeThreshold=temporalThreshold,
                                                   altitudeThreshold=altitudeThreshold, earthRadius=earthRadius,
                                                   maxDelay1=maxDelay2[i], maxDelay2=maxDelay2[j])
                if isConflict11 or isConflict12 or isConflict21 or isConflict22:
                    # check if one of the raw point conflicts is a parallel conflict
                    multiConflictsFirst.push_back(conflictIndex[i])
                    multiConflictsSecond.push_back(conflictIndex[j])
                    conflictTypeFirst.push_back(conflictIndex[i] >= N1)
                    conflictTypeSecond.push_back(conflictIndex[j] >= N1)
                    conflict11.push_back(isConflict11)
                    conflict12.push_back(isConflict12)
                    conflict21.push_back(isConflict21)
                    conflict22.push_back(isConflict22)
                    # get minimum time difference
                    deltaTMin = 1E6
                    if isConflict11:
                        deltaT = np.abs(time1[i] - time1[j])
                        if deltaT < deltaTMin:
                            deltaTMin = deltaT
                    if isConflict12:
                        deltaT = np.abs(time1[i] - time2[j])
                        if deltaT < deltaTMin:
                            deltaTMin = deltaT
                    if isConflict21:
                        deltaT = np.abs(time2[i] - time1[j])
                        if deltaT < deltaTMin:
                            deltaTMin = deltaT
                    if isConflict22:
                        deltaT = np.abs(time2[i] - time2[j])
                        if deltaT < deltaTMin:
                            deltaTMin = deltaT
                    deltaTMinVec.push_back(deltaTMin)
                    # check if the multi conflict is build out of point or parallel conflicts
                    if conflictIndex[i] < N1 and conflictIndex[j] < N1:
                        conflictType.push_back(0);
                    elif conflictIndex[i] >= N1 and conflictIndex[j] >= N1:
                        conflictType.push_back(1);
                    else:
                        conflictType.push_back(0.5);
    pbar.finish()

    multiConflicts = pd.DataFrame({'conflict1': np.array(multiConflictsFirst),
                                   'conflict2': np.array(multiConflictsSecond),
                                   'conflictType1': np.array(conflictTypeFirst),
                                   'conflictType2': np.array(conflictTypeSecond),
                                   'isConflict11': np.array(conflict11),
                                   'isConflict12': np.array(conflict12),
                                   'isConflict21': np.array(conflict21),
                                   'isConflict22': np.array(conflict22),
                                   'deltaTMin': np.array(deltaTMinVec),
                                   'multiConflictType': np.array(conflictType)
                                   })

    # enforce conflict1 < conflict2 and remove duplicated
    mc1 = multiConflicts[multiConflicts['conflict1'] > multiConflicts['conflict2']]
    mc1.columns = ['conflict2', 'conflict1', 'conflictType2', 'conflictType1', 'isConflict11', 'isConflict21', 'isConflict12', 'isConflict22', 'deltaTMin', 'multiConflictType']
    mc2 = multiConflicts[multiConflicts['conflict1'] <= multiConflicts['conflict2']]
    multiConflicts = pd.concat([mc1, mc2])
    multiConflicts.drop_duplicates(inplace=True)
    multiConflicts.sort_values(by=['conflict1', 'conflict2'], inplace=True)
    multiConflicts.index.name = 'multiConflictIndex'

    return multiConflicts
