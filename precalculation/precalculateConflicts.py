#!/usr/bin/env python
import pandas as pd
import numpy as np
import conflict
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Calculate point conflicts from trajectory data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', default='data/TrajDataV2_20120729.txt', help='input file containing the trajectory data')
    parser.add_argument('-d', '--mindistance', default=30, help='Minimum distance in nautic miles to qualify as a conflict', type=float)
    parser.add_argument('-t', '--mintime', default=21, help='Minimum time difference in minutes to qualify as a potential conflict', type=int)
    parser.add_argument('--delayPerConflict', default=0, help='Delay introduced by each conflict avoiding maneuver', type=int)
    parser.add_argument('--dthreshold', default=3, help='Minimum time difference in minutes to qualify as a real conflict', type=int)
    parser.add_argument('--maxDepartDelay', default=18, help='Maximum departure delay', type=int)
    parser.add_argument('--maxIter', default=50, help='Maximal number of iterations used in reduction of conflicts', type=int)
    parser.add_argument('--use_snapshots', action='store_true', help='Force recalulation of intermediate step of caluclatin consecutive flight indices in the data')
    parser.add_argument('--multi', action='store_true', help='Calculate non-pairwise conflicts')
    parser.add_argument('--coarseGridLat', default=2, help='Latitude grid step in degrees for coarse grid during raw conflict detection', type=float)
    parser.add_argument('--coarseGridLon', default=2, help='Longitude grid step in degrees for coarse grid during raw conflict detection', type=float)
    args = parser.parse_args()

    precalculateConflicts(inputFile=args.input,
                          mindistance=args.mindistance,
                          mintime=args.mintime,
                          delayPerConflict=args.delayPerConflict,
                          dthreshold=args.dthreshold,
                          maxDepartDelay=args.maxDepartDelay,
                          maxIter=args.maxIter,
                          use_snapshots=args.use_snapshots,
                          coarseGridLat=args.coarseGridLat,
                          coarseGridLon=args.coarseGridLon,
                          multi=args.multi)

def precalculateConflicts(inputFile, mindistance, mintime, delayPerConflict, dthreshold, maxDepartDelay, maxIter, use_snapshots, multi, coarseGridLat, coarseGridLon):
    # nautic mile in kilometers
    nautic = 1.852

    inputDataFile = inputFile
    filename = "%s.mindist%05.1f_mintime%03i" % (inputDataFile, mindistance, mintime)
    trajectoryFile = inputFile + ".csv"
    trajectories = None

    # minimal acceptable distance in kilometers
    mindistance = mindistance * nautic
    if not os.path.exists(trajectoryFile) or not use_snapshots:
        print "Read in trajectories ..."
        trajectories = pd.read_csv(inputDataFile,
                                   delimiter="\t",
                                   names=('flight', 'date', 'wind', 'time', 'speed', 'altitude', 'latitude', 'longitude', 'nan'),
                                   usecols=('flight', 'date', 'wind', 'time', 'speed', 'altitude', 'latitude', 'longitude')
                                   )
        print "Change latitude and longitude to radians"
        trajectories['latitude'] = trajectories['latitude'].apply(np.radians)
        trajectories['longitude'] = trajectories['longitude'].apply(np.radians)
        print "Add consecutive flight index to data ..."
        # add consecutive flight index to the data
        flightNames = trajectories['flight'].unique()
        trajectories['flightIndex'] = trajectories['flight'].map(lambda x: np.where(flightNames == x)[0][0])
        # set consecutive flight index as dataset index
        trajectories = trajectories.set_index('flightIndex')

        trajectories.to_csv(trajectoryFile, mode='w')
    else:
        print "read in trajectories ..."
        trajectories = pd.read_csv(trajectoryFile)

    # calulate point conflicts
    rawPointConflictFile = filename + ".rawPointConflicts.csv"
    if not os.path.exists(rawPointConflictFile) or not use_snapshots:
        rawPointConflicts = conflict.detectRawConflicts(trajectories.index, trajectories.time, trajectories.latitude, trajectories.longitude, trajectories.altitude, mindistance, mintime, coarseGridLat, coarseGridLon)
        # save to csv file
        rawPointConflicts.to_csv(rawPointConflictFile, mode='w')
        print "Point conflict data written to", rawPointConflictFile
    else:
        rawPointConflicts = pd.read_csv(rawPointConflictFile, index_col='conflictIndex')
    print rawPointConflicts.shape[0], "raw point conflicts detected"

    # calulate point conflicts
    print "parse conflicts ..."
    pointConflictFile = filename + ".pointConflicts.csv"
    parallelConflictFile = filename + ".parallelConflicts.csv"
    if not os.path.exists(pointConflictFile) or not os.path.exists(parallelConflictFile) or not use_snapshots:
        pointConflicts, parallelConflicts = conflict.parseRawPointConflicts(rawPointConflicts, deltaT=2)
        pointConflicts.to_csv(pointConflictFile, mode='w')
        print "Point conflict data written to", pointConflictFile
        parallelConflicts.to_csv(parallelConflictFile, mode='w')
        print "Parallel conflict data written to", parallelConflictFile
    else:
        pointConflicts = pd.read_csv(pointConflictFile, index_col='conflictIndex')
        parallelConflicts = pd.read_csv(parallelConflictFile, index_col='parallelConflict')
    print pointConflicts.shape[0], "point conflicts identified"
    print len(parallelConflicts.index.unique()), "parallel conflicts involving", parallelConflicts.shape[0], "trajectory points identified"

    # calulate mapping of flight index to temporal sorted conflicts and reduce number of conflicts
    flights2ConflictsFile = filename + ".flights2Conflicts_delay%03i_thres%03i_depart%03i.csv" % (delayPerConflict, dthreshold, maxDepartDelay)
    reducedPointConflictFile = filename + ".reducedPointConflicts_delay%03i_thres%03i_depart%03i.csv" % (delayPerConflict, dthreshold, maxDepartDelay)
    reducedParallelConflictFile = filename + ".reducedParallelConflicts_delay%03i_thres%03i_depart%03i.csv" % (delayPerConflict, dthreshold, maxDepartDelay)
    if not os.path.exists(flights2ConflictsFile) or not os.path.exists(reducedPointConflictFile) or not os.path.exists(reducedParallelConflictFile) or not use_snapshots:
        if delayPerConflict == 0:
            print "Reduction not necessary since delayPerConflict is 0. Calculate flight2Conflict though."
            flights2Conflicts = conflict.getFlightConflicts(pointConflicts, parallelConflicts)
            flights2Conflicts.to_csv(flights2ConflictsFile, mode='w')
        else:
            diff = 1
            count = 0
            iterMax = maxIter
            logfile = filename + ".reduceConflicts_delay%03i_thres%03i_depart%03i.log" % (delayPerConflict, dthreshold, maxDepartDelay)
            f = open(logfile, 'w')
            f.write('# count\tnumber of conflicts\n')
            while not diff == 0 and count < iterMax:
                flights2Conflicts = conflict.getFlightConflicts(pointConflicts, parallelConflicts)
                NConflicts = len(pointConflicts) + len(parallelConflicts)
                pointConflicts, parallelConflicts = conflict.reduceConflicts(flights2Conflicts, pointConflicts, parallelConflicts, delayPerConflict, dthreshold, maxDepartDelay)
                NConflictsNew = len(pointConflicts) + len(parallelConflicts)
                diff = NConflicts - NConflictsNew
                f.write('%i\t%i\n' % (count, NConflicts))
                count += 1
                print "Iteration:", count, ". Number of Conflicts:", NConflicts
            if count == iterMax:
                f.close()
                print "No convergence. Break."
                exit(1)
            else:
                print "Convergence after", count, "iterations. Number of Conflicts:", NConflicts
            f.close()

        print "Flight to conflict data written to", flights2ConflictsFile
        flights2Conflicts.to_csv(flights2ConflictsFile)
        pointConflicts.to_csv(reducedPointConflictFile, mode='w')
        print "Reduced point conflict data written to", reducedPointConflictFile
        parallelConflicts.to_csv(reducedParallelConflictFile, mode='w')
        print "Parallel conflict data written to", reducedParallelConflictFile
    else:
        flights2Conflicts = pd.read_csv(flights2ConflictsFile, index_col='flight')
        pointConflicts = pd.read_csv(reducedPointConflictFile, index_col='conflictIndex')
        parallelConflicts = pd.read_csv(reducedParallelConflictFile, index_col='parallelConflict')
    print pointConflicts.shape[0], "reduced point conflicts identified"
    print len(parallelConflicts.index.unique()), "reduced parallel conflicts involving", parallelConflicts.shape[0], "trajectory points identified"

    if multi:
        multiConflictFile = filename + ".multiConflicts_delay%03i_thres%03i_depart%03i.csv" % (delayPerConflict, dthreshold, maxDepartDelay)
        if not os.path.exists(multiConflictFile) or not use_snapshots:
            multiConflicts = conflict.getMultiConflicts(pointConflicts, parallelConflicts, flights2Conflicts, mindistance, delayPerConflict, dthreshold, maxDepartDelay)
            # save to csv file
            multiConflicts.to_csv(multiConflictFile, mode='w')
            print "Multi conflict data written to", multiConflictFile
        else:
            multiConflicts = pd.read_csv(multiConflictFile, index_col='multiConflictIndex')
        print multiConflicts.shape[0], "non-pairwise conflicts identified"
if __name__ == "__main__":
    main()
